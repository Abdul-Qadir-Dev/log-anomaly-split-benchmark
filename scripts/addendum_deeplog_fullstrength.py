"""Step 10: full-strength DeepLog robustness run, HDFS_v1 only, seed 0,
RANDOM and CHRONO. Larger model/budget than the main run's lightweight config:
window cap 1,000,000 (or all available if fewer), up to 15 epochs with early
stopping on validation loss (patience 3), hidden 128, 2 layers, same top-k=9
scoring. Hard cap 90 minutes wall-clock per split; if exceeded, stop at the
last completed epoch and record it. Robustness only -- no verdict depends on
this, and results/full/ (including the main-run deeplog cells) are untouched.
Writes to results/full/addendum/deeplog_fullstrength/.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from context import ArmContext
from dataset_paths import processed_path
from evaluate import evaluate_cell

SEED = 0
WINDOW = 10
TOP_K = 9
HIDDEN = 128
LAYERS = 2
MAX_WINDOWS = 1_000_000
MAX_EPOCHS = 15
PATIENCE = 3
BATCH = 256
EVAL_BATCH = 4096
WALLCLOCK_CAP_SECONDS = 90 * 60


class DeepLogNetBig(nn.Module):
    def __init__(self, vocab_size, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        e = self.embed(x)
        out, _ = self.lstm(e)
        return self.fc(out[:, -1, :])


def make_windows(sessions, pad_id):
    X, y = [], []
    for seq in sessions["template_seq"]:
        s = list(seq)
        if len(s) == 0:
            continue
        padded = [pad_id] * WINDOW + s
        for i in range(len(s)):
            X.append(padded[i:i + WINDOW])
            y.append(padded[i + WINDOW])
    return X, y


def score_sessions(model, sessions, pad_id, device):
    model.eval()
    windows, targets, owner = [], [], []
    for i, seq in enumerate(sessions["template_seq"]):
        s = list(seq)
        if len(s) == 0:
            continue
        padded = [pad_id] * WINDOW + s
        for j in range(len(s)):
            windows.append(padded[j:j + WINDOW])
            targets.append(padded[j + WINDOW])
            owner.append(i)

    scores = np.zeros(len(sessions), dtype="float32")
    counts = np.zeros(len(sessions), dtype="float32")
    n = len(windows)
    if n == 0:
        return scores
    Xt = torch.tensor(windows, dtype=torch.long)
    targets = np.array(targets)
    owner = np.array(owner)
    miss = np.zeros(n, dtype=bool)
    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH):
            end = min(start + EVAL_BATCH, n)
            logits = model(Xt[start:end].to(device))
            k = min(TOP_K, logits.shape[1])
            topk = torch.topk(logits, k=k, dim=1).indices.cpu().numpy()
            batch_targets = targets[start:end]
            for r in range(end - start):
                miss[start + r] = batch_targets[r] not in topk[r]
    np.add.at(counts, owner, 1.0)
    np.add.at(scores, owner, miss.astype("float32"))
    return np.divide(scores, counts, out=np.zeros_like(scores), where=counts > 0)


def val_loss(model, val_windows_X, val_windows_y, device, loss_fn, batch=4096):
    model.eval()
    Xt = torch.tensor(val_windows_X, dtype=torch.long)
    yt = torch.tensor(val_windows_y, dtype=torch.long)
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(Xt), batch):
            end = min(start + batch, len(Xt))
            logits = model(Xt[start:end].to(device))
            loss = loss_fn(logits, yt[start:end].to(device))
            total_loss += loss.item() * (end - start)
            total_n += (end - start)
    return total_loss / total_n if total_n else float("inf")


def run_split(dataset, split, out_dir):
    final_path = out_dir / f"{dataset}_{split}.json"
    if final_path.exists():
        print(f"{dataset}/{split} already has a final result; skipping.", flush=True)
        with open(final_path) as f:
            return json.load(f)

    ckpt_path = out_dir / f"{dataset}_{split}_checkpoint.pt"

    sessions = pd.read_parquet(processed_path(dataset, "sessions.parquet", False)).reset_index(drop=True)
    n_vocab = len(pd.read_parquet(processed_path(dataset, "templates.parquet", False)))
    d = Path("results/full") / dataset / split / f"seed{SEED}"
    with open(d / "train_ids.json") as f:
        train_ids = set(json.load(f))
    with open(d / "val_ids.json") as f:
        val_ids = set(json.load(f))
    with open(d / "test_ids.json") as f:
        test_ids = set(json.load(f))

    train = sessions[sessions["session_id"].isin(train_ids)]
    train_normal = train[train["label"] == 0].reset_index(drop=True)
    val = sessions[sessions["session_id"].isin(val_ids)].reset_index(drop=True)
    val_normal = val[val["label"] == 0].reset_index(drop=True)
    test = sessions[sessions["session_id"].isin(test_ids)].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_id = n_vocab

    X, y = make_windows(train_normal, pad_id)
    n_avail = len(X)
    rng = np.random.RandomState(SEED)
    if n_avail > MAX_WINDOWS:
        idx = rng.choice(n_avail, MAX_WINDOWS, replace=False)
        X = [X[i] for i in idx]
        y = [y[i] for i in idx]
    n_windows_used = len(X)

    val_X, val_y = make_windows(val_normal, pad_id)

    ds = TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = DeepLogNetBig(n_vocab).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_log = []
    start_epoch = 1
    cumulative_seconds_before = 0.0
    stopped_reason = "completed_all_epochs"
    last_completed_epoch = 0

    # This process is capped well under the 90-minute wall-clock budget, so a long
    # training run must survive being killed and restarted. Checkpoint
    # after every epoch and resume from it, carrying the *real* cumulative training time
    # forward so the 90-minute cap is enforced against actual compute, not wall-clock
    # including time spent not running.
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        best_val_loss = ckpt["best_val_loss"]
        best_state = ckpt["best_state"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        epoch_log = ckpt["epoch_log"]
        start_epoch = ckpt["last_completed_epoch"] + 1
        cumulative_seconds_before = ckpt["cumulative_seconds"]
        last_completed_epoch = ckpt["last_completed_epoch"]
        print(f"{dataset}/{split}: resumed from checkpoint at epoch {start_epoch} "
              f"({cumulative_seconds_before:.0f}s already spent)", flush=True)

    t0 = time.time()

    def cumulative():
        return cumulative_seconds_before + (time.time() - t0)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if cumulative() > WALLCLOCK_CAP_SECONDS:
            stopped_reason = f"wallclock_cap_exceeded_before_epoch_{epoch}"
            break
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        vl = val_loss(model, val_X, val_y, device, loss_fn)
        epoch_time = cumulative()
        epoch_log.append({"epoch": epoch, "val_loss": vl, "cumulative_seconds": epoch_time})
        print(f"{dataset}/{split} epoch {epoch}: val_loss={vl:.4f} cumulative={epoch_time:.0f}s", flush=True)
        last_completed_epoch = epoch

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        torch.save({
            "model_state": model.state_dict(), "opt_state": opt.state_dict(),
            "best_val_loss": best_val_loss, "best_state": best_state,
            "epochs_no_improve": epochs_no_improve, "epoch_log": epoch_log,
            "last_completed_epoch": last_completed_epoch, "cumulative_seconds": epoch_time,
        }, ckpt_path)

        if epochs_no_improve >= PATIENCE:
            stopped_reason = f"early_stopping_at_epoch_{epoch}_patience_{PATIENCE}"
            break
        if cumulative() > WALLCLOCK_CAP_SECONDS:
            stopped_reason = f"wallclock_cap_exceeded_after_epoch_{epoch}"
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    total_fit_seconds = cumulative()

    score_val = np.asarray(score_sessions(model, val, pad_id, device), dtype="float64")
    score_test = np.asarray(score_sessions(model, test, pad_id, device), dtype="float64")
    metrics = evaluate_cell(val["label"].to_numpy(), score_val, test["label"].to_numpy(), score_test)
    metrics.update({
        "dataset": dataset, "split": split, "seed": SEED,
        "config": {"max_windows": MAX_WINDOWS, "windows_used": n_windows_used, "n_windows_available": n_avail,
                   "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "hidden": HIDDEN, "layers": LAYERS,
                   "top_k": TOP_K},
        "fit_seconds": total_fit_seconds,
        "last_completed_epoch": last_completed_epoch,
        "stopped_reason": stopped_reason,
        "best_val_loss": best_val_loss,
        "epoch_log": epoch_log,
    })

    out_path = out_dir / f"{dataset}_{split}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    if ckpt_path.exists():
        ckpt_path.unlink()  # final result saved; no need to keep the (large) model checkpoint
    print(f"{dataset}/{split}: AUPRC={metrics['auprc']:.4f} AUROC={metrics['auroc']:.4f} "
          f"fit={total_fit_seconds:.0f}s stopped={stopped_reason}", flush=True)
    return metrics


def main():
    out_dir = Path("results/full/addendum/deeplog_fullstrength")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split in ["random", "chrono"]:
        results[split] = run_split("HDFS_v1", split, out_dir)

    main_df = pd.read_csv("results/full/summary.csv")
    main_deeplog = main_df[(main_df.dataset == "HDFS_v1") & (main_df.arm == "deeplog") & (main_df.seed == SEED)]

    lines = ["# Full-strength DeepLog robustness run (Step 10)\n",
             "HDFS_v1 only, seed 0, RANDOM and CHRONO. Config: window cap 1,000,000, up to 15 epochs "
             "with early stopping (patience 3) on validation loss, hidden 128, 2 layers, top-k=9 scoring. "
             "90-minute wall-clock cap per split. Robustness only -- no verdict depends on this.\n",
             "| Split | Config | AUROC | AUPRC | F1_val | F1_oracle | Epochs run | Stop reason | Fit time |",
             "|---|---|---|---|---|---|---|---|---|"]
    for split in ["random", "chrono"]:
        r = results[split]
        lines.append(f"| {split} | full-strength | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['f1_val']:.4f} | "
                     f"{r['f1_oracle']:.4f} | {r['last_completed_epoch']} | {r['stopped_reason']} | "
                     f"{r['fit_seconds']:.0f}s |")
        m = main_deeplog[main_deeplog.split == split]
        if len(m):
            mr = m.iloc[0]
            lines.append(f"| {split} | main-run (100k windows, 5 epochs, hidden 64) | {mr.auroc:.4f} | "
                         f"{mr.auprc:.4f} | {mr.f1_val:.4f} | {mr.f1_oracle:.4f} | 5 | n/a (fixed) | "
                         f"{mr.fit_seconds:.0f}s |")

    with open(out_dir / "deeplog_fullstrength_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_dir / 'deeplog_fullstrength_table.md'}")


if __name__ == "__main__":
    main()
