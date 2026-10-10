"""Step 11: DeepLog diagnosis. Investigates the Step 10 full-strength result
(RANDOM AUPRC falling from ~0.78 to 0.38 with a stronger model) before it is
accepted as a finding.

Retrains both configs (lightweight = main-run equivalent, fullstrength = Step
10's config) on both HDFS_v1 splits, seed 0, keeping the checkpoints this time
(Step 10 deleted them) so they can be inspected. results/full/ is read-only
here (split ids, session data) and is not modified; all outputs go to
results/full/addendum/deeplog_diagnosis/. Checkpoints are large binaries and
are deleted again at the end of this script, after every number needed for the
diagnosis has been extracted into JSON/markdown.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from context import ArmContext
from dataset_paths import processed_path
from evaluate import best_f1_threshold, evaluate_cell, f1_at_threshold

SEED = 0
WINDOW = 10
BATCH = 256
EVAL_BATCH = 4096
WALLCLOCK_CAP_SECONDS = 90 * 60

OUT_DIR = Path("results/full/addendum/deeplog_diagnosis")
CKPT_DIR = OUT_DIR / "checkpoints"

CONFIGS = {
    "lightweight": dict(hidden=64, layers=2, max_windows=100_000, max_epochs=5,
                         early_stop=False, patience=None),
    "fullstrength": dict(hidden=128, layers=2, max_windows=1_000_000, max_epochs=15,
                          early_stop=True, patience=3),
}
K_SWEEP = [1, 2, 3, 5, 7, 9]


class DeepLogNet(nn.Module):
    def __init__(self, vocab_size, hidden, layers):
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


def windows_for_sessions(sessions, pad_id):
    """Like make_windows but also returns an owner array mapping window -> session index."""
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
    return windows, targets, owner


def avg_loss(model, X, y, device, loss_fn, batch=EVAL_BATCH):
    if len(X) == 0:
        return float("nan")
    model.eval()
    Xt = torch.tensor(X, dtype=torch.long)
    yt = torch.tensor(y, dtype=torch.long)
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(Xt), batch):
            end = min(start + batch, len(Xt))
            logits = model(Xt[start:end].to(device))
            loss = loss_fn(logits, yt[start:end].to(device))
            total_loss += loss.item() * (end - start)
            total_n += (end - start)
    return total_loss / total_n if total_n else float("nan")


def state_dict_hash(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def topk_score(model, windows, targets, owner, n_sessions, device, k):
    n = len(windows)
    scores = np.zeros(n_sessions, dtype="float64")
    counts = np.zeros(n_sessions, dtype="float64")
    if n == 0:
        return scores
    Xt = torch.tensor(windows, dtype=torch.long)
    targets = np.array(targets)
    owner = np.array(owner)
    miss = np.zeros(n, dtype=bool)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH):
            end = min(start + EVAL_BATCH, n)
            logits = model(Xt[start:end].to(device))
            kk = min(k, logits.shape[1])
            topk_idx = torch.topk(logits, k=kk, dim=1).indices.cpu().numpy()
            batch_targets = targets[start:end]
            for r in range(end - start):
                miss[start + r] = batch_targets[r] not in topk_idx[r]
    np.add.at(counts, owner, 1.0)
    np.add.at(scores, owner, miss.astype("float64"))
    return np.divide(scores, counts, out=np.zeros_like(scores), where=counts > 0)


def nll_score(model, windows, targets, owner, n_sessions, device):
    n = len(windows)
    scores = np.zeros(n_sessions, dtype="float64")
    counts = np.zeros(n_sessions, dtype="float64")
    if n == 0:
        return scores
    Xt = torch.tensor(windows, dtype=torch.long)
    targets_t = torch.tensor(np.array(targets), dtype=torch.long)
    owner = np.array(owner)
    nlls = np.zeros(n, dtype="float64")
    model.eval()
    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH):
            end = min(start + EVAL_BATCH, n)
            logits = model(Xt[start:end].to(device))
            logp = torch.log_softmax(logits, dim=1)
            tgt = targets_t[start:end].to(device)
            picked = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            nlls[start:end] = (-picked).cpu().numpy()
    np.add.at(counts, owner, 1.0)
    np.add.at(scores, owner, nlls)
    return np.divide(scores, counts, out=np.zeros_like(scores), where=counts > 0)


def load_split(dataset, split):
    sessions = pd.read_parquet(processed_path(dataset, "sessions.parquet", False)).reset_index(drop=True)
    d = Path("results/full") / dataset / split / f"seed{SEED}"
    with open(d / "train_ids.json") as f:
        train_ids = set(json.load(f))
    with open(d / "val_ids.json") as f:
        val_ids = set(json.load(f))
    with open(d / "test_ids.json") as f:
        test_ids = set(json.load(f))
    train = sessions[sessions["session_id"].isin(train_ids)].reset_index(drop=True)
    val = sessions[sessions["session_id"].isin(val_ids)].reset_index(drop=True)
    test = sessions[sessions["session_id"].isin(test_ids)].reset_index(drop=True)
    return train, val, test


def train_model(dataset, split, config_name, out_dir):
    """Train (or resume) one (config, split) model, tracking train/val loss every epoch
    and using NORMAL-ONLY validation sessions for both early stopping (if enabled) and
    the diagnostic val-loss trajectory, matching what Step 10's full-strength script did.
    """
    cfg = CONFIGS[config_name]
    meta_path = out_dir / f"{dataset}_{split}_{config_name}_meta.json"
    ckpt_path = CKPT_DIR / f"{dataset}_{split}_{config_name}.pt"
    if meta_path.exists():
        print(f"{dataset}/{split}/{config_name}: meta already exists, loading.", flush=True)
        with open(meta_path) as f:
            return json.load(f)

    train, val, test = load_split(dataset, split)
    train_normal = train[train["label"] == 0].reset_index(drop=True)
    val_normal = val[val["label"] == 0].reset_index(drop=True)
    n_vocab = len(pd.read_parquet(processed_path(dataset, "templates.parquet", False)))
    pad_id = n_vocab
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y = make_windows(train_normal, pad_id)
    n_avail = len(X)
    rng = np.random.RandomState(SEED)
    if n_avail > cfg["max_windows"]:
        idx = rng.choice(n_avail, cfg["max_windows"], replace=False)
        X = [X[i] for i in idx]
        y = [y[i] for i in idx]
    n_used = len(X)

    val_X, val_y = make_windows(val_normal, pad_id)

    ds = TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = DeepLogNet(n_vocab, cfg["hidden"], cfg["layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_log = []
    start_epoch = 1
    cumulative_before = 0.0
    last_completed_epoch = 0
    stopped_reason = "completed_all_epochs"

    resume_ckpt = CKPT_DIR / f"{dataset}_{split}_{config_name}_resume.pt"
    if resume_ckpt.exists():
        ck = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt_state"])
        best_val_loss = ck["best_val_loss"]
        best_state = ck["best_state"]
        epochs_no_improve = ck["epochs_no_improve"]
        epoch_log = ck["epoch_log"]
        start_epoch = ck["last_completed_epoch"] + 1
        cumulative_before = ck["cumulative_seconds"]
        last_completed_epoch = ck["last_completed_epoch"]
        print(f"{dataset}/{split}/{config_name}: resumed at epoch {start_epoch}", flush=True)

    t0 = time.time()

    def cumulative():
        return cumulative_before + (time.time() - t0)

    for epoch in range(start_epoch, cfg["max_epochs"] + 1):
        if cumulative() > WALLCLOCK_CAP_SECONDS:
            stopped_reason = f"wallclock_cap_exceeded_before_epoch_{epoch}"
            break
        model.train()
        total_train_loss, total_train_n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_train_loss += loss.item() * len(xb)
            total_train_n += len(xb)
        train_loss_epoch = total_train_loss / total_train_n if total_train_n else float("nan")
        vl = avg_loss(model, val_X, val_y, device, loss_fn)
        epoch_time = cumulative()
        epoch_log.append({"epoch": epoch, "train_loss": train_loss_epoch, "val_loss": vl,
                           "cumulative_seconds": epoch_time})
        print(f"{dataset}/{split}/{config_name} epoch {epoch}: train_loss={train_loss_epoch:.4f} "
              f"val_loss={vl:.4f} cumulative={epoch_time:.0f}s", flush=True)
        last_completed_epoch = epoch

        improved = vl < best_val_loss
        if improved:
            best_val_loss = vl
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(), "opt_state": opt.state_dict(),
            "best_val_loss": best_val_loss, "best_state": best_state,
            "epochs_no_improve": epochs_no_improve, "epoch_log": epoch_log,
            "last_completed_epoch": last_completed_epoch, "cumulative_seconds": epoch_time,
        }, resume_ckpt)

        if cfg["early_stop"]:
            if epochs_no_improve >= cfg["patience"]:
                stopped_reason = f"early_stopping_at_epoch_{epoch}_patience_{cfg['patience']}"
                break
        if cumulative() > WALLCLOCK_CAP_SECONDS:
            stopped_reason = f"wallclock_cap_exceeded_after_epoch_{epoch}"
            break

    # For the lightweight (main-run-equivalent) config there is no early stopping: the
    # final epoch's weights are what the main run actually used and scored, so don't
    # swap in a different "best" checkpoint for it. For fullstrength, match Step 10's
    # behavior and use the best-val-loss checkpoint.
    if cfg["early_stop"] and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    total_fit_seconds = cumulative()
    n_params = sum(p.numel() for p in model.parameters())
    train_loss_final = epoch_log[-1]["train_loss"] if epoch_log else float("nan")
    hash_before_scoring = state_dict_hash(model)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    if resume_ckpt.exists():
        resume_ckpt.unlink()

    meta = {
        "dataset": dataset, "split": split, "config": config_name,
        "hidden": cfg["hidden"], "layers": cfg["layers"], "n_params": int(n_params),
        "max_windows": cfg["max_windows"], "n_windows_used": n_used, "n_windows_available": n_avail,
        "max_epochs": cfg["max_epochs"], "early_stop_enabled": cfg["early_stop"], "patience": cfg["patience"],
        "last_completed_epoch": last_completed_epoch, "stopped_reason": stopped_reason,
        "train_loss_final": train_loss_final, "best_val_loss": best_val_loss,
        "epoch_log": epoch_log, "fit_seconds": total_fit_seconds,
        "validation_set_used": "normal_only (label==0 validation sessions)",
        "state_dict_hash_after_training": hash_before_scoring,
        "pad_id": pad_id, "n_vocab": n_vocab,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{dataset}/{split}/{config_name}: DONE params={n_params} epochs={last_completed_epoch} "
          f"stopped={stopped_reason} train_loss={train_loss_final:.4f} val_loss={best_val_loss:.4f}", flush=True)
    return meta


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for config_name in ["lightweight", "fullstrength"]:
        for split in ["random", "chrono"]:
            train_model("HDFS_v1", split, config_name, OUT_DIR)

    print("All 4 models trained/loaded. Training-integrity phase complete.", flush=True)


if __name__ == "__main__":
    main()
