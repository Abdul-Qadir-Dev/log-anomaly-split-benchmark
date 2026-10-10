"""DeepLog: LSTM next-template prediction anomaly score (Du et al. 2017).

Step 11 correction: a diagnostic review found the original top-k-miss scoring
heuristic saturates almost completely on HDFS_v1's small (55-template)
vocabulary (>97% of sessions score exactly 0 at k=9, confirmed in both this
arm's original lightweight config and a much larger model) -- so the primary
score is now mean negative log-likelihood (continuous, does not saturate),
with top-9 miss-rate kept as a secondary/diagnostic score. Early stopping on
normal-only validation loss was also added (the original fixed-5-epoch
schedule had no early-stopping mechanism at all). See DECISIONS.md "Step 11"
and results/full/addendum/deeplog_diagnosis/DIAGNOSIS.md for the full
analysis that led to this fix.

Step 13: training is now fully seeded from the cell's seed (random, numpy,
torch CPU and CUDA RNGs, plus cudnn determinism flags) at the start of
fit(). The released results/full/ deeplog cells were trained before this
fix and are not bit-reproducible (see DECISIONS.md "Step 13" and
REPRODUCE.md); they were not re-run, since the fix only affects
reproducibility, not the score computed for any given training run.
"""
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WINDOW = 10
TOP_K = 9
HIDDEN = 64
LAYERS = 2
MAX_WINDOWS = 100_000
MAX_EPOCHS = 15
PATIENCE = 3
BATCH = 256
EVAL_BATCH = 4096


class _DeepLogNet(nn.Module):
    def __init__(self, vocab_size, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, hidden)  # +1 for PAD id == vocab_size
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        e = self.embed(x)
        out, _ = self.lstm(e)
        return self.fc(out[:, -1, :])


def _make_windows(sessions, pad_id):
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


def _avg_loss(model, X, y, device, loss_fn, batch=EVAL_BATCH):
    if len(X) == 0:
        return float("inf")
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
    return total_loss / total_n if total_n else float("inf")


class DeepLogArm:
    name = "deeplog"

    def fit(self, ctx, train_sessions, val_sessions=None):
        random.seed(ctx.seed)
        np.random.seed(ctx.seed)
        torch.manual_seed(ctx.seed)
        torch.cuda.manual_seed_all(ctx.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        self.pad_id_ = ctx.n_vocab
        X, y = _make_windows(train_sessions, self.pad_id_)
        n = len(X)
        if n > MAX_WINDOWS:
            rng = np.random.RandomState(ctx.seed)
            idx = rng.choice(n, MAX_WINDOWS, replace=False)
            X = [X[i] for i in idx]
            y = [y[i] for i in idx]
        if len(X) == 0:
            X, y = [[self.pad_id_] * WINDOW], [0]

        ds = TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long))
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

        self.model_ = _DeepLogNet(ctx.n_vocab).to(self.device_)
        opt = torch.optim.Adam(self.model_.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Early stopping uses normal-only validation sessions, matching the training
        # objective (the model is trained on normal sessions only).
        val_X, val_y = [], []
        if val_sessions is not None and len(val_sessions) > 0:
            val_normal = val_sessions[val_sessions["label"] == 0] if "label" in val_sessions else val_sessions
            if len(val_normal) > 0:
                val_X, val_y = _make_windows(val_normal, self.pad_id_)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for _epoch in range(MAX_EPOCHS):
            self.model_.train()
            for xb, yb in dl:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                opt.step()

            if val_X:
                vl = _avg_loss(self.model_, val_X, val_y, self.device_, loss_fn)
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= PATIENCE:
                        break
            # No validation data (e.g. a tiny --selftest split with 0 normal val
            # sessions): fall back to running the full fixed-epoch schedule.

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _nll_and_topk(self, sessions):
        self.model_.eval()
        windows, targets, owner = [], [], []
        for i, seq in enumerate(sessions["template_seq"]):
            s = list(seq)
            if len(s) == 0:
                continue
            padded = [self.pad_id_] * WINDOW + s
            for j in range(len(s)):
                windows.append(padded[j:j + WINDOW])
                targets.append(padded[j + WINDOW])
                owner.append(i)

        n = len(windows)
        nll_scores = np.zeros(len(sessions), dtype="float64")
        topk_scores = np.zeros(len(sessions), dtype="float64")
        counts = np.zeros(len(sessions), dtype="float64")
        if n == 0:
            return nll_scores, topk_scores

        Xt = torch.tensor(windows, dtype=torch.long)
        targets_arr = np.array(targets)
        targets_t = torch.tensor(targets_arr, dtype=torch.long)
        owner_arr = np.array(owner)
        nlls = np.zeros(n, dtype="float64")
        miss = np.zeros(n, dtype=bool)
        with torch.no_grad():
            for start in range(0, n, EVAL_BATCH):
                end = min(start + EVAL_BATCH, n)
                logits = self.model_(Xt[start:end].to(self.device_))
                logp = torch.log_softmax(logits, dim=1)
                tgt = targets_t[start:end].to(self.device_)
                picked = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
                nlls[start:end] = (-picked).cpu().numpy()
                k = min(TOP_K, logits.shape[1])
                topk_idx = torch.topk(logits, k=k, dim=1).indices.cpu().numpy()
                batch_targets = targets_arr[start:end]
                for r in range(end - start):
                    miss[start + r] = batch_targets[r] not in topk_idx[r]

        np.add.at(counts, owner_arr, 1.0)
        np.add.at(nll_scores, owner_arr, nlls)
        np.add.at(topk_scores, owner_arr, miss.astype("float64"))
        nll_scores = np.divide(nll_scores, counts, out=np.zeros_like(nll_scores), where=counts > 0)
        topk_scores = np.divide(topk_scores, counts, out=np.zeros_like(topk_scores), where=counts > 0)
        return nll_scores, topk_scores

    def score(self, ctx, sessions):
        """Primary score is mean NLL (continuous; does not saturate). The secondary
        top-9 miss-rate score is stashed on the instance for callers that want it
        (e.g. run_all.py records it alongside the primary metrics for deeplog)."""
        nll_scores, topk_scores = self._nll_and_topk(sessions)
        self.last_topk_scores_ = topk_scores
        return nll_scores
