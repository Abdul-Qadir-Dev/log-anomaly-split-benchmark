"""Semantic k-NN: mean-cosine-distance to nearest normal training sessions in embedding space.

Implemented as a batched, GPU-accelerated matmul (cosine similarity = dot product of
L2-normalised vectors) rather than sklearn's brute-force NearestNeighbors, which does not
scale to HDFS's ~335k-session training set (it was taking ~45 minutes per cell; see
DECISIONS.md, Step 04). Numerically this computes the identical quantity -- mean cosine
distance to the k nearest training session embeddings -- just with a faster backend.
"""
import numpy as np
import torch

from features import session_embedding_matrix, template_embeddings

K = 10
MAX_SIM_ELEMENTS = 20_000_000  # cap batch*n_train similarity-matrix elements (~80MB) to fit 6GB GPUs


def _l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class SemanticKNNArm:
    name = "semantic_knn"

    def __init__(self, k=K):
        self.k = k

    def fit(self, ctx, train_sessions):
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        self.template_emb_ = template_embeddings(ctx.dataset, ctx.selftest)
        train_emb = _l2_normalize(session_embedding_matrix(train_sessions, self.template_emb_))
        self.k_eff_ = min(self.k, len(train_sessions))
        self.train_t_ = torch.tensor(train_emb, dtype=torch.float32, device=self.device_)
        return self

    def score(self, ctx, sessions):
        q = _l2_normalize(session_embedding_matrix(sessions, self.template_emb_))
        qt = torch.tensor(q, dtype=torch.float32, device=self.device_)
        scores = np.zeros(len(sessions), dtype="float32")
        n_train = self.train_t_.shape[0]
        batch = max(1, min(len(sessions), MAX_SIM_ELEMENTS // max(1, n_train)))
        with torch.no_grad():
            for start in range(0, len(sessions), batch):
                end = min(start + batch, len(sessions))
                sims = qt[start:end] @ self.train_t_.T
                dist = 1.0 - sims
                topk = torch.topk(dist, k=self.k_eff_, largest=False, dim=1).values
                scores[start:end] = topk.mean(dim=1).cpu().numpy()
        return scores
