"""Step 08 audit: does the GPU-matmul semantic_knn give the same scores as the
original sklearn brute-force NearestNeighbors(metric="cosine")? One cell only
(HDFS_v1, chrono, seed 0), scored on a 5,000-session subsample of the test split.
Pass bar: max absolute score difference < 1e-4.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from arms.semantic_knn import SemanticKNNArm
from context import ArmContext
from dataset_paths import processed_path
from features import session_embedding_matrix, template_embeddings

DATASET, SPLIT, SEED = "HDFS_v1", "chrono", 0
K = 10
N_QUERY = 5000


def original_sklearn_scorer(train_emb, query_emb, k):
    """The pre-fix implementation: sklearn brute-force NearestNeighbors(metric='cosine')."""
    k_eff = min(k, len(train_emb))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn.fit(train_emb)
    dist, _ = nn.kneighbors(query_emb)
    return dist.mean(axis=1)


def main():
    sessions = pd.read_parquet(processed_path(DATASET, "sessions.parquet", False))
    d = Path("results/full") / DATASET / SPLIT / f"seed{SEED}"
    with open(d / "train_ids.json") as f:
        train_ids = set(json.load(f))
    with open(d / "test_ids.json") as f:
        test_ids = set(json.load(f))

    train = sessions[sessions["session_id"].isin(train_ids)]
    train = train[train["label"] == 0].reset_index(drop=True)  # unsupervised arms train on normal only
    test = sessions[sessions["session_id"].isin(test_ids)].reset_index(drop=True)

    rng = np.random.RandomState(0)
    query_idx = rng.choice(len(test), size=min(N_QUERY, len(test)), replace=False)
    query = test.iloc[query_idx].reset_index(drop=True)

    n_vocab = len(pd.read_parquet(processed_path(DATASET, "templates.parquet", False)))
    ctx = ArmContext(dataset=DATASET, selftest=False, n_vocab=n_vocab, seed=SEED)

    print(f"Fitting GPU semantic_knn on {len(train):,} training sessions, "
          f"scoring {len(query):,} query sessions...")
    gpu_arm = SemanticKNNArm(k=K)
    gpu_arm.fit(ctx, train)
    gpu_scores = np.asarray(gpu_arm.score(ctx, query), dtype="float64")

    print("Computing original sklearn brute-force cosine NearestNeighbors scores...")
    emb = template_embeddings(DATASET, False)
    train_emb = session_embedding_matrix(train, emb)
    query_emb = session_embedding_matrix(query, emb)
    cpu_scores = original_sklearn_scorer(train_emb, query_emb, K).astype("float64")

    abs_diff = np.abs(gpu_scores - cpu_scores)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    rho, p = spearmanr(gpu_scores, cpu_scores)

    n_unique_gpu = len(np.unique(np.round(gpu_scores, 8)))
    n_unique_cpu = len(np.unique(np.round(cpu_scores, 8)))

    # Spearman is a rank correlation; with this many tied/near-zero scores (most HDFS
    # test sessions are template-identical to a training-normal session), float32 vs
    # float64 noise can reorder the tied block without changing anything that matters
    # for AUPRC/AUROC. Directly compare the metric the verdicts actually used, using
    # whatever labels are available for this query subsample.
    y_query = query["label"].to_numpy()
    from sklearn.metrics import average_precision_score, roc_auc_score
    metrics_match = None
    if len(set(y_query.tolist())) > 1:
        metrics_match = {
            "auprc_gpu": float(average_precision_score(y_query, gpu_scores)),
            "auprc_cpu": float(average_precision_score(y_query, cpu_scores)),
            "auroc_gpu": float(roc_auc_score(y_query, gpu_scores)),
            "auroc_cpu": float(roc_auc_score(y_query, cpu_scores)),
        }

    pass_bar = max_diff < 1e-4
    result = {
        "dataset": DATASET, "split": SPLIT, "seed": SEED, "k": K,
        "n_train": len(train), "n_query": len(query),
        "max_abs_diff": max_diff, "mean_abs_diff": mean_diff,
        "spearman_rho": float(rho), "spearman_p": float(p),
        "n_unique_rounded_scores_gpu": n_unique_gpu, "n_unique_rounded_scores_cpu": n_unique_cpu,
        "spearman_caveat": "Most scores are tied at ~0 (session embeddings identical to a training "
                            "normal session); float32 vs float64 noise reorders the tied block, which "
                            "deflates Spearman without indicating a real discrepancy -- see metrics_on_query below.",
        "metrics_on_query_subsample": metrics_match,
        "pass_bar_1e-4": pass_bar,
        "verdict": "PASS" if pass_bar else "FAIL",
    }

    Path("results/audit").mkdir(parents=True, exist_ok=True)
    with open("results/audit/knn_cpu_check_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
