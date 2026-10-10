"""Step 09 audit: within-dataset test of the template-drift mechanism.

Diagnostic re-score only (results/full/ stays frozen) -- fits a fresh semantic_knn
on the existing BGL/CHRONO/seed0 split and compares its scores to per-session
unseen-template rate. Outputs to results/audit/drift_mechanism/.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from arms.semantic_knn import SemanticKNNArm
from context import ArmContext
from dataset_paths import processed_path

SEED = 0


def load_split(dataset, split, seed):
    sessions = pd.read_parquet(processed_path(dataset, "sessions.parquet", False))
    d = Path("results/full") / dataset / split / f"seed{seed}"
    with open(d / "train_ids.json") as f:
        train_ids = set(json.load(f))
    with open(d / "test_ids.json") as f:
        test_ids = set(json.load(f))
    train = sessions[sessions["session_id"].isin(train_ids)].reset_index(drop=True)
    test = sessions[sessions["session_id"].isin(test_ids)].reset_index(drop=True)
    return train, test


def per_session_unseen_rate(train_normal, test):
    train_templates = set()
    for seq in train_normal["template_seq"]:
        train_templates.update(seq)
    rates = np.zeros(len(test), dtype="float64")
    for i, seq in enumerate(test["template_seq"]):
        if len(seq) == 0:
            rates[i] = 0.0
            continue
        unseen = sum(1 for t in seq if t not in train_templates)
        rates[i] = unseen / len(seq)
    return rates, train_templates


def normal_vs_anomalous_unseen_rate(dataset, out_dir):
    train, test = load_split(dataset, "chrono", SEED)
    train_normal = train[train["label"] == 0].reset_index(drop=True)
    rates, _ = per_session_unseen_rate(train_normal, test)

    y = test["label"].to_numpy()
    normal_rates = rates[y == 0]
    anom_rates = rates[y == 1]
    auroc_unseen_alone = float(roc_auc_score(y, rates)) if len(set(y.tolist())) > 1 else float("nan")

    result = {
        "dataset": dataset, "split": "chrono", "seed": SEED,
        "n_test": len(test), "n_normal": int((y == 0).sum()), "n_anomalous": int((y == 1).sum()),
        "mean_unseen_rate_normal": float(normal_rates.mean()),
        "mean_unseen_rate_anomalous": float(anom_rates.mean()),
        "auroc_unseen_rate_alone": auroc_unseen_alone,
    }
    with open(out_dir / f"{dataset}_unseen_rate_normal_vs_anomalous.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result, rates, test


def main():
    out_dir = Path("results/audit/drift_mechanism")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== BGL/CHRONO/seed0: full mechanism test ===")
    bgl_result, bgl_rates, bgl_test = normal_vs_anomalous_unseen_rate("BGL", out_dir)

    # Diagnostic re-score with semantic_knn on the same frozen split (results/full untouched).
    train, test = load_split("BGL", "chrono", SEED)
    train_normal = train[train["label"] == 0].reset_index(drop=True)
    n_vocab = len(pd.read_parquet(processed_path("BGL", "templates.parquet", False)))
    ctx = ArmContext(dataset="BGL", selftest=False, n_vocab=n_vocab, seed=SEED)
    arm = SemanticKNNArm(k=10)
    arm.fit(ctx, train_normal)
    sem_scores = np.asarray(arm.score(ctx, test), dtype="float64")

    y = test["label"].to_numpy()
    normal_mask = y == 0
    anom_mask = y == 1

    rho_normal, p_normal = spearmanr(bgl_rates[normal_mask], sem_scores[normal_mask])
    rho_anom, p_anom = spearmanr(bgl_rates[anom_mask], sem_scores[anom_mask])
    rho_all, p_all = spearmanr(bgl_rates, sem_scores)

    semantic_result = {
        "dataset": "BGL", "split": "chrono", "seed": SEED,
        "spearman_unseen_vs_semantic_score_normal_only": {"rho": float(rho_normal), "p": float(p_normal), "n": int(normal_mask.sum())},
        "spearman_unseen_vs_semantic_score_anomalous_only": {"rho": float(rho_anom), "p": float(p_anom), "n": int(anom_mask.sum())},
        "spearman_unseen_vs_semantic_score_all": {"rho": float(rho_all), "p": float(p_all), "n": len(y)},
        "semantic_knn_auroc_this_rescore": float(roc_auc_score(y, sem_scores)),
    }
    with open(out_dir / "BGL_semantic_knn_vs_unseen_rate.json", "w") as f:
        json.dump(semantic_result, f, indent=2)
    print(json.dumps(semantic_result, indent=2))

    print("\n=== HDFS_v1/CHRONO/seed0: normal-vs-anomalous unseen rate only ===")
    hdfs_result, _, _ = normal_vs_anomalous_unseen_rate("HDFS_v1", out_dir)

    report = ["# Template-drift mechanism test (Step 09 audit)\n",
              "Diagnostic re-score only -- results/full/ is untouched; this re-fits semantic_knn on the "
              "existing BGL/CHRONO/seed0 split independently to compare against per-session unseen-template rate.\n",
              "## BGL/CHRONO/seed0\n",
              f"- Mean per-session unseen-template rate, normal test sessions: **{bgl_result['mean_unseen_rate_normal']:.4f}**",
              f"- Mean per-session unseen-template rate, anomalous test sessions: **{bgl_result['mean_unseen_rate_anomalous']:.4f}**",
              f"- AUROC of unseen-rate-alone as an anomaly score: **{bgl_result['auroc_unseen_rate_alone']:.4f}**",
              f"- Spearman(unseen rate, semantic_knn score), normal sessions only: "
              f"rho={semantic_result['spearman_unseen_vs_semantic_score_normal_only']['rho']:.4f}, "
              f"p={semantic_result['spearman_unseen_vs_semantic_score_normal_only']['p']:.2e}",
              f"- Spearman(unseen rate, semantic_knn score), anomalous sessions only: "
              f"rho={semantic_result['spearman_unseen_vs_semantic_score_anomalous_only']['rho']:.4f}, "
              f"p={semantic_result['spearman_unseen_vs_semantic_score_anomalous_only']['p']:.2e}",
              f"- (For reference) semantic_knn AUROC on this diagnostic re-score: "
              f"{semantic_result['semantic_knn_auroc_this_rescore']:.4f} "
              f"(results/full/BGL/chrono/semantic_knn/seed0.json records the same quantity from the frozen run)",
              "",
              "## HDFS_v1/CHRONO/seed0 (normal-vs-anomalous unseen rate only)\n",
              f"- Mean per-session unseen-template rate, normal test sessions: **{hdfs_result['mean_unseen_rate_normal']:.4f}**",
              f"- Mean per-session unseen-template rate, anomalous test sessions: **{hdfs_result['mean_unseen_rate_anomalous']:.4f}**",
              f"- AUROC of unseen-rate-alone as an anomaly score: **{hdfs_result['auroc_unseen_rate_alone']:.4f}**",
              ""]

    with open(out_dir / "REPORT.md", "w") as f:
        f.write("\n".join(report))
    print(f"\nwrote {out_dir / 'REPORT.md'} and per-cell JSON files")


if __name__ == "__main__":
    main()
