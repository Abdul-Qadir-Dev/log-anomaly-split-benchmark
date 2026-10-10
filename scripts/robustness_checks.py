"""Section 9 robustness checks: reported only, never used to move a verdict.

1. BGL with 1-hour time windows instead of fixed-100-line windows.
2. HDFS with a 70/15/15 split instead of 60/20/20.
3. semantic_knn with k=5 and k=20 instead of k=10.

Runs seed 0 only (these are spot checks, not full re-runs) for all 6 non-LLM arms
where applicable, and writes results/full/robustness.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from arms import ARMS, NON_LLM_ARMS
from arms.semantic_knn import SemanticKNNArm
from context import ArmContext
from dataset_paths import processed_path
from evaluate import evaluate_cell
from split import chrono_split, random_split, unseen_template_rate

SEED = 0


def run_matrix(sessions, n_vocab, dataset_label, label):
    out = {}
    for split_name in ["random", "chrono"]:
        if split_name == "random":
            train_idx, val_idx, test_idx = random_split(sessions, SEED)
        else:
            train_idx, val_idx, test_idx = chrono_split(sessions)
        utr = unseen_template_rate(sessions, train_idx, test_idx)

        for arm_name in NON_LLM_ARMS:
            train = sessions.loc[train_idx].reset_index(drop=True)
            val = sessions.loc[val_idx].reset_index(drop=True)
            test = sessions.loc[test_idx].reset_index(drop=True)
            if arm_name != "supervised_lgbm":
                train = train[train["label"] == 0].reset_index(drop=True)

            ctx = ArmContext(dataset=dataset_label, selftest=False, n_vocab=n_vocab, seed=SEED)
            arm = ARMS[arm_name]()
            arm.fit(ctx, train)
            score_val = np.asarray(arm.score(ctx, val), dtype="float64")
            score_test = np.asarray(arm.score(ctx, test), dtype="float64")
            m = evaluate_cell(val["label"].to_numpy(), score_val, test["label"].to_numpy(), score_test)
            key = f"{label}/{split_name}/{arm_name}"
            out[key] = {"auroc": m["auroc"], "auprc": m["auprc"], "unseen_template_rate": utr}
            print(key, out[key])
    return out


def main():
    results = {}

    # 1. BGL 1-hour windows
    bgl_tw = pd.read_parquet(processed_path("BGL", "sessions_timewindow.parquet", False)).reset_index(drop=True)
    n_vocab_bgl = len(pd.read_parquet(processed_path("BGL", "templates.parquet", False)))
    results.update(run_matrix(bgl_tw, n_vocab_bgl, "BGL", "BGL_timewindow"))

    # 2. HDFS 70/15/15 split
    hdfs = pd.read_parquet(processed_path("HDFS_v1", "sessions.parquet", False)).reset_index(drop=True)
    n_vocab_hdfs = len(pd.read_parquet(processed_path("HDFS_v1", "templates.parquet", False)))

    def random_split_701515(sessions, seed):
        rng = np.random.RandomState(seed)
        idx = sessions.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    def chrono_split_701515(sessions):
        order = sessions.sort_values("timestamp").index.to_numpy()
        n = len(order)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]

    for split_name, fn in [("random", random_split_701515), ("chrono", chrono_split_701515)]:
        if split_name == "random":
            train_idx, val_idx, test_idx = fn(hdfs, SEED)
        else:
            train_idx, val_idx, test_idx = fn(hdfs)
        utr = unseen_template_rate(hdfs, train_idx, test_idx)
        for arm_name in NON_LLM_ARMS:
            train = hdfs.loc[train_idx].reset_index(drop=True)
            val = hdfs.loc[val_idx].reset_index(drop=True)
            test = hdfs.loc[test_idx].reset_index(drop=True)
            if arm_name != "supervised_lgbm":
                train = train[train["label"] == 0].reset_index(drop=True)
            ctx = ArmContext(dataset="HDFS_v1", selftest=False, n_vocab=n_vocab_hdfs, seed=SEED)
            arm = ARMS[arm_name]()
            arm.fit(ctx, train)
            score_val = np.asarray(arm.score(ctx, val), dtype="float64")
            score_test = np.asarray(arm.score(ctx, test), dtype="float64")
            m = evaluate_cell(val["label"].to_numpy(), score_val, test["label"].to_numpy(), score_test)
            key = f"HDFS_701515/{split_name}/{arm_name}"
            results[key] = {"auroc": m["auroc"], "auprc": m["auprc"], "unseen_template_rate": utr}
            print(key, results[key])

    # 3. semantic_knn with k=5 and k=20
    for dataset, sessions, n_vocab in [("HDFS_v1", hdfs, n_vocab_hdfs), ("BGL", bgl_tw, n_vocab_bgl)]:
        for split_name in ["random", "chrono"]:
            if split_name == "random":
                train_idx, val_idx, test_idx = random_split(sessions, SEED)
            else:
                train_idx, val_idx, test_idx = chrono_split(sessions)
            for k in [5, 20]:
                train = sessions.loc[train_idx]
                train = train[train["label"] == 0].reset_index(drop=True)
                val = sessions.loc[val_idx].reset_index(drop=True)
                test = sessions.loc[test_idx].reset_index(drop=True)
                ctx = ArmContext(dataset=dataset, selftest=False, n_vocab=n_vocab, seed=SEED)
                arm = SemanticKNNArm(k=k)
                arm.fit(ctx, train)
                score_val = np.asarray(arm.score(ctx, val), dtype="float64")
                score_test = np.asarray(arm.score(ctx, test), dtype="float64")
                m = evaluate_cell(val["label"].to_numpy(), score_val, test["label"].to_numpy(), score_test)
                key = f"{dataset}_semantic_k{k}/{split_name}"
                results[key] = {"auroc": m["auroc"], "auprc": m["auprc"]}
                print(key, results[key])

    out_path = Path("results/full/robustness.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
