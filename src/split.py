"""RANDOM and CHRONO 60/20/20 splits of sessions, per seed.

Writes, per (dataset, split_condition, seed):
  results/<dataset>/<split>/seed<k>/{train,val,test}_ids.json
  results/<dataset>/<split>/seed<k>/unseen_template_rate.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_paths import processed_path


def load_sessions(dataset, selftest, variant="sessions.parquet"):
    return pd.read_parquet(processed_path(dataset, variant, selftest))


def random_split(sessions, seed):
    rng = np.random.RandomState(seed)
    idx = sessions.index.to_numpy().copy()
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def chrono_split(sessions):
    order = sessions.sort_values("timestamp").index.to_numpy()
    n = len(order)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    t = sessions["timestamp"]
    assert t.loc[train_idx].max() <= t.loc[val_idx].min(), "CHRONO leak: train overlaps val"
    assert t.loc[val_idx].max() <= t.loc[test_idx].min(), "CHRONO leak: val overlaps test"
    return train_idx, val_idx, test_idx


def unseen_template_rate(sessions, train_idx, test_idx):
    train_templates = set()
    for seq in sessions.loc[train_idx, "template_seq"]:
        train_templates.update(seq)
    total_occ = 0
    unseen_occ = 0
    for seq in sessions.loc[test_idx, "template_seq"]:
        total_occ += len(seq)
        unseen_occ += sum(1 for t in seq if t not in train_templates)
    return unseen_occ / total_occ if total_occ else 0.0


def run_one(dataset, selftest, seed, variant="sessions.parquet", out_root="results"):
    sessions = load_sessions(dataset, selftest, variant)
    sessions = sessions.reset_index(drop=True)

    for split_name, split_fn in [("random", None), ("chrono", None)]:
        if split_name == "random":
            train_idx, val_idx, test_idx = random_split(sessions, seed)
        else:
            train_idx, val_idx, test_idx = chrono_split(sessions)

        utr = unseen_template_rate(sessions, train_idx, test_idx)

        tag = "selftest" if selftest else "full"
        d = Path(out_root) / tag / dataset / split_name / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        for name, ids in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            with open(d / f"{name}_ids.json", "w") as f:
                json.dump(sessions.loc[ids, "session_id"].tolist(), f)
        with open(d / "unseen_template_rate.json", "w") as f:
            json.dump({"unseen_template_rate": utr, "n_train": len(train_idx),
                       "n_val": len(val_idx), "n_test": len(test_idx)}, f, indent=2)
        print(f"{dataset} {split_name} seed={seed}: train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)} unseen_template_rate={utr:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["HDFS_v1", "BGL"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    for seed in args.seeds:
        run_one(args.dataset, args.selftest, seed)


if __name__ == "__main__":
    main()
