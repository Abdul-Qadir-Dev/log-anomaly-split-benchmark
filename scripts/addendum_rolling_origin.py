"""Step 09 audit: rolling-origin robustness. Chronological splits with
train cutoffs at 50% and 70% (val = next 20%, test = remainder), all 6 non-LLM
arms, both datasets, seed 0 only. Robustness only -- no verdict depends on this.
Writes to results/full/addendum/rolling_origin/.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from arms import ARMS, NON_LLM_ARMS
from context import ArmContext
from dataset_paths import processed_path
from evaluate import evaluate_cell

SEED = 0
CUTOFFS = [0.50, 0.70]


def chrono_cutoff_split(sessions, train_frac):
    order = sessions.sort_values("timestamp").index.to_numpy()
    n = len(order)
    n_train = int(n * train_frac)
    n_val = int(n * 0.20)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]  # remainder (30% at cutoff .50, 10% at cutoff .70)

    t = sessions["timestamp"]
    assert t.loc[train_idx].max() <= t.loc[val_idx].min()
    assert t.loc[val_idx].max() <= t.loc[test_idx].min()
    return train_idx, val_idx, test_idx


def unseen_template_rate(sessions, train_idx, test_idx):
    train_templates = set()
    for seq in sessions.loc[train_idx, "template_seq"]:
        train_templates.update(seq)
    total_occ, unseen_occ = 0, 0
    for seq in sessions.loc[test_idx, "template_seq"]:
        total_occ += len(seq)
        unseen_occ += sum(1 for t in seq if t not in train_templates)
    return unseen_occ / total_occ if total_occ else 0.0


def run_dataset_cutoff(dataset, cutoff, out_dir):
    sessions = pd.read_parquet(processed_path(dataset, "sessions.parquet", False)).reset_index(drop=True)
    n_vocab = len(pd.read_parquet(processed_path(dataset, "templates.parquet", False)))

    train_idx, val_idx, test_idx = chrono_cutoff_split(sessions, cutoff)
    utr = unseen_template_rate(sessions, train_idx, test_idx)

    rows = []
    for arm_name in NON_LLM_ARMS:
        train = sessions.loc[train_idx].reset_index(drop=True)
        val = sessions.loc[val_idx].reset_index(drop=True)
        test = sessions.loc[test_idx].reset_index(drop=True)
        if arm_name != "supervised_lgbm":
            train = train[train["label"] == 0].reset_index(drop=True)

        ctx = ArmContext(dataset=dataset, selftest=False, n_vocab=n_vocab, seed=SEED)
        arm = ARMS[arm_name]()
        t0 = time.time()
        arm.fit(ctx, train)
        fit_s = time.time() - t0
        score_val = np.asarray(arm.score(ctx, val), dtype="float64")
        score_test = np.asarray(arm.score(ctx, test), dtype="float64")
        m = evaluate_cell(val["label"].to_numpy(), score_val, test["label"].to_numpy(), score_test)
        m.update({"dataset": dataset, "arm": arm_name, "cutoff": cutoff, "seed": SEED,
                  "fit_seconds": fit_s, "unseen_template_rate": utr,
                  "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)})
        rows.append(m)
        print(f"{dataset} cutoff={cutoff} {arm_name}: AUPRC={m['auprc']:.4f} AUROC={m['auroc']:.4f} "
              f"n_test={len(test_idx)} utr={utr:.4f}")

        cell_path = out_dir / f"cutoff{int(cutoff*100)}_{dataset}_{arm_name}.json"
        with open(cell_path, "w") as f:
            json.dump(m, f, indent=2)
    return rows


def main():
    out_dir = Path("results/full/addendum/rolling_origin")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset in ["HDFS_v1", "BGL"]:
        for cutoff in CUTOFFS:
            all_rows.extend(run_dataset_cutoff(dataset, cutoff, out_dir))

    # Bring in the existing 60% (main-run) values for the side-by-side table.
    main_df = pd.read_csv("results/full/summary.csv")
    main_df = main_df[(main_df.split == "chrono") & (main_df.seed == SEED)]

    lines = ["# Rolling-origin robustness: chronological cutoffs at 50%, 60% (main run), 70%\n",
             "Seed 0 only, all 6 non-LLM arms, both datasets. Robustness check only -- no verdict depends on this.\n",
             "| Dataset | Arm | Cutoff | n_test | unseen_template_rate | AUPRC | AUROC |",
             "|---|---|---|---|---|---|---|"]
    for dataset in ["HDFS_v1", "BGL"]:
        for arm in NON_LLM_ARMS:
            m60 = main_df[(main_df.dataset == dataset) & (main_df.arm == arm)]
            if len(m60):
                r = m60.iloc[0]
                lines.append(f"| {dataset} | {arm} | 60% (main) | {int(r.n_test)} | "
                             f"{r.unseen_template_rate:.4f} | {r.auprc:.4f} | {r.auroc:.4f} |")
            for row in [r for r in all_rows if r["dataset"] == dataset and r["arm"] == arm]:
                pct = int(row["cutoff"] * 100)
                lines.append(f"| {dataset} | {arm} | {pct}% | {row['n_test']} | "
                             f"{row['unseen_template_rate']:.4f} | {row['auprc']:.4f} | {row['auroc']:.4f} |")

    with open(out_dir / "rolling_origin_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {out_dir / 'rolling_origin_table.md'}")


if __name__ == "__main__":
    main()
