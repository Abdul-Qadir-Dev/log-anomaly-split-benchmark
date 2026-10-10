"""Step 12 consistency sweep: re-run only the deeplog rows of the
rolling-origin robustness check (results/full/addendum/rolling_origin/) at
cutoffs 50% and 70%, seed 0, with the Step 11 corrected arm (NLL-primary
scoring, normal-only early stopping). The other 5 arms' cutoff cells are
untouched -- their scoring did not change.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd

from arms.deeplog import DeepLogArm
from context import ArmContext
from dataset_paths import processed_path
from evaluate import evaluate_cell

from addendum_rolling_origin import chrono_cutoff_split, unseen_template_rate

SEED = 0
CUTOFFS = [0.50, 0.70]
OUT_DIR = Path("results/full/addendum/rolling_origin")


def run_one(dataset, cutoff):
    sessions = pd.read_parquet(processed_path(dataset, "sessions.parquet", False)).reset_index(drop=True)
    n_vocab = len(pd.read_parquet(processed_path(dataset, "templates.parquet", False)))

    train_idx, val_idx, test_idx = chrono_cutoff_split(sessions, cutoff)
    utr = unseen_template_rate(sessions, train_idx, test_idx)

    train = sessions.loc[train_idx].reset_index(drop=True)
    val = sessions.loc[val_idx].reset_index(drop=True)
    test = sessions.loc[test_idx].reset_index(drop=True)
    train = train[train["label"] == 0].reset_index(drop=True)

    ctx = ArmContext(dataset=dataset, selftest=False, n_vocab=n_vocab, seed=SEED)
    arm = DeepLogArm()
    arm.fit(ctx, train, val)
    score_val = arm.score(ctx, val)
    score_test = arm.score(ctx, test)
    m = evaluate_cell(val["label"].to_numpy(), score_val, test["label"].to_numpy(), score_test)
    m.update({"dataset": dataset, "arm": "deeplog", "cutoff": cutoff, "seed": SEED,
              "unseen_template_rate": utr, "n_train": len(train_idx), "n_val": len(val_idx),
              "n_test": len(test_idx), "scoring": "nll_corrected_step12"})
    print(f"{dataset} cutoff={cutoff} deeplog: AUPRC={m['auprc']:.4f} AUROC={m['auroc']:.4f} "
          f"n_test={len(test_idx)} utr={utr:.4f}")

    cell_path = OUT_DIR / f"cutoff{int(cutoff*100)}_{dataset}_deeplog.json"
    with open(cell_path, "w") as f:
        json.dump(m, f, indent=2)
    return m


def main():
    rows = []
    for dataset in ["HDFS_v1", "BGL"]:
        for cutoff in CUTOFFS:
            rows.append(run_one(dataset, cutoff))

    # Rebuild the full table (all 6 arms), pulling the freshly-written deeplog
    # cells from disk and every other arm's existing (unchanged) cutoff cells.
    main_df = pd.read_csv("results/full/summary.csv")
    main_df = main_df[(main_df.split == "chrono") & (main_df.seed == SEED)]
    non_llm_arms = ["pca", "iforest", "ocsvm", "deeplog", "semantic_knn", "supervised_lgbm"]

    lines = ["# Rolling-origin robustness: chronological cutoffs at 50%, 60% (main run), 70%\n",
             "Seed 0 only, all 6 non-LLM arms, both datasets. Robustness check only -- no verdict "
             "depends on this. `deeplog` rows reflect the Step 11/12 corrected (NLL) scoring; "
             "see `results/full/addendum/deeplog_diagnosis/DIAGNOSIS.md`.\n",
             "| Dataset | Arm | Cutoff | n_test | unseen_template_rate | AUPRC | AUROC |",
             "|---|---|---|---|---|---|---|"]
    for dataset in ["HDFS_v1", "BGL"]:
        for arm in non_llm_arms:
            m60 = main_df[(main_df.dataset == dataset) & (main_df.arm == arm)]
            if len(m60):
                r = m60.iloc[0]
                lines.append(f"| {dataset} | {arm} | 60% (main) | {int(r.n_test)} | "
                             f"{r.unseen_template_rate:.4f} | {r.auprc:.4f} | {r.auroc:.4f} |")
            for cutoff in CUTOFFS:
                cell_path = OUT_DIR / f"cutoff{int(cutoff*100)}_{dataset}_{arm}.json"
                with open(cell_path) as f:
                    row = json.load(f)
                pct = int(cutoff * 100)
                lines.append(f"| {dataset} | {arm} | {pct}% | {row['n_test']} | "
                             f"{row['unseen_template_rate']:.4f} | {row['auprc']:.4f} | {row['auroc']:.4f} |")

    with open(OUT_DIR / "rolling_origin_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {OUT_DIR / 'rolling_origin_table.md'}")


if __name__ == "__main__":
    main()
