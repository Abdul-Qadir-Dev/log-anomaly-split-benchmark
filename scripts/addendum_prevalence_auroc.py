"""Step 09 audit: prevalence + AUROC-drop addendum, built entirely from
results/full/summary.csv (no cell re-run). Writes results/full/addendum/prevalence.csv
and results/full/addendum/auroc_drops.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from analysis import NON_LLM_ARMS, per_arm_dataset_drop_dict, seed_bootstrap_ci


def write_prevalence(df, out_dir):
    cols = ["dataset", "split", "seed", "n_test", "n_anomaly_test", "n_val", "n_anomaly_val"]
    sub = df[cols].drop_duplicates().sort_values(["dataset", "split", "seed"]).reset_index(drop=True)
    sub["prevalence"] = sub["n_anomaly_test"] / sub["n_test"]
    sub = sub[["dataset", "split", "seed", "n_test", "n_anomaly_test", "prevalence", "n_val", "n_anomaly_val"]]
    out = out_dir / "prevalence.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out} ({len(sub)} rows)")
    return sub


def per_arm_dataset_drop_dict_metric(df, arm, dataset, metric):
    r = df[(df.arm == arm) & (df.dataset == dataset) & (df.split == "random")].set_index("seed")[metric]
    c = df[(df.arm == arm) & (df.dataset == dataset) & (df.split == "chrono")].set_index("seed")[metric]
    common = sorted(set(r.index) & set(c.index))
    return {s: float(r[s] - c[s]) for s in common}


def write_auroc_drops(df, out_dir):
    datasets = sorted(df["dataset"].unique())
    rows = []
    for dataset in datasets:
        for arm in NON_LLM_ARMS:
            auroc_diffs = per_arm_dataset_drop_dict_metric(df, arm, dataset, "auroc")
            auprc_diffs = per_arm_dataset_drop_dict_metric(df, arm, dataset, "auprc")
            a_mean = float(np.mean(list(auroc_diffs.values())))
            a_lo, a_hi = seed_bootstrap_ci(list(auroc_diffs.values()))
            p_mean = float(np.mean(list(auprc_diffs.values())))
            p_lo, p_hi = seed_bootstrap_ci(list(auprc_diffs.values()))
            rows.append({
                "dataset": dataset, "arm": arm,
                "auroc_drop_mean": a_mean, "auroc_drop_ci_lo": a_lo, "auroc_drop_ci_hi": a_hi,
                "auprc_drop_mean": p_mean, "auprc_drop_ci_lo": p_lo, "auprc_drop_ci_hi": p_hi,
            })

    lines = ["# RANDOM->CHRONO AUROC drop vs AUPRC drop, per dataset x arm\n",
             "Same seed-level paired bootstrap CI methodology as AUPRC-based hypotheses "
             "(resample the 5 per-seed differences, 1000x, seed 0 -- see DECISIONS.md Step 05).\n",
             "| Dataset | Arm | AUROC drop (mean) | AUROC drop 95% CI | AUPRC drop (mean) | AUPRC drop 95% CI |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['arm']} | {r['auroc_drop_mean']:.4f} | "
            f"[{r['auroc_drop_ci_lo']:.4f}, {r['auroc_drop_ci_hi']:.4f}] | "
            f"{r['auprc_drop_mean']:.4f} | [{r['auprc_drop_ci_lo']:.4f}, {r['auprc_drop_ci_hi']:.4f}] |"
        )

    out = out_dir / "auroc_drops.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return rows


def main():
    df = pd.read_csv("results/full/summary.csv")
    out_dir = Path("results/full/addendum")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_prevalence(df, out_dir)
    write_auroc_drops(df, out_dir)


if __name__ == "__main__":
    main()
