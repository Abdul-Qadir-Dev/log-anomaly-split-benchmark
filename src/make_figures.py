"""Regenerate all figures from results/summary.csv and results/verdicts.json. PNG + PDF."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NON_LLM_ARMS = ["pca", "iforest", "ocsvm", "deeplog", "semantic_knn", "supervised_lgbm"]


def load(selftest=False):
    tag = "selftest" if selftest else "full"
    root = Path("results") / tag
    df = pd.read_csv(root / "summary.csv")
    with open(root / "verdicts.json") as f:
        verdicts = json.load(f)
    return df, verdicts, root


def save(fig, figdir, name):
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1_random_vs_chrono(df, figdir):
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        x = np.arange(len(NON_LLM_ARMS))
        width = 0.35
        for off, split, color in [(-width / 2, "random", "tab:blue"), (width / 2, "chrono", "tab:orange")]:
            means, errs = [], []
            for arm in NON_LLM_ARMS:
                sub = df[(df.dataset == dataset) & (df.split == split) & (df.arm == arm)]["auprc"]
                means.append(sub.mean())
                errs.append(sub.std() if len(sub) > 1 else 0)
            ax.bar(x + off, means, width, yerr=errs, label=split, color=color, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(NON_LLM_ARMS, rotation=30, ha="right")
        ax.set_ylabel("AUPRC (test)")
        ax.set_title(dataset)
        ax.legend()
    fig.suptitle("Fig 1: AUPRC, RANDOM vs CHRONO, per arm (error bars = across-seed SD)")
    save(fig, figdir, "fig1_random_vs_chrono")


def fig2_unseen_vs_drop(df, figdir):
    rows = []
    for dataset in sorted(df["dataset"].unique()):
        for seed in sorted(df["seed"].unique()):
            p = Path("results") / "full" / dataset / "chrono" / f"seed{seed}" / "unseen_template_rate.json"
            if not p.exists():
                continue
            with open(p) as f:
                utr = json.load(f)["unseen_template_rate"]
            for arm in NON_LLM_ARMS:
                r = df[(df.dataset == dataset) & (df.split == "random") & (df.arm == arm) & (df.seed == seed)]["auprc"]
                c = df[(df.dataset == dataset) & (df.split == "chrono") & (df.arm == arm) & (df.seed == seed)]["auprc"]
                if len(r) and len(c):
                    rows.append({"dataset": dataset, "drop": float(r.iloc[0] - c.iloc[0]), "utr": utr})
    sub = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 5))
    for dataset, marker in zip(sorted(sub["dataset"].unique()), ["o", "s", "^"]):
        d = sub[sub.dataset == dataset]
        ax.scatter(d["utr"], d["drop"], label=dataset, marker=marker, alpha=0.7)
    ax.set_xlabel("unseen_template_rate (CHRONO test split)")
    ax.set_ylabel("AUPRC drop (RANDOM - CHRONO)")
    ax.set_title("Fig 2: unseen-template rate vs AUPRC drop (H3)")
    ax.legend()
    save(fig, figdir, "fig2_unseen_vs_drop")


def fig3_oracle_vs_val(df, figdir):
    sub = df[df.arm.isin(NON_LLM_ARMS) & (df.split == "chrono")]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub["f1_val"], sub["f1_oracle"], alpha=0.6)
    lims = [0, 1]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("F1_val (validation-chosen threshold)")
    ax.set_ylabel("F1_oracle (test-tuned threshold)")
    ax.set_title("Fig 3: F1_oracle vs F1_val, CHRONO (H5)")
    save(fig, figdir, "fig3_oracle_vs_val")


def fig4_bump_chart(df, figdir):
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for split, x in [("random", 0), ("chrono", 1)]:
            means = {arm: df[(df.dataset == dataset) & (df.split == split) & (df.arm == arm)]["auprc"].mean()
                     for arm in NON_LLM_ARMS}
            ranked = sorted(means, key=lambda a: -means[a])
            for rank, arm in enumerate(ranked):
                ax.scatter(x, rank, s=80)
                ax.annotate(arm, (x, rank), xytext=(5 if x == 0 else -5, 0), textcoords="offset points",
                            ha="left" if x == 0 else "right", fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["random", "chrono"])
        ax.invert_yaxis()
        ax.set_ylabel("rank (1 = best AUPRC)")
        ax.set_title(dataset)
    fig.suptitle("Fig 4: ranking change, RANDOM -> CHRONO (H2)")
    save(fig, figdir, "fig4_bump_chart")


def main(selftest=False):
    df, verdicts, root = load(selftest)
    figdir = Path("figures/selftest") if selftest else Path("figures")
    fig1_random_vs_chrono(df, figdir)
    fig2_unseen_vs_drop(df, figdir)
    fig3_oracle_vs_val(df, figdir)
    fig4_bump_chart(df, figdir)
    print(f"figures written to {figdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    main(args.selftest)
