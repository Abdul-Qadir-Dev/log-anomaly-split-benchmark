"""Compute H1-H5 verdicts exactly as pre-registered in PROTOCOL.md Section 5.5.

CI methodology note (see DECISIONS.md, Step 05): for hypotheses about a
difference (H1, H4, H5) we use a seed-level paired bootstrap (resample the
per-seed differences with replacement, 1000 times, seed 0) rather than the
per-condition test-resampling CI stored in summary.csv (that one is for the
single-condition AUPRC estimate itself, not for a cross-condition difference).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

NON_LLM_ARMS = ["pca", "iforest", "ocsvm", "deeplog", "semantic_knn", "supervised_lgbm"]
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 1000
BOOT_SEED = 0


def load_summary(selftest=False):
    tag = "selftest" if selftest else "full"
    return pd.read_csv(Path("results") / tag / "summary.csv")


def seed_bootstrap_ci(values, n_boot=N_BOOT, seed=BOOT_SEED):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = [values[rng.randint(0, len(values), len(values))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def per_arm_dataset_drop_dict(df, arm, dataset):
    r = df[(df.arm == arm) & (df.dataset == dataset) & (df.split == "random")].set_index("seed")["auprc"]
    c = df[(df.arm == arm) & (df.dataset == dataset) & (df.split == "chrono")].set_index("seed")["auprc"]
    common = sorted(set(r.index) & set(c.index))
    return {s: float(r[s] - c[s]) for s in common}


def h1(df):
    datasets = sorted(df["dataset"].unique())
    detail, n_meet_point, n_meet_ci = {}, {}, {}
    for dataset in datasets:
        n_meet_point[dataset] = 0
        n_meet_ci[dataset] = 0
        for arm in NON_LLM_ARMS:
            diffs = list(per_arm_dataset_drop_dict(df, arm, dataset).values())
            if not diffs:
                detail[f"{dataset}/{arm}"] = None
                continue
            mean_diff = float(np.mean(diffs))
            lo, hi = seed_bootstrap_ci(diffs)
            point_ok = mean_diff >= 0.05
            ci_ok = point_ok and lo > 0
            detail[f"{dataset}/{arm}"] = {"mean_drop": mean_diff, "ci_lo": lo, "ci_hi": hi,
                                           "point_ok": point_ok, "ci_ok": ci_ok}
            n_meet_point[dataset] += int(point_ok)
            n_meet_ci[dataset] += int(ci_ok)

    confirmed = any(v >= 4 for v in n_meet_ci.values())
    inconclusive = (not confirmed) and any(v >= 4 for v in n_meet_point.values())
    verdict = "CONFIRMED" if confirmed else ("INCONCLUSIVE" if inconclusive else "NOT CONFIRMED")
    return {"bar": "AUPRC_random - AUPRC_chrono >= 0.05, CI excl. 0, on >=4/6 arms, on >=1 dataset",
            "n_meeting_bar_by_dataset_point": n_meet_point,
            "n_meeting_bar_by_dataset_ci": n_meet_ci,
            "detail": detail, "verdict": verdict}


def h2(df):
    datasets = sorted(df["dataset"].unique())
    detail, any_unstable = {}, False
    for dataset in datasets:
        means = {}
        for split in ["random", "chrono"]:
            means[split] = {arm: df[(df.dataset == dataset) & (df.split == split) & (df.arm == arm)]["auprc"].mean()
                             for arm in NON_LLM_ARMS}
        rank_random = sorted(NON_LLM_ARMS, key=lambda a: -means["random"][a])
        rank_chrono = sorted(NON_LLM_ARMS, key=lambda a: -means["chrono"][a])
        r_ranks = [rank_random.index(a) for a in NON_LLM_ARMS]
        c_ranks = [rank_chrono.index(a) for a in NON_LLM_ARMS]
        tau, p = kendalltau(r_ranks, c_ranks)
        unstable = bool(tau < 0.8)
        detail[dataset] = {"tau": float(tau), "p": float(p), "rank_random": rank_random,
                            "rank_chrono": rank_chrono, "unstable": unstable}
        any_unstable = any_unstable or unstable
    verdict = "CONFIRMED" if any_unstable else "NOT CONFIRMED"
    return {"bar": "Kendall tau < 0.8 between RANDOM and CHRONO AUPRC rankings, on >=1 dataset",
            "detail": detail, "verdict": verdict}


def h3(df):
    rows = []
    for dataset in sorted(df["dataset"].unique()):
        for seed in SEEDS:
            p = Path("results") / "full" / dataset / "chrono" / f"seed{seed}" / "unseen_template_rate.json"
            if not p.exists():
                continue
            with open(p) as f:
                utr = json.load(f)["unseen_template_rate"]
            for arm in NON_LLM_ARMS:
                r = df[(df.dataset == dataset) & (df.split == "random") & (df.arm == arm) & (df.seed == seed)]["auprc"]
                c = df[(df.dataset == dataset) & (df.split == "chrono") & (df.arm == arm) & (df.seed == seed)]["auprc"]
                if len(r) == 0 or len(c) == 0:
                    continue
                rows.append({"dataset": dataset, "seed": seed, "arm": arm,
                             "drop": float(r.iloc[0] - c.iloc[0]), "unseen_template_rate": utr})
    sub = pd.DataFrame(rows)
    if len(sub) > 2:
        rho, p = spearmanr(sub["unseen_template_rate"], sub["drop"])
    else:
        rho, p = float("nan"), float("nan")
    point_ok = rho > 0.4
    sig = p < 0.05
    verdict = "CONFIRMED" if (point_ok and sig) else ("INCONCLUSIVE" if point_ok else "NOT CONFIRMED")
    return {"bar": "Spearman rho > 0.4, p < 0.05, across (dataset x seed x arm) cells",
            "rho": float(rho), "p": float(p), "n": len(sub), "verdict": verdict}


def h4(df):
    datasets = sorted(df["dataset"].unique())
    detail = {}
    all_point_ok, all_ci_ok = True, True
    for dataset in datasets:
        sem = per_arm_dataset_drop_dict(df, "semantic_knn", dataset)
        res = {}
        for other in ["pca", "deeplog"]:
            oth = per_arm_dataset_drop_dict(df, other, dataset)
            common = sorted(set(sem) & set(oth))
            diffs = [oth[s] - sem[s] for s in common]  # >0 means semantic's drop is smaller
            mean_diff = float(np.mean(diffs)) if diffs else float("nan")
            lo, hi = seed_bootstrap_ci(diffs) if diffs else (float("nan"), float("nan"))
            point_ok = mean_diff > 0
            ci_ok = point_ok and lo > 0
            res[other] = {"mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi, "point_ok": point_ok, "ci_ok": ci_ok}
            all_point_ok = all_point_ok and point_ok
            all_ci_ok = all_ci_ok and ci_ok
        detail[dataset] = res
    verdict = "CONFIRMED" if all_ci_ok else ("INCONCLUSIVE" if all_point_ok else "NOT CONFIRMED")
    return {"bar": "semantic_knn's RANDOM->CHRONO AUPRC drop smaller than pca's and deeplog's, "
                   "both datasets, CI of difference excl. 0",
            "detail": detail, "verdict": verdict}


def h5(df):
    datasets = sorted(df["dataset"].unique())
    detail = {}
    any_confirmed, any_inconclusive = False, False
    for dataset in datasets:
        sub = df[(df.dataset == dataset) & (df.split == "chrono") & (df.arm.isin(NON_LLM_ARMS))]
        gaps = sub["f1_leakage_gap"].dropna().to_numpy()
        mean_gap = float(gaps.mean()) if len(gaps) else float("nan")
        lo, hi = seed_bootstrap_ci(gaps) if len(gaps) else (float("nan"), float("nan"))
        point_ok = mean_gap >= 0.10
        ci_ok = point_ok and lo > 0
        detail[dataset] = {"mean_gap": mean_gap, "ci_lo": lo, "ci_hi": hi, "point_ok": point_ok,
                            "ci_ok": ci_ok, "n": len(gaps)}
        any_confirmed = any_confirmed or ci_ok
        any_inconclusive = any_inconclusive or (point_ok and not ci_ok)
    verdict = "CONFIRMED" if any_confirmed else ("INCONCLUSIVE" if any_inconclusive else "NOT CONFIRMED")
    return {"bar": "mean(F1_oracle - F1_val) >= 0.10 under CHRONO, CI excl. 0, on >=1 dataset",
            "detail": detail, "verdict": verdict}


INTERPRETATIONS = {
    "H1": {
        "CONFIRMED": "Random splits substantially inflate AUPRC relative to chronological splits for most arms.",
        "NOT CONFIRMED": "Chronological evaluation does not change AUPRC enough, on these datasets, to call it split inflation by this bar.",
        "INCONCLUSIVE": "The point estimates suggest split inflation but the bootstrap CI is too wide to call it confidently.",
    },
    "H2": {
        "CONFIRMED": "Which method looks best depends on how you split the data.",
        "NOT CONFIRMED": "The relative ranking of methods is stable across RANDOM and CHRONO splits.",
    },
    "H3": {
        "CONFIRMED": "Cells with more unseen templates in the chronological test set show larger AUPRC drops, supporting template drift as a mechanism.",
        "NOT CONFIRMED": "Unseen-template rate does not explain the AUPRC drop pattern.",
        "INCONCLUSIVE": "A positive association exists but is not statistically significant at p<0.05.",
    },
    "H4": {
        "CONFIRMED": "The semantic embedding arm degrades less under chronological splitting than PCA or DeepLog, consistent with it handling unseen templates better.",
        "NOT CONFIRMED": "The semantic arm is not more robust to chronological splitting than PCA/DeepLog on these datasets.",
        "INCONCLUSIVE": "The semantic arm's point-estimate advantage does not clear the CI bar.",
    },
    "H5": {
        "CONFIRMED": "Tuning the threshold on the test set (as many papers implicitly do) materially inflates F1 over the honest, validation-tuned threshold.",
        "NOT CONFIRMED": "Threshold leakage is not large enough, on these datasets, to meet the pre-registered bar.",
        "INCONCLUSIVE": "The leakage gap looks large on average but the CI is too wide to confirm.",
    },
}


def main(selftest=False):
    df = load_summary(selftest)
    verdicts = {"H1": h1(df), "H2": h2(df), "H3": h3(df), "H4": h4(df), "H5": h5(df)}

    tag = "selftest" if selftest else "full"
    out_root = Path("results") / tag
    with open(out_root / "verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=2)

    lines = ["# Verdicts\n"]
    for h, v in verdicts.items():
        lines.append(f"## {h}: {v['verdict']}\n")
        lines.append(f"**Bar:** {v['bar']}\n")
        lines.append(f"**Interpretation:** {INTERPRETATIONS[h].get(v['verdict'], '')}\n")
        lines.append("```json")
        lines.append(json.dumps({k: val for k, val in v.items() if k != "bar"}, indent=2, default=str))
        lines.append("```\n")
    with open(out_root / "VERDICTS.md", "w") as f:
        f.write("\n".join(lines))

    for h, v in verdicts.items():
        print(f"{h}: {v['verdict']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    main(args.selftest)
