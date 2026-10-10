"""Loop datasets x splits x arms x seeds; resumable (skips completed cells); writes results/summary.csv."""
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arms import ARMS, NON_LLM_ARMS, SAMPLE_BASED_ARMS
from context import ArmContext
from dataset_paths import processed_path
from evaluate import evaluate_cell, save_cell

DATASETS = ["HDFS_v1", "BGL"]
SPLITS = ["random", "chrono"]
SEEDS = [0, 1, 2, 3, 4]
MAX_FIT_SECONDS = 90 * 60

LLM_SAMPLE_PER_CLASS = 200


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_git_sha(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "git_sha.txt").write_text(git_sha() + "\n")


def load_sessions(dataset, selftest=False):
    return pd.read_parquet(processed_path(dataset, "sessions.parquet", selftest))


def load_split_ids(dataset, split, seed, selftest=False):
    tag = "selftest" if selftest else "full"
    d = Path("results") / tag / dataset / split / f"seed{seed}"
    out = {}
    for name in ["train", "val", "test"]:
        with open(d / f"{name}_ids.json") as f:
            out[name] = json.load(f)
    with open(d / "unseen_template_rate.json") as f:
        out["unseen_template_rate"] = json.load(f)["unseen_template_rate"]
    return out


def subset(sessions, ids):
    return sessions[sessions["session_id"].isin(set(ids))].reset_index(drop=True)


def log(msg, selftest=False):
    tag = "selftest" if selftest else "full"
    print(msg, flush=True)
    with open(Path("results") / tag / "run.log", "a") as f:
        f.write(msg + "\n")


def run_cell(dataset, split, arm_name, seed, sessions, selftest=False):
    tag = "selftest" if selftest else "full"
    out_path = Path("results") / tag / dataset / split / arm_name / f"seed{seed}.json"
    if out_path.exists():
        return "skipped"

    split_ids = load_split_ids(dataset, split, seed, selftest)
    train = subset(sessions, split_ids["train"])
    val = subset(sessions, split_ids["val"])
    test = subset(sessions, split_ids["test"])

    if arm_name != "supervised_lgbm":
        train = train[train["label"] == 0].reset_index(drop=True)

    n_vocab = len(pd.read_parquet(processed_path(dataset, "templates.parquet", selftest)))
    ctx = ArmContext(dataset=dataset, selftest=selftest, n_vocab=n_vocab, seed=seed)

    arm = ARMS[arm_name]()
    t0 = time.time()
    if arm_name == "deeplog":
        arm.fit(ctx, train, val)  # normal-only early stopping needs the validation split
    else:
        arm.fit(ctx, train)
    fit_s = time.time() - t0
    if fit_s > MAX_FIT_SECONDS:
        log(f"WARNING: {dataset}/{split}/{arm_name}/seed{seed} fit took {fit_s:.0f}s > 90min cap. "
            f"See DECISIONS.md for the halving rule; this run was NOT halved automatically.", selftest)

    score_val = np.asarray(arm.score(ctx, val), dtype="float64")
    topk_val = arm.last_topk_scores_.copy() if hasattr(arm, "last_topk_scores_") else None
    score_test = np.asarray(arm.score(ctx, test), dtype="float64")
    topk_test = arm.last_topk_scores_.copy() if hasattr(arm, "last_topk_scores_") else None

    metrics = evaluate_cell(val["label"].to_numpy(), score_val, test["label"].to_numpy(), score_test)
    metrics["fit_seconds"] = fit_s
    metrics["unseen_template_rate"] = split_ids["unseen_template_rate"]
    metrics["dataset"], metrics["split"], metrics["arm"], metrics["seed"] = dataset, split, arm_name, seed

    # deeplog's primary score (NLL) comes with a secondary top-9 miss-rate score
    # (Step 11 correction) -- record it alongside the primary metrics, never used
    # as the primary ranking metric.
    if arm_name == "deeplog" and topk_val is not None and topk_test is not None:
        secondary = evaluate_cell(val["label"].to_numpy(), topk_val, test["label"].to_numpy(), topk_test)
        for k in ("auroc", "auprc", "f1_val", "f1_oracle", "f1_leakage_gap"):
            metrics[f"secondary_topk9_{k}"] = secondary[k]

    path = save_cell(dataset, split, arm_name, seed, metrics, selftest)
    write_git_sha(path.parent)
    log(f"{dataset}/{split}/{arm_name}/seed{seed}: AUPRC={metrics['auprc']:.4f} "
        f"AUROC={metrics['auroc']:.4f} fit={fit_s:.1f}s", selftest)
    del arm
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return "ran"


def run_llm_cell(dataset, split, sessions, selftest=False, seed=0):
    tag = "selftest" if selftest else "full"
    out_path = Path("results") / tag / dataset / split / "llm_zeroshot" / f"seed{seed}.json"
    if out_path.exists():
        return "skipped"

    split_ids = load_split_ids(dataset, split, seed, selftest)
    test = subset(sessions, split_ids["test"])
    pos = test[test["label"] == 1]
    neg = test[test["label"] == 0]
    n_per_class = min(LLM_SAMPLE_PER_CLASS, len(pos), len(neg))
    if n_per_class == 0:
        log(f"SKIPPED llm_zeroshot {dataset}/{split}: not enough of one class in test "
            f"(pos={len(pos)}, neg={len(neg)})", selftest)
        return "skipped_insufficient_classes"
    sample = pd.concat([
        pos.sample(n=n_per_class, random_state=seed),
        neg.sample(n=n_per_class, random_state=seed),
    ]).reset_index(drop=True)

    n_vocab = len(pd.read_parquet(processed_path(dataset, "templates.parquet", selftest)))
    ctx = ArmContext(dataset=dataset, selftest=selftest, n_vocab=n_vocab, seed=seed)

    from arms import LLMZeroShotArm
    arm = LLMZeroShotArm()
    t0 = time.time()
    arm.fit(ctx, None)
    scores = np.asarray(arm.score(ctx, sample), dtype="float64")
    elapsed = time.time() - t0

    from sklearn.metrics import average_precision_score, roc_auc_score
    y = sample["label"].to_numpy()
    from evaluate import best_f1_threshold
    thr, f1 = best_f1_threshold(y, scores)
    metrics = {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "f1_oracle": float(f1),
        "n_sample": len(sample), "n_per_class": n_per_class,
        "elapsed_seconds": elapsed, "sample_based": True,
        "model_name_used": getattr(arm, "model_name_used_", "unknown"),
        "device_used": getattr(arm, "device_", "unknown"),
        "dataset": dataset, "split": split, "arm": "llm_zeroshot", "seed": seed,
    }
    path = save_cell(dataset, split, "llm_zeroshot", seed, metrics, selftest)
    write_git_sha(path.parent)
    log(f"{dataset}/{split}/llm_zeroshot/seed{seed}: AUPRC={metrics['auprc']:.4f} "
        f"AUROC={metrics['auroc']:.4f} n={len(sample)} elapsed={elapsed:.0f}s", selftest)
    return "ran"


def write_summary(selftest=False, out_root="results"):
    tag = "selftest" if selftest else "full"
    root = Path(out_root) / tag
    rows = []
    for p in root.rglob("seed*.json"):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    out = root / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out} ({len(rows)} rows)")


def main(selftest=False):
    for dataset in DATASETS:
        sessions = load_sessions(dataset, selftest)
        for split in SPLITS:
            for arm_name in NON_LLM_ARMS:
                for seed in SEEDS:
                    run_cell(dataset, split, arm_name, seed, sessions, selftest)
    for dataset in DATASETS:
        sessions = load_sessions(dataset, selftest)
        for split in SPLITS:
            run_llm_cell(dataset, split, sessions, selftest, seed=0)
    write_summary(selftest)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    main(args.selftest)
