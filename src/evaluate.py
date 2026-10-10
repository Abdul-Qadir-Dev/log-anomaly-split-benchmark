"""All metrics + bootstrap CIs for one (dataset, split, arm, seed) cell.

The test split is touched exactly once per cell: run_all.py calls arm.score(test)
once, then evaluate_cell() computes every metric from that single set of scores.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

N_BOOT = 1000
BOOT_SEED = 0  # fixed regardless of the run's own seed (Section 5.4)


def best_f1_threshold(y, scores):
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float(np.median(scores)), 0.0
    precision, recall, thresholds = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    if len(thresholds) == 0:
        return float(np.median(scores)), 0.0
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def f1_at_threshold(y, scores, thr):
    y = np.asarray(y)
    pred = (np.asarray(scores) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def bootstrap_ci(y_test, score_test, metric_fn, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    n = len(y_test)
    vals = []
    y_test = np.asarray(y_test)
    score_test = np.asarray(score_test)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, sb = y_test[idx], score_test[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        vals.append(metric_fn(yb, sb))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_cell(y_val, score_val, y_test, score_test):
    y_val, score_val = np.asarray(y_val), np.asarray(score_val)
    y_test, score_test = np.asarray(y_test), np.asarray(score_test)

    has_both_classes = len(set(y_test.tolist())) > 1

    auroc = roc_auc_score(y_test, score_test) if has_both_classes else float("nan")
    auprc = average_precision_score(y_test, score_test) if has_both_classes else float("nan")

    thr_val, _f1_val_on_val = best_f1_threshold(y_val, score_val)
    precision_val, recall_val, f1_val = f1_at_threshold(y_test, score_test, thr_val)

    thr_oracle, f1_oracle = best_f1_threshold(y_test, score_test)

    auroc_ci = bootstrap_ci(y_test, score_test, roc_auc_score) if has_both_classes else (float("nan"), float("nan"))
    auprc_ci = bootstrap_ci(y_test, score_test, average_precision_score) if has_both_classes else (float("nan"), float("nan"))

    return {
        "auroc": auroc, "auroc_ci_lo": auroc_ci[0], "auroc_ci_hi": auroc_ci[1],
        "auprc": auprc, "auprc_ci_lo": auprc_ci[0], "auprc_ci_hi": auprc_ci[1],
        "threshold_val": thr_val,
        "precision_val": precision_val, "recall_val": recall_val, "f1_val": f1_val,
        "threshold_oracle": thr_oracle, "f1_oracle": f1_oracle,
        "f1_leakage_gap": f1_oracle - f1_val,
        "n_val": len(y_val), "n_test": len(y_test),
        "n_anomaly_val": int(y_val.sum()), "n_anomaly_test": int(y_test.sum()),
    }


def save_cell(dataset, split, arm, seed, metrics, selftest=False, out_root="results"):
    tag = "selftest" if selftest else "full"
    d = Path(out_root) / tag / dataset / split / arm
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"seed{seed}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return d / f"seed{seed}.json"
