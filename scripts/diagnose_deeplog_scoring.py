"""Step 11, tasks 2-3: score saturation check, k-sweep, and NLL scoring variant,
using the checkpoints trained by diagnose_deeplog.py. Reads results/full/ (frozen,
read-only) for split membership; writes to results/full/addendum/deeplog_diagnosis/.
Per-combo results are cached to disk so a kill/retry (this machine has shown
occasional transient slowdowns) doesn't waste already-finished combos.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from dataset_paths import processed_path
from evaluate import evaluate_cell

sys.path.insert(0, "scripts")
from diagnose_deeplog import (CKPT_DIR, K_SWEEP, OUT_DIR, DeepLogNet, load_split,
                               nll_score, state_dict_hash, topk_score, windows_for_sessions)

CONFIG_NAMES = ["lightweight", "fullstrength"]
SPLITS = ["random", "chrono"]


def load_checkpoint_model(dataset, split, config_name):
    meta_path = OUT_DIR / f"{dataset}_{split}_{config_name}_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    ckpt_path = CKPT_DIR / f"{dataset}_{split}_{config_name}.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepLogNet(meta["n_vocab"], meta["hidden"], meta["layers"]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, meta, device


def score_distribution(scores, y):
    frac_zero = float((scores == 0).mean())
    mean_normal = float(scores[y == 0].mean()) if (y == 0).any() else float("nan")
    mean_anom = float(scores[y == 1].mean()) if (y == 1).any() else float("nan")
    hist, edges = np.histogram(scores, bins=10, range=(0, 1))
    return {
        "fraction_exactly_zero": frac_zero,
        "mean_score_normal": mean_normal,
        "mean_score_anomalous": mean_anom,
        "histogram_counts": hist.tolist(),
        "histogram_bin_edges": edges.tolist(),
    }


def process_combo(config_name, split, out_dir):
    combo_path = out_dir / f"combo_HDFS_v1_{split}_{config_name}.json"
    if combo_path.exists():
        print(f"{config_name}/{split}: cached, loading.", flush=True)
        with open(combo_path) as f:
            return json.load(f)

    model, meta, device = load_checkpoint_model("HDFS_v1", split, config_name)
    hash_loaded = state_dict_hash(model)

    train, val, test = load_split("HDFS_v1", split)
    pad_id = meta["pad_id"]
    y_test = test["label"].to_numpy()
    y_val = val["label"].to_numpy()

    windows, targets, owner = windows_for_sessions(test, pad_id)
    val_windows, val_targets, val_owner = windows_for_sessions(val, pad_id)

    topk_test_by_k, topk_val_by_k = {}, {}
    for k in K_SWEEP:
        topk_val_by_k[k] = topk_score(model, val_windows, val_targets, val_owner, len(val), device, k=k)
        topk_test_by_k[k] = topk_score(model, windows, targets, owner, len(test), device, k=k)
        print(f"  k={k} scored", flush=True)

    hash_after = state_dict_hash(model)
    hash_check = {
        "dataset": "HDFS_v1", "split": split, "config": config_name,
        "hash_after_training": meta["state_dict_hash_after_training"],
        "hash_after_loading": hash_loaded,
        "hash_after_scoring": hash_after,
        "unchanged_load_vs_train": hash_loaded == meta["state_dict_hash_after_training"],
        "unchanged_score_vs_load": hash_after == hash_loaded,
    }

    dist = score_distribution(topk_test_by_k[9], y_test)
    print(f"{config_name}/{split}: frac_zero={dist['fraction_exactly_zero']:.4f} "
          f"mean_normal={dist['mean_score_normal']:.4f} mean_anom={dist['mean_score_anomalous']:.4f}",
          flush=True)

    k_sweep_rows = []
    has_both = len(set(y_test.tolist())) > 1
    for k in K_SWEEP:
        auroc = float(roc_auc_score(y_test, topk_test_by_k[k])) if has_both else float("nan")
        auprc = float(average_precision_score(y_test, topk_test_by_k[k])) if has_both else float("nan")
        k_sweep_rows.append({"config": config_name, "split": split, "k": k, "auroc": auroc, "auprc": auprc})
        print(f"  k={k}: AUROC={auroc:.4f} AUPRC={auprc:.4f}", flush=True)

    nll_val = nll_score(model, val_windows, val_targets, val_owner, len(val), device)
    nll_test = nll_score(model, windows, targets, owner, len(test), device)
    metrics_nll = evaluate_cell(y_val, nll_val, y_test, nll_test)
    metrics_topk = evaluate_cell(y_val, topk_val_by_k[9], y_test, topk_test_by_k[9])
    rho, p = spearmanr(nll_test, topk_test_by_k[9])

    nll_row = {
        "config": config_name, "split": split,
        "nll_auroc": metrics_nll["auroc"], "nll_auprc": metrics_nll["auprc"],
        "nll_f1_val": metrics_nll["f1_val"], "nll_f1_oracle": metrics_nll["f1_oracle"],
        "topk9_auroc": metrics_topk["auroc"], "topk9_auprc": metrics_topk["auprc"],
        "topk9_f1_val": metrics_topk["f1_val"], "topk9_f1_oracle": metrics_topk["f1_oracle"],
        "spearman_nll_vs_topk9": float(rho), "spearman_p": float(p),
    }
    print(f"  NLL: AUROC={metrics_nll['auroc']:.4f} AUPRC={metrics_nll['auprc']:.4f} "
          f"| top9: AUROC={metrics_topk['auroc']:.4f} AUPRC={metrics_topk['auprc']:.4f} "
          f"| rho(nll,top9)={rho:.4f}", flush=True)

    result = {"hash_check": hash_check, "saturation": dist, "k_sweep": k_sweep_rows, "nll_row": nll_row}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(combo_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    k_sweep_rows = []
    saturation = {}
    nll_rows = []
    hash_checks = []

    for config_name in CONFIG_NAMES:
        for split in SPLITS:
            result = process_combo(config_name, split, OUT_DIR)
            hash_checks.append(result["hash_check"])
            saturation[f"HDFS_v1_{split}_{config_name}"] = result["saturation"]
            k_sweep_rows.extend(result["k_sweep"])
            nll_rows.append(result["nll_row"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "hash_checks.json", "w") as f:
        json.dump(hash_checks, f, indent=2)
    with open(OUT_DIR / "score_saturation.json", "w") as f:
        json.dump(saturation, f, indent=2)
    with open(OUT_DIR / "k_sweep.json", "w") as f:
        json.dump(k_sweep_rows, f, indent=2)
    with open(OUT_DIR / "nll_vs_topk.json", "w") as f:
        json.dump(nll_rows, f, indent=2)

    lines = ["# K-sweep: AUROC/AUPRC vs top-k, both configs, both splits\n",
             "| Config | Split | k | AUROC | AUPRC |", "|---|---|---|---|---|"]
    for r in k_sweep_rows:
        lines.append(f"| {r['config']} | {r['split']} | {r['k']} | {r['auroc']:.4f} | {r['auprc']:.4f} |")
    with open(OUT_DIR / "k_sweep_table.md", "w") as f:
        f.write("\n".join(lines))

    lines2 = ["# NLL scoring vs top-9 scoring, both configs, both splits\n",
              "| Config | Split | NLL AUROC | NLL AUPRC | NLL F1_val | NLL F1_oracle | "
              "Top9 AUROC | Top9 AUPRC | Top9 F1_val | Top9 F1_oracle | Spearman(nll,top9) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in nll_rows:
        lines2.append(f"| {r['config']} | {r['split']} | {r['nll_auroc']:.4f} | {r['nll_auprc']:.4f} | "
                       f"{r['nll_f1_val']:.4f} | {r['nll_f1_oracle']:.4f} | {r['topk9_auroc']:.4f} | "
                       f"{r['topk9_auprc']:.4f} | {r['topk9_f1_val']:.4f} | {r['topk9_f1_oracle']:.4f} | "
                       f"{r['spearman_nll_vs_topk9']:.4f} |")
    with open(OUT_DIR / "nll_vs_topk_table.md", "w") as f:
        f.write("\n".join(lines2))

    print("\nAll saturation/scoring outputs written.", flush=True)


if __name__ == "__main__":
    main()
