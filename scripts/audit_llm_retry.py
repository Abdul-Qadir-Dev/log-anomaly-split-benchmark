"""Step 08 audit: retry the exploratory llm_zeroshot arm with the smaller
Qwen2.5-0.5B-Instruct model (GPU fp16, CPU fallback on CUDA failure). This arm has
no verdict attached to it, so re-running it is allowed; results go only to
results/audit/llm_zeroshot/ and results/audit/llm_table.md -- never into the main
ranking or results/full/, which stays frozen.
"""
import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch

from context import ArmContext
from dataset_paths import processed_path

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LINES = 40
N_PER_CLASS = 100  # -> up to 200 sessions per dataset per split
SEED = 0

PROMPT_TEMPLATE = (
    "You are a log anomaly detector. Given this sequence of log templates from a system,"
    " estimate the probability (0-100) that it represents an anomaly.\n"
    "Log templates:\n{text}\n\n"
    "Respond with only a single integer from 0 to 100."
)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.cuda.empty_cache() if device == "cuda" else None
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        return tok, model, device
    except (torch.OutOfMemoryError, OSError) as e:
        if device == "cuda":
            print(f"GPU load failed ({e}); falling back to CPU.")
            torch.cuda.empty_cache()
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to("cpu")
            return tok, model, "cpu"
        raise


def render(seq, templates):
    lines = [templates.get(t, "<unknown>") for t in list(seq)[:MAX_LINES]]
    return "\n".join(lines)


def score_sessions(tok, model, device, sample, templates):
    scores = np.zeros(len(sample), dtype="float32")
    for i, seq in enumerate(sample["template_seq"]):
        prompt = PROMPT_TEMPLATE.format(text=render(seq, templates))
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\d+", gen)
        scores[i] = min(100.0, float(m.group())) if m else 0.0
    return scores


def main():
    out_dir = Path("results/audit/llm_zeroshot")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Loading {MODEL_NAME}...")
        tok, model, device = load_model()
        print(f"Loaded on device={device}")
    except Exception:
        tb = traceback.format_exc()
        Path("results/audit/llm_failure.txt").write_text(tb)
        print("Model load failed entirely; traceback saved to results/audit/llm_failure.txt")
        print(tb)
        return

    rows = []
    for dataset in ["HDFS_v1", "BGL"]:
        sessions = pd.read_parquet(processed_path(dataset, "sessions.parquet", False))
        templates = pd.read_parquet(processed_path(dataset, "templates.parquet", False)).set_index("template_id")["template"]
        for split in ["random", "chrono"]:
            d = Path("results/full") / dataset / split / f"seed{SEED}"
            with open(d / "test_ids.json") as f:
                test_ids = set(json.load(f))
            test = sessions[sessions["session_id"].isin(test_ids)].reset_index(drop=True)
            pos = test[test["label"] == 1]
            neg = test[test["label"] == 0]
            n = min(N_PER_CLASS, len(pos), len(neg))
            if n == 0:
                print(f"SKIP {dataset}/{split}: insufficient classes (pos={len(pos)}, neg={len(neg)})")
                continue
            sample = pd.concat([
                pos.sample(n=n, random_state=SEED),
                neg.sample(n=n, random_state=SEED),
            ]).reset_index(drop=True)

            try:
                scores = score_sessions(tok, model, device, sample, templates)
            except Exception:
                tb = traceback.format_exc()
                Path("results/audit/llm_failure.txt").write_text(tb)
                print(f"Scoring failed on {dataset}/{split}; traceback saved to results/audit/llm_failure.txt")
                print(tb)
                return

            y = sample["label"].to_numpy()
            from sklearn.metrics import average_precision_score, roc_auc_score
            auroc = float(roc_auc_score(y, scores))
            auprc = float(average_precision_score(y, scores))

            cell = {
                "dataset": dataset, "split": split, "seed": SEED,
                "model": MODEL_NAME, "device": device,
                "n_sample": len(sample), "n_per_class": n,
                "auroc": auroc, "auprc": auprc,
            }
            rows.append(cell)
            with open(out_dir / f"{dataset}_{split}_seed{SEED}.json", "w") as f:
                json.dump(cell, f, indent=2)
            print(f"{dataset}/{split}: AUROC={auroc:.4f} AUPRC={auprc:.4f} n={len(sample)} device={device}")

    if not rows:
        print("No cells completed.")
        return

    lines = ["# llm_zeroshot audit retry (Qwen2.5-0.5B-Instruct)\n",
             "Exploratory arm, no pre-registered verdict attached -- not part of the main ranking.\n",
             "| Dataset | Split | AUROC | AUPRC | N sample | Device |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['split']} | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['n_sample']} | {r['device']} |")
    Path("results/audit/llm_table.md").write_text("\n".join(lines))
    print("\nwrote results/audit/llm_table.md and results/audit/llm_zeroshot/*.json")


if __name__ == "__main__":
    main()
