"""Step 10: unseen-template sanity check for BGL/CHRONO/seed0.

For every test-session template occurrence whose template id never appears in
training, find the nearest training template by Jaccard similarity on
lowercased whitespace tokens, and report the 20 most frequent such unseen
templates with their nearest-training-template match and similarity score.
Reads only from results/full/ (frozen) and data/processed/full/; writes to
results/full/addendum/.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd

from dataset_paths import processed_path

SEED = 0


def jaccard(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def tokenize(template):
    return template.lower().split()


def main():
    sessions = pd.read_parquet(processed_path("BGL", "sessions.parquet", False))
    templates = pd.read_parquet(processed_path("BGL", "templates.parquet", False)).set_index("template_id")["template"]

    d = Path("results/full/BGL/chrono") / f"seed{SEED}"
    with open(d / "train_ids.json") as f:
        train_ids = set(json.load(f))
    with open(d / "test_ids.json") as f:
        test_ids = set(json.load(f))

    train = sessions[sessions["session_id"].isin(train_ids)]
    test = sessions[sessions["session_id"].isin(test_ids)]

    train_templates = set()
    for seq in train["template_seq"]:
        train_templates.update(seq)

    unseen_counts = Counter()
    total_occ = 0
    unseen_occ = 0
    for seq in test["template_seq"]:
        for t in seq:
            total_occ += 1
            if t not in train_templates:
                unseen_occ += 1
                unseen_counts[t] += 1

    print(f"Total test occurrences: {total_occ}, unseen: {unseen_occ} ({unseen_occ/total_occ:.4f})")

    train_tokens = {tid: tokenize(templates[tid]) for tid in train_templates}

    top20 = unseen_counts.most_common(20)

    rows = []
    high_sim_occ = 0  # occurrences (not distinct templates) whose nearest match has sim > 0.8
    # Compute, for every unseen template (not just top 20), its best match, to get the
    # occurrence-weighted fraction above 0.8 accurately.
    unseen_best_sim = {}
    for tid in unseen_counts:
        toks = tokenize(templates[tid])
        best_tid, best_sim = None, -1.0
        for ttid, ttoks in train_tokens.items():
            sim = jaccard(toks, ttoks)
            if sim > best_sim:
                best_sim = sim
                best_tid = ttid
        unseen_best_sim[tid] = (best_tid, best_sim)
        if best_sim > 0.8:
            high_sim_occ += unseen_counts[tid]

    frac_high_sim = high_sim_occ / unseen_occ if unseen_occ else float("nan")

    for tid, count in top20:
        best_tid, best_sim = unseen_best_sim[tid]
        rows.append({
            "template_id": int(tid),
            "template": templates[tid],
            "occurrences": int(count),
            "nearest_train_template_id": int(best_tid) if best_tid is not None else None,
            "nearest_train_template": templates[best_tid] if best_tid is not None else None,
            "jaccard_similarity": round(best_sim, 4),
        })

    out_dir = Path("results/full/addendum")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "unseen_templates_bgl.json", "w") as f:
        json.dump({
            "total_test_occurrences": total_occ,
            "unseen_test_occurrences": unseen_occ,
            "unseen_rate": unseen_occ / total_occ,
            "fraction_unseen_occ_with_nearest_train_similarity_above_0.8": frac_high_sim,
            "similarity_metric": "Jaccard similarity on lowercased whitespace-split tokens",
            "top_20": rows,
        }, f, indent=2)

    lines = ["# BGL unseen-template sanity check (Step 10)\n",
             f"BGL/CHRONO/seed{SEED}. `results/full/` frozen -- reads the existing split's train/test session "
             "ids only, no cell re-run.\n",
             f"- Total test-session template occurrences: {total_occ:,}",
             f"- Unseen (never in train) occurrences: {unseen_occ:,} ({unseen_occ/total_occ:.4f} = "
             f"{unseen_occ/total_occ*100:.3g}%)",
             f"- Similarity metric: **Jaccard similarity on lowercased whitespace-split tokens** of the "
             "Drain3-masked template string (e.g. `<NUM>`, `<HEX>`, `<*>` count as tokens).",
             f"- Fraction of unseen occurrences whose nearest training template has similarity > 0.8: "
             f"**{frac_high_sim:.4f}**",
             "",
             "## 20 most frequent unseen test templates\n",
             "| Rank | Occurrences | Unseen test template | Nearest train template | Jaccard sim |",
             "|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        t = r["template"].replace("|", "\\|")
        nt = (r["nearest_train_template"] or "").replace("|", "\\|")
        if len(t) > 80:
            t = t[:80] + "..."
        if len(nt) > 80:
            nt = nt[:80] + "..."
        lines.append(f"| {i} | {r['occurrences']:,} | `{t}` | `{nt}` | {r['jaccard_similarity']:.4f} |")

    with open(out_dir / "unseen_templates_bgl.md", "w") as f:
        f.write("\n".join(lines))

    print(f"wrote {out_dir / 'unseen_templates_bgl.md'} and .json")
    print(f"fraction with similarity > 0.8: {frac_high_sim:.4f}")


if __name__ == "__main__":
    main()
