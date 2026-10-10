"""Step 08 audit: verify the BGL parse. Reads raw data + results/full/ only
(both frozen / read-only here) and writes results/audit/bgl_parse_verification.md.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import pandas as pd

from dataset_paths import processed_path
from parse import parse_bgl_line

RAW_BGL = Path("data/raw/BGL/BGL.log")
EXPECTED_LINES = 4_747_963


def count_raw_lines_and_alerts():
    total = 0
    alert_lines = 0
    dropped = 0
    drop_reasons = {}
    with open(RAW_BGL, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            parts = line.split(maxsplit=1)
            if parts and parts[0] != "-":
                alert_lines += 1
            parsed = parse_bgl_line(line)
            if parsed is None:
                dropped += 1
                n_fields = len(line.split(maxsplit=9))
                drop_reasons[n_fields] = drop_reasons.get(n_fields, 0) + 1
    return total, alert_lines, dropped, drop_reasons


def template_counts():
    hdfs_templates = pd.read_parquet(processed_path("HDFS_v1", "templates.parquet", False))
    bgl_templates = pd.read_parquet(processed_path("BGL", "templates.parquet", False))
    return len(hdfs_templates), len(bgl_templates), bgl_templates


def top_templates(bgl_structured, bgl_templates, n=20):
    counts = bgl_structured["template"].value_counts().head(n)
    rows = []
    for template, count in counts.items():
        rows.append({"template": str(template), "count": int(count)})
    return rows


def unseen_template_rate_two_ways(dataset, split, seed):
    sessions = pd.read_parquet(processed_path(dataset, "sessions.parquet", False))
    d = Path("results/full") / dataset / split / f"seed{seed}"
    with open(d / "train_ids.json") as f:
        train_ids = set(json.load(f))
    with open(d / "test_ids.json") as f:
        test_ids = set(json.load(f))
    with open(d / "unseen_template_rate.json") as f:
        recorded = json.load(f)

    train = sessions[sessions["session_id"].isin(train_ids)]
    test = sessions[sessions["session_id"].isin(test_ids)]

    train_templates = set()
    for seq in train["template_seq"]:
        train_templates.update(seq)

    total_occ, unseen_occ = 0, 0
    test_templates_distinct = set()
    for seq in test["template_seq"]:
        total_occ += len(seq)
        unseen_occ += sum(1 for t in seq if t not in train_templates)
        test_templates_distinct.update(seq)

    unseen_distinct = test_templates_distinct - train_templates
    per_occurrence = unseen_occ / total_occ if total_occ else float("nan")
    per_unique = len(unseen_distinct) / len(test_templates_distinct) if test_templates_distinct else float("nan")

    return {
        "recorded_in_split_json": recorded["unseen_template_rate"],
        "recomputed_per_occurrence": per_occurrence,
        "per_unique_template": per_unique,
        "n_test_templates_distinct": len(test_templates_distinct),
        "n_test_templates_unseen_distinct": len(unseen_distinct),
        "n_test_occurrences": total_occ,
        "n_test_occurrences_unseen": unseen_occ,
    }


def main():
    print("Counting raw BGL lines (this reads all 4.7M lines)...")
    total, alert_lines, dropped, drop_reasons = count_raw_lines_and_alerts()

    n_hdfs_templates, n_bgl_templates, bgl_templates = template_counts()

    bgl_structured = pd.read_parquet(processed_path("BGL", "structured.parquet", False))
    top20 = top_templates(bgl_structured, bgl_templates, n=20)

    utr = unseen_template_rate_two_ways("BGL", "chrono", 0)

    report = {
        "total_lines_read": total,
        "expected_lines": EXPECTED_LINES,
        "lines_match_expected": total == EXPECTED_LINES,
        "lines_dropped_during_parse": dropped,
        "drop_reasons_by_field_count": drop_reasons,
        "alert_labelled_line_count": alert_lines,
        "non_alert_line_count": total - alert_lines,
        "n_templates_hdfs": n_hdfs_templates,
        "n_templates_bgl": n_bgl_templates,
        "bgl_over_fragmentation_flag": n_bgl_templates > 5000,
        "bgl_chrono_seed0_unseen_template_rate": utr,
    }

    Path("results/audit").mkdir(parents=True, exist_ok=True)
    with open("results/audit/bgl_parse_verification.json", "w") as f:
        json.dump({**report, "top_20_templates": top20}, f, indent=2)

    lines = ["# BGL parse verification (Step 08 audit)\n"]
    lines.append(f"- Total lines read from `data/raw/BGL/BGL.log`: **{total:,}** "
                 f"(expected {EXPECTED_LINES:,} -- {'MATCH' if total == EXPECTED_LINES else 'MISMATCH'})")
    lines.append(f"- Lines dropped during parse (content field absent / fewer than 9 whitespace fields): "
                 f"**{dropped}** (breakdown by field count: {drop_reasons})")
    lines.append(f"- Alert-labelled lines (first field != `-`): **{alert_lines:,}** "
                 f"({alert_lines/total*100:.2f}% of all lines)")
    lines.append(f"- Non-alert lines: **{total - alert_lines:,}**")
    lines.append(f"- Drain3 templates: HDFS_v1 = **{n_hdfs_templates}**, BGL = **{n_bgl_templates}**")
    lines.append(f"- BGL template count {'EXCEEDS' if n_bgl_templates > 5000 else 'is within'} the "
                 f"5,000 over-fragmentation flag threshold (bar: >5,000).")
    lines.append("")
    lines.append("## BGL CHRONO seed 0 unseen_template_rate, two ways\n")
    lines.append(f"- As recorded by `src/split.py` (per occurrence, what every H1/H3/H5 verdict used): "
                 f"**{utr['recorded_in_split_json']:.4f}**")
    lines.append(f"- Recomputed independently, per occurrence: **{utr['recomputed_per_occurrence']:.4f}** "
                 f"({utr['n_test_occurrences_unseen']:,} / {utr['n_test_occurrences']:,} occurrences)")
    lines.append(f"- Per unique template (fraction of *distinct* templates in test never seen in train): "
                 f"**{utr['per_unique_template']:.4f}** "
                 f"({utr['n_test_templates_unseen_distinct']} / {utr['n_test_templates_distinct']} distinct templates)")
    lines.append("")
    lines.append("## Top 20 most frequent BGL templates (shown for verification regardless of the "
                 "5,000-template flag, since it was not triggered)\n")
    lines.append("| Rank | Count | Template |")
    lines.append("|---|---|---|")
    for i, row in enumerate(top20, 1):
        t = row["template"].replace("|", "\\|")
        if len(t) > 100:
            t = t[:100] + "..."
        lines.append(f"| {i} | {row['count']:,} | `{t}` |")

    with open("results/audit/bgl_parse_verification.md", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\nwrote results/audit/bgl_parse_verification.{md,json}")


if __name__ == "__main__":
    main()
