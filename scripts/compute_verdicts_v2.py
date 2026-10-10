"""Step 11: compute verdicts from the corrected-deeplog summary.csv into
results/full/VERDICTS_v2.md and verdicts_v2.json, WITHOUT touching the original
results/full/VERDICTS.md / verdicts.json (which stay as the pre-registered v1
record, computed from the original saturated-deeplog results).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from analysis import INTERPRETATIONS, h1, h2, h3, h4, h5, load_summary


def main():
    df = load_summary(selftest=False)
    verdicts = {"H1": h1(df), "H2": h2(df), "H3": h3(df), "H4": h4(df), "H5": h5(df)}

    out_root = Path("results/full")
    with open(out_root / "verdicts_v2.json", "w") as f:
        json.dump(verdicts, f, indent=2)

    lines = ["# Verdicts (v2 -- corrected deeplog, Step 11)\n",
             "Computed from `results/full/summary.csv` after the Step 11 deeplog correction "
             "(NLL-primary scoring, normal-only early stopping) was re-run. The original "
             "pre-registered verdicts, computed from the saturated top-k deeplog scores, remain "
             "unchanged in `VERDICTS.md` / `verdicts.json`. Superseded v1 deeplog cells: "
             "`results/superseded/deeplog_v1/`.\n"]
    for h, v in verdicts.items():
        lines.append(f"## {h}: {v['verdict']}\n")
        lines.append(f"**Bar:** {v['bar']}\n")
        lines.append(f"**Interpretation:** {INTERPRETATIONS[h].get(v['verdict'], '')}\n")
        lines.append("```json")
        lines.append(json.dumps({k: val for k, val in v.items() if k != "bar"}, indent=2, default=str))
        lines.append("```\n")
    with open(out_root / "VERDICTS_v2.md", "w") as f:
        f.write("\n".join(lines))

    for h, v in verdicts.items():
        print(f"{h}: {v['verdict']}")


if __name__ == "__main__":
    main()
