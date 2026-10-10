"""Step 11: re-run only the deeplog cells (both datasets, both splits, 5
seeds = 20 cells) with the corrected arm (src/arms/deeplog.py, Step 11 fix).
Reuses run_all.py's run_cell() unchanged -- the correction lives entirely in the
arm itself and in run_cell()'s existing special-case for passing val to deeplog.
"""
import sys

sys.path.insert(0, "src")

from run_all import DATASETS, SEEDS, SPLITS, load_sessions, run_cell, write_summary


def main():
    for dataset in DATASETS:
        sessions = load_sessions(dataset, selftest=False)
        for split in SPLITS:
            for seed in SEEDS:
                run_cell(dataset, split, "deeplog", seed, sessions, selftest=False)
    write_summary(selftest=False)


if __name__ == "__main__":
    main()
