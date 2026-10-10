"""Parse HDFS or BGL raw logs into a structured parquet using Drain3.

Columns written: line_id, timestamp (UTC, int64 ns), label_line (BGL only),
block_ids (HDFS only, '|'-joined blk_ ids or ''), template (category dtype;
template_id = template.cat.codes).
"""
import argparse
import re
import sys
import time

import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

from dataset_paths import kind, log_path, processed_path

HDFS_LINE_RE = re.compile(
    r"^(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([\w\.\$]+):\s?(.*)$"
)
BLK_RE = re.compile(r"blk_-?\d+")


def make_template_miner():
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth = 4
    config.masking_instructions = [
        MaskingInstruction(r"blk_-?\d+", "BLK"),
        MaskingInstruction(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?", "IP"),
        MaskingInstruction(r"0x[0-9a-fA-F]+", "HEX"),
        MaskingInstruction(r"\d+", "NUM"),
    ]
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def parse_hdfs_line(line):
    m = HDFS_LINE_RE.match(line)
    if not m:
        return None
    date_s, time_s, _pid, _level, _component, content = m.groups()
    year = 2000 + int(date_s[0:2])
    month, day = int(date_s[2:4]), int(date_s[4:6])
    hour, minute, second = int(time_s[0:2]), int(time_s[2:4]), int(time_s[4:6])
    ts = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second, tz="UTC")
    block_ids = "|".join(BLK_RE.findall(line))
    return ts, content, block_ids


def parse_bgl_line(line):
    parts = line.rstrip("\n").split(maxsplit=9)
    if len(parts) < 9:
        return None
    label = parts[0]
    try:
        epoch = int(parts[1])
    except ValueError:
        return None
    ts = pd.Timestamp(epoch, unit="s", tz="UTC")
    content = parts[9] if len(parts) == 10 else ""
    return ts, content, label


def parse_file(dataset, selftest, progress_every=500_000):
    path = log_path(dataset, selftest)
    k = kind(dataset)
    miner = make_template_miner()

    line_ids, timestamps, templates = [], [], []
    extra = []  # block_ids (hdfs) or label (bgl)

    t0 = time.time()
    n_bad = 0
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            if k == "hdfs":
                parsed = parse_hdfs_line(line)
            else:
                parsed = parse_bgl_line(line)
            if parsed is None:
                n_bad += 1
                continue
            ts, content, extra_val = parsed
            result = miner.add_log_message(content)
            line_ids.append(i)
            timestamps.append(ts.value)  # int64 ns since epoch
            templates.append(result["template_mined"])
            extra.append(extra_val)
            if progress_every and (i + 1) % progress_every == 0:
                elapsed = time.time() - t0
                print(f"  parsed {i+1:,} lines in {elapsed:.0f}s ({(i+1)/elapsed:.0f} lines/s)", flush=True)

    df = pd.DataFrame(
        {
            "line_id": pd.array(line_ids, dtype="int64"),
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "template": pd.Categorical(templates),
        }
    )
    if k == "hdfs":
        df["block_ids"] = extra
    else:
        df["label_line"] = extra

    out = processed_path(dataset, "structured.parquet", selftest)
    df.to_parquet(out, index=False)

    vocab = pd.DataFrame(
        {
            "template_id": range(len(df["template"].cat.categories)),
            "template": df["template"].cat.categories,
        }
    )
    vocab_out = processed_path(dataset, "templates.parquet", selftest)
    vocab.to_parquet(vocab_out, index=False)

    n_templates = len(vocab)
    print(f"dataset={dataset} selftest={selftest}: {len(df):,} lines parsed ({n_bad} unparseable), {n_templates} templates -> {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["HDFS_v1", "BGL"], required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    parse_file(args.dataset, args.selftest)


if __name__ == "__main__":
    main()
