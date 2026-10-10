"""Turn structured, parsed logs into labelled sessions.

HDFS: one session per block id, label from anomaly_label.csv.
BGL: non-overlapping 100-line windows (primary) + 1-hour time windows (robustness).
"""
import argparse

import pandas as pd

from dataset_paths import DATASETS, processed_path


def load_structured(dataset, selftest):
    path = processed_path(dataset, "structured.parquet", selftest)
    return pd.read_parquet(path)


def template_ids(df):
    return df["template"].cat.codes.astype("int32")


def sessionize_hdfs(dataset, selftest):
    df = load_structured(dataset, selftest)
    df["template_id"] = template_ids(df)
    df = df[df["block_ids"].str.len() > 0].copy()
    df["block_id"] = df["block_ids"].str.split("|")
    exploded = df.explode("block_id")[["line_id", "timestamp", "template_id", "block_id"]]
    exploded = exploded.sort_values("line_id")

    sessions = (
        exploded.groupby("block_id")
        .agg(
            timestamp=("timestamp", "first"),
            template_seq=("template_id", list),
            num_lines=("template_id", "size"),
        )
        .reset_index()
        .rename(columns={"block_id": "session_id"})
    )

    labels_path = DATASETS[dataset]["labels"]
    if labels_path is not None and labels_path.exists():
        # The selftest sample is literally a prefix of the full log, so its block ids
        # are a subset of the full anomaly_label.csv -- real labels work for selftest too.
        labels = pd.read_csv(labels_path)
        labels["label"] = (labels["Label"] == "Anomaly").astype("int8")
        sessions = sessions.merge(
            labels[["BlockId", "label"]], left_on="session_id", right_on="BlockId", how="left"
        ).drop(columns=["BlockId"])
    else:
        sessions["label"] = 0

    sessions["label"] = sessions["label"].fillna(0).astype("int8")
    return sessions


def sessionize_bgl_fixed(dataset, selftest, window_size=100):
    df = load_structured(dataset, selftest)
    df = df.sort_values("line_id").reset_index(drop=True)
    df["template_id"] = template_ids(df)
    df["window_id"] = df.index // window_size
    df["is_alert"] = df["label_line"] != "-"

    sessions = (
        df.groupby("window_id")
        .agg(
            timestamp=("timestamp", "first"),
            template_seq=("template_id", list),
            num_lines=("template_id", "size"),
            label=("is_alert", "any"),
        )
        .reset_index()
        .rename(columns={"window_id": "session_id"})
    )
    sessions["session_id"] = "win_" + sessions["session_id"].astype(str)
    sessions["label"] = sessions["label"].astype("int8")
    return sessions


def sessionize_bgl_timewindow(dataset, selftest):
    df = load_structured(dataset, selftest)
    df = df.sort_values("line_id").reset_index(drop=True)
    df["template_id"] = template_ids(df)
    df["is_alert"] = df["label_line"] != "-"
    df["hour_bucket"] = df["timestamp"].dt.floor("h")

    sessions = (
        df.groupby("hour_bucket")
        .agg(
            timestamp=("timestamp", "first"),
            template_seq=("template_id", list),
            num_lines=("template_id", "size"),
            label=("is_alert", "any"),
        )
        .reset_index(drop=True)
    )
    sessions["session_id"] = "hr_" + sessions.index.astype(str)
    sessions["label"] = sessions["label"].astype("int8")
    return sessions[["session_id", "timestamp", "template_seq", "num_lines", "label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["HDFS_v1", "BGL"], required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.dataset == "HDFS_v1":
        sessions = sessionize_hdfs(args.dataset, args.selftest)
        out = processed_path(args.dataset, "sessions.parquet", args.selftest)
        sessions.to_parquet(out, index=False)
        print(f"HDFS_v1 selftest={args.selftest}: {len(sessions):,} sessions, "
              f"anomaly rate={sessions['label'].mean():.4f} -> {out}")
    else:
        sessions = sessionize_bgl_fixed(args.dataset, args.selftest)
        out = processed_path(args.dataset, "sessions.parquet", args.selftest)
        sessions.to_parquet(out, index=False)
        print(f"BGL selftest={args.selftest}: {len(sessions):,} fixed-100 sessions, "
              f"anomaly rate={sessions['label'].mean():.4f} -> {out}")

        tw = sessionize_bgl_timewindow(args.dataset, args.selftest)
        tw_out = processed_path(args.dataset, "sessions_timewindow.parquet", args.selftest)
        tw.to_parquet(tw_out, index=False)
        print(f"BGL selftest={args.selftest}: {len(tw):,} 1-hour-window sessions, "
              f"anomaly rate={tw['label'].mean():.4f} -> {tw_out}")


if __name__ == "__main__":
    main()
