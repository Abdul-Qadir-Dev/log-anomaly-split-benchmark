"""Shared path/format lookup for the two datasets, full data or selftest sample."""
from pathlib import Path

DATASETS = {
    "HDFS_v1": {
        "raw_log": Path("data/raw/HDFS_v1/HDFS.log"),
        "sample_log": Path("data/sample/HDFS_2k.log"),
        "labels": Path("data/raw/HDFS_v1/preprocessed/anomaly_label.csv"),
        "kind": "hdfs",
    },
    "BGL": {
        "raw_log": Path("data/raw/BGL/BGL.log"),
        "sample_log": Path("data/sample/BGL_2k.log"),
        "labels": None,
        "kind": "bgl",
    },
}


def log_path(dataset, selftest):
    return DATASETS[dataset]["sample_log" if selftest else "raw_log"]


def processed_path(dataset, name, selftest):
    tag = "selftest" if selftest else "full"
    d = Path("data/processed") / tag
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{dataset}_{name}"


def kind(dataset):
    return DATASETS[dataset]["kind"]
