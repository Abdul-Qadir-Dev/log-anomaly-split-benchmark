"""Write results/data_stats.json (Section 5.1: sessions, anomaly rate, template count per dataset)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import pandas as pd

from dataset_paths import processed_path

stats = {}
for dataset in ["HDFS_v1", "BGL"]:
    sessions = pd.read_parquet(processed_path(dataset, "sessions.parquet", False))
    templates = pd.read_parquet(processed_path(dataset, "templates.parquet", False))
    entry = {
        "n_sessions": int(len(sessions)),
        "anomaly_rate": float(sessions["label"].mean()),
        "n_templates": int(len(templates)),
    }
    if dataset == "BGL":
        tw = pd.read_parquet(processed_path(dataset, "sessions_timewindow.parquet", False))
        entry["n_sessions_timewindow"] = int(len(tw))
        entry["anomaly_rate_timewindow"] = float(tw["label"].mean())
    stats[dataset] = entry

Path("results").mkdir(exist_ok=True)
with open("results/data_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
