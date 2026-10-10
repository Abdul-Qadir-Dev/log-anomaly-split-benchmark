"""Download the 2k-line Loghub sample logs used by every --selftest flag.

These are small, fixed-content files from the Loghub GitHub repo (not the full
Zenodo datasets scripts/download_data.py fetches) -- kept separate because they're
needed to validate the pipeline before committing to the full download.
"""
from pathlib import Path

import requests

SAMPLES = {
    "HDFS_2k.log": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
    "BGL_2k.log": "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log",
}


def main():
    out_dir = Path("data/sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLES.items():
        dest = out_dir / name
        print(f"Downloading {name} from {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"  wrote {dest} ({len(r.content):,} bytes)")


if __name__ == "__main__":
    main()
