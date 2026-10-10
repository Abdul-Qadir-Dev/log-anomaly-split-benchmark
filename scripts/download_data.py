"""Download Loghub datasets with streaming + retry + resume."""
import sys
import time
import zipfile
from pathlib import Path

import requests

DATASETS = {
    "HDFS_v1": "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1",
    "BGL": "https://zenodo.org/records/8196385/files/BGL.zip?download=1",
}

FALLBACK_BASE = "https://zenodo.org/records/3227177/files/{name}.zip?download=1"

MIN_SIZE_BYTES = 10 * 1024 * 1024  # sanity floor; real check is against Content-Length (see download())
RETRIES = 5
RAW_DIR = Path("data/raw")


def download(url, dest, retries=RETRIES):
    for attempt in range(1, retries + 1):
        try:
            headers = {}
            mode = "wb"
            existing = 0
            if dest.exists():
                existing = dest.stat().st_size
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if r.status_code == 404:
                    return False, "404"
                if r.status_code == 416:
                    # Range beyond EOF: the resumed file is already complete.
                    return True, None
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"status {r.status_code}")
                total = int(r.headers.get("content-length", 0)) + existing
                with open(dest, mode) as f:
                    downloaded = existing
                    last_pct = -1
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = int(downloaded * 100 / total)
                                if pct != last_pct:
                                    print(f"  {dest.name}: {pct}% ({downloaded/1e6:.0f}MB/{total/1e6:.0f}MB)", flush=True)
                                    last_pct = pct
            if dest.stat().st_size < MIN_SIZE_BYTES:
                raise RuntimeError(f"file too small: {dest.stat().st_size} bytes")
            return True, None
        except Exception as e:
            print(f"  attempt {attempt}/{retries} failed: {e}", flush=True)
            time.sleep(5 * attempt)
    return False, "exhausted retries"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, url in DATASETS.items():
        ddir = RAW_DIR / name
        ddir.mkdir(parents=True, exist_ok=True)
        zpath = ddir / f"{name}.zip"
        print(f"Downloading {name} from {url}", flush=True)
        ok, err = download(url, zpath)
        used_url = url
        if not ok:
            fb_url = FALLBACK_BASE.format(name=name)
            print(f"Primary failed ({err}); trying fallback {fb_url}", flush=True)
            if zpath.exists():
                zpath.unlink()
            ok, err = download(fb_url, zpath)
            used_url = fb_url
        if not ok:
            failures.append((name, url, FALLBACK_BASE.format(name=name), err))
            continue
        print(f"Extracting {zpath} ...", flush=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(ddir)
        print(f"{name} done, used URL: {used_url}", flush=True)

    if failures:
        with open("data/DOWNLOAD_FAILED.md", "w") as f:
            f.write("# Download failures\n\n")
            for name, url, fb, err in failures:
                f.write(f"## {name}\n\nPrimary: {url}\nFallback: {fb}\nError: {err}\n\n")
                f.write(f"To fix: download manually and extract into `data/raw/{name}/`.\n\n")
        print("Some datasets failed to download; see data/DOWNLOAD_FAILED.md", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
