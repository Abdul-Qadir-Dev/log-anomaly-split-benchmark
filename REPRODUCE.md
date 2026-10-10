# Reproducing this benchmark

## 1. Environment

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/hwinfo.py
```

`requirements.txt` pins `torch==2.6.0` without a local build suffix (e.g. `+cu124`) on purpose: PEP 440 matches a bare version against any locally-built variant, so this installs correctly whether `pip` resolves it to a plain CPU wheel or you've pointed `pip` at a CUDA-specific index first (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124` *before* `pip install -r requirements.txt`, for a matching GPU setup) -- without hardcoding one CUDA version that would fail to resolve on PyPI directly or on hardware with a different driver. `results/hardware.json` records the hardware this run used.

**Windows note:** installing `torch` can fail with `OSError: [Errno 2] No such file or directory: ...` on a deeply nested install path (its internal header files are long enough to exceed the default 260-character `MAX_PATH`). Install from a short path (e.g. `C:\dev\loganomaly`, not a deeply nested temp/profile directory) or enable [Windows Long Path support](https://pip.pypa.io/warnings/enable-long-paths) first.

## 2. Data

```
python scripts/download_data.py
python scripts/download_sample_data.py
```

`download_data.py` fetches HDFS_v1 and BGL from Zenodo (Loghub, record 8196385) into `data/raw/`, with retry/resume and a fallback to record 3227177 if the primary URL fails. See `data/raw/INVENTORY.md` for the inspected format and line counts, and `data/DOWNLOAD_FAILED.md` (only present if a download failed) for manual recovery steps.

`download_sample_data.py` fetches the small 2k-line HDFS/BGL sample logs (from the Loghub GitHub repo) into `data/sample/` -- these are required for every `--selftest` run below and are not part of the Zenodo download or committed to this repo (both are gitignored, like all raw data).

## 3. Pipeline

```
python src/parse.py --dataset HDFS_v1
python src/parse.py --dataset BGL
python src/sessionize.py --dataset HDFS_v1
python src/sessionize.py --dataset BGL
python scripts/make_data_stats.py
python src/split.py --dataset HDFS_v1 --seeds 0 1 2 3 4
python src/split.py --dataset BGL --seeds 0 1 2 3 4
python src/run_all.py
python src/analysis.py
python src/make_figures.py
python scripts/make_tables.py
```

Every script accepts `--selftest`, which runs the same code against the 2k-line sample logs in `data/sample/` (see Section 2) instead of the full datasets — use this to validate the pipeline quickly before committing to the full run. A full `--selftest` pass of every script (after `download_sample_data.py`) takes a few minutes on CPU alone; see `results/audit/AUDIT.md`'s reproducibility-smoke-test section (Task 6) for measured timings from a clean clone.

`src/run_all.py` is resumable: it skips any `results/full/<dataset>/<split>/<arm>/seed<k>.json` that already exists, so an interrupted run can be restarted with the same command.

**`deeplog` reproducibility note (Step 13).** The released `results/full/` `deeplog` cells were trained before `src/arms/deeplog.py` fully seeded its training (random/numpy/torch/cuDNN); re-running them from scratch reproduces each cell's AUPRC/AUROC within ordinary seed-to-seed variance (its across-seed AUPRC standard deviation ranges from 0.0055 to 0.0903 depending on dataset/split, from `results/full/summary.csv`) but not bit-for-bit. Every other arm, and any `deeplog` cell trained after Step 13, is exactly reproducible given the same seed.

## 4. Runtimes measured on this machine

See `results/run.log` for the actual per-stage timings from this run, and `results/hardware.json` for the hardware. Headline figures (full detail in those files):

- Drain3 parsing: HDFS_v1 (11.18M lines) and BGL (4.75M lines) each parse in a few minutes at roughly 30,000-70,000 lines/second on this machine's CPU.
- The full `run_all.py` matrix (2 datasets x 2 splits x 6 arms x 5 seeds, plus 4 LLM sample-based cells) — see `results/run.log` for the measured wall-clock total and `results/full/*/*/*/seed*.json`'s `fit_seconds` field for the slowest individual arm.

