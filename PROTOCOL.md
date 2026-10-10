# PROTOCOL v0.1 — Does log anomaly detection performance depend on how the data is split?

Pre-registered before any experiment is run. This benchmark tests whether published log-based anomaly detection results depend on how the data was split. Many methods report very high F1 on HDFS and BGL when evaluations shuffle data randomly and tune thresholds on the test set. Under a chronological split with thresholds chosen on validation data, we test whether scores fall, unseen log templates appear, and method rankings change.

Datasets: HDFS_v1 and BGL (Loghub). Citation: Zhu et al., "Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics", ISSRE 2023; Jiang et al., "A Large-scale Evaluation for Log Parsing Techniques: How Far are We?" (Loghub-2.0), ISSTA 2024.

Hardware for this run (`results/hardware.json`): NVIDIA GeForce GTX 1660 Ti, 6 GB VRAM, 12 logical CPUs, ~19.85 GB RAM → "GPU present (>=6GB VRAM)" tier (Section 6 below). All arms run as specified; LLM arm uses `Qwen/Qwen2.5-1.5B-Instruct` in fp16.

## 5.1 Parsing and sessionisation

- Parse both logs with **Drain3** (similarity threshold 0.4, depth 4), masking block ids, IPs, hex values, and integers before clustering. Output: `data/processed/<dataset>_structured.parquet` with columns `line_id, timestamp (UTC), label_line (BGL only), block_id (HDFS only), template_id, template`.
- **HDFS sessions**: one session per block id. Label comes from `anomaly_label.csv` (`Normal`/`Anomaly`). Session timestamp = timestamp of the block's first line. A log line that mentions multiple block ids (confirmed present in the data, see `data/raw/INVENTORY.md`) is attached to every block id it names. All ~575k sessions are kept.
- **BGL sessions**: fixed, non-overlapping windows of 100 consecutive lines in time order (primary). A window is anomalous if any line in it carries a non-`-` alert tag. A second, 1-hour time-window variant is also produced and stored as a robustness check (Section 9), never used to decide H1–H5.
- `results/data_stats.json` records, per dataset: number of sessions, anomaly rate, number of distinct templates.

## 5.2 Splits (the independent variable)

- **RANDOM**: shuffle sessions with the run's seed, then split 60/20/20 into train/val/test.
- **CHRONO**: sort sessions by session timestamp, then take the first 60% as train, next 20% as val, last 20% as test.
- Both conditions use identical proportions and the same 5 seeds (0–4); only the ordering before the cut differs. CHRONO's ordering does not depend on the seed (it is deterministic), but seed still controls any downstream stochastic steps (model init, subsampling) so CHRONO is re-run per seed for comparability.
- For unsupervised / one-class arms (`pca`, `iforest`, `ocsvm`, `deeplog`, `semantic_knn`), training uses only sessions labelled normal in the train split (matches DeepLog's standard practice). Validation contains both classes and is used for threshold selection. `supervised_lgbm` uses all labels in train (it is a ceiling reference, not a competitor).
- `unseen_template_rate` = fraction of test-split template *occurrences* (not distinct templates) whose template id never appears in train, computed per dataset × split condition × seed, stored in `results/<dataset>/<split>/unseen_template_rate_seed<k>.json`.
- **Chronological-integrity check**: `src/split.py` asserts, for every CHRONO split, `max(train.timestamp) <= min(val.timestamp)` and `max(val.timestamp) <= min(test.timestamp)`. This assertion is the enforcement mechanism for the protocol's own rule that chronological splits must not leak future sessions into training ("chronological means chronological").

## 5.3 Arms

Each arm lives in `src/arms/<name>.py` and implements `fit(train_sessions)` and `score(sessions) -> anomaly_score_per_session` (higher = more anomalous).

| Arm | Type | Notes |
|---|---|---|
| `pca` | classic unsupervised | Count vector of templates per session → PCA (Xu et al. 2009), score = squared prediction error. Components kept to explain 95% variance. |
| `iforest` | classic unsupervised | Isolation Forest on count vectors, 200 trees. |
| `ocsvm` | classic one-class | One-Class SVM (RBF) on TF-IDF count vectors; train subsampled to 50k sessions if larger. |
| `deeplog` | deep, sequence | LSTM next-template prediction on normal sessions (window 10, hidden 64, 2 layers). Score = fraction of positions where the true next template is not in the top-9 predictions. Training capped at 100k windows (sampled; documented if triggered). |
| `semantic_knn` | embedding | `sentence-transformers/all-MiniLM-L6-v2` encodes each distinct template once. Session vector = count-weighted mean of its template embeddings. Score = mean cosine distance to the 10 nearest normal training session vectors. Expected to be the most robust arm to unseen templates, since it can embed a template never seen in training. |
| `llm_zeroshot` | small LLM | `Qwen/Qwen2.5-1.5B-Instruct`, fp16 on GPU. Sample-based: 400 test sessions per dataset per split condition (200 anomalous, 200 normal), seed 0 only. Shown the session's template sequence (truncated to 40 lines), asked for an anomaly probability 0–100; the parsed number is the score. Reported separately, never in the main ranking table. |
| `supervised_lgbm` | supervised reference | LightGBM on count vectors with true labels. A ceiling showing available headroom, not a competitor in H1/H2/H4. |

## 5.4 Metrics

Primary, threshold-free: **AUROC** and **AUPRC** (average precision) on the test split.

Secondary, thresholded:
- `F1_val`: threshold = the value maximising F1 on validation, applied unchanged to test. The honest number.
- `F1_oracle`: best achievable F1 on the test set itself (threshold tuned on test). The number many papers effectively report.
- `F1_oracle - F1_val` = threshold-leakage gap.
- Precision and recall at the validation threshold.

All metrics computed per seed (5 seeds: 0–4). Report mean and 95% bootstrap CI (1000 resamples of test sessions, seed 0 only, for the CI resampling) and the across-seed standard deviation.

## 5.5 Hypotheses and decision rules (pre-registered before any experiment)

- **H1 (split inflation):** For every arm, `AUPRC_random - AUPRC_chrono >= 0.05` with the 95% CI excluding zero, on at least one dataset. Confirmed if >= 4 of the 6 non-LLM arms meet it on HDFS or BGL.
- **H2 (ranking instability):** The rank order of the 6 non-LLM arms by AUPRC differs between RANDOM and CHRONO on at least one dataset (Kendall tau < 0.8 between the two rankings).
- **H3 (unseen templates explain degradation):** Across (dataset x seed x arm) cells, the per-cell drop in AUPRC from RANDOM to CHRONO correlates positively with the cell's unseen_template_rate (Spearman rho > 0.4, p < 0.05).
- **H4 (semantic robustness):** The `semantic_knn` arm's RANDOM->CHRONO drop in AUPRC is smaller than `deeplog`'s and `pca`'s on both datasets, with CIs for the difference of drops excluding zero.
- **H5 (threshold leakage):** Mean `F1_oracle - F1_val` across arms >= 0.10 under CHRONO on at least one dataset, CI excluding zero.

Decision rules:
- Each hypothesis gets exactly one verdict: CONFIRMED / NOT CONFIRMED / INCONCLUSIVE (CI straddles the bar but the point estimate meets it). No re-runs to move a verdict once written.
- If H1 is NOT CONFIRMED on both datasets, the paper's framing becomes "chronological evaluation does not change conclusions on HDFS/BGL — a negative result with a reproducible protocol," and H3-H5 are still reported. This is a valid outcome.
- If an arm fails to run after two attempts and one simplification (e.g. smaller model, smaller sample), mark it SKIPPED in the results and proceed.
- Bug fixes discovered after a verdict is written are allowed only if documented in `DECISIONS.md`, with the superseded result moved to `results/superseded/`.

## Compute plan

- Total wall-clock target: under 14 hours.
- Single training run cap: 90 minutes. If exceeded, halve the training set (documented in `DECISIONS.md`) and continue.
- Order: parse -> sessionize -> split (all seeds) -> run_all (loops dataset x split x arm x seed, resumable, skips completed cells) -> analysis -> figures -> paper.
- LLM arm runs last, seed 0 only, after all other arms are complete for both datasets.

## Exact commands

```
python src/parse.py --dataset HDFS_v1
python src/parse.py --dataset BGL
python src/sessionize.py --dataset HDFS_v1
python src/sessionize.py --dataset BGL
python src/split.py --dataset HDFS_v1 --seeds 0 1 2 3 4
python src/split.py --dataset BGL --seeds 0 1 2 3 4
python src/run_all.py
python src/analysis.py
python src/make_figures.py
```

Each script also accepts `--selftest`, which runs the same logic against `data/sample/HDFS_2k.log` / `data/sample/BGL_2k.log` to validate the pipeline before touching full data.
