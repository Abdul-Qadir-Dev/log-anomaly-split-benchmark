# Summary

This document summarises the outcomes, the deviations from the pre-registered protocol, the post-hoc analyses, and the open questions, for readers of the paper.

## 1. Outcome

We pre-registered and ran a benchmark testing whether random versus chronological data splitting changes log anomaly detection results on HDFS_v1 and BGL (Loghub), across six methods and five seeds (120 cells, all completed).

Split inflation is dataset-dependent and follows template drift. It is not a general property of chronological evaluation. On BGL, where 77% of chronological test-template occurrences are unseen in training, chronological splitting lowers both AUPRC and AUROC substantially and reorders the best-performing method, and this holds at three different chronological cutoffs. On HDFS_v1, where the 55-template vocabulary is almost entirely covered within the first 60% of the timeline, it does not. HDFS_v1's apparent AUPRC gains under chronological splitting turned out, in a post-hoc analysis, to be mostly a test-set prevalence artifact (AUROC stays nearly flat) and were not stable across cutoffs.

The AUPRC drop correlates with unseen-template rate (Spearman rho = 0.80, p < 0.001; 0.83 before the deeplog scoring correction described in Section 10). We report this as consistent with the drift mechanism rather than as proof of it, since the effective sample size for generalising across datasets is two. Threshold leakage (test-tuned versus validation-tuned F1) was small on both datasets.

The exploratory LLM-scoring arm was skipped in the main run (the 1.5B model could not be run reliably on the machine used) and later completed with a smaller 0.5B model as a sample-based, post-hoc analysis with no verdict attached (Section 8). A later diagnosis found and corrected a scoring defect in deeplog; every deeplog cell was re-run and every verdict recomputed, and none changed (Section 10).

## 2. The five verdicts

| Hypothesis | Bar | Verdict |
|---|---|---|
| H1, split inflation | AUPRC_random minus AUPRC_chrono at least 0.05, CI excluding 0, on at least 4 of 6 arms, on at least one dataset | CONFIRMED (BGL: 5 of 6 arms) |
| H2, ranking instability | Kendall tau below 0.8 between RANDOM and CHRONO rankings, on at least one dataset | CONFIRMED (BGL: tau 0.20 pre-registered, 0.47 corrected) |
| H3, unseen templates track degradation | Spearman rho above 0.4, p below 0.05, across dataset x seed x arm cells | CONFIRMED (rho 0.83 pre-registered, 0.80 corrected) |
| H4, semantic-arm robustness | semantic_knn's drop smaller than pca's and deeplog's, both datasets, CI excluding 0 | NOT CONFIRMED |
| H5, threshold leakage | mean(F1_oracle minus F1_val) at least 0.10 under CHRONO, CI excluding 0, on at least one dataset | NOT CONFIRMED |

The pre-registered record is `results/full/VERDICTS.md`. After the deeplog scoring correction (Section 10) the verdicts were recomputed into `results/full/VERDICTS_v2.md`; all five are identical to the pre-registered ones.

## 3. Deviations from protocol

All are logged with reasons in `DECISIONS.md`. In brief:

- Hardware tier. The GTX 1660 Ti reports exactly 6144 MiB, the stated threshold for the GPU-present tier, so that tier was used.
- Download sanity check. `scripts/download_data.py` assumed both dataset archives exceed 100 MB; BGL's compresses to 57.5 MB, so a complete download was flagged as failed. The threshold was corrected; the data was unaffected (verified with `zipfile.testzip()`).
- `.gitignore` excludes `data/raw/` except `INVENTORY.md`, so the inspection notes are tracked while the raw logs are not.
- BGL parsing. `parse_bgl_line` originally rejected lines with an empty message field (9 fields instead of 10), wrongly treating 34,470 real lines such as a bare `RAS KERNEL FATAL` as unparseable. Fixed before any verdict existed; BGL was re-parsed from scratch with zero unparseable lines.
- CI for cross-condition differences (H1, H4, H5). The protocol defines a bootstrap CI for a single AUPRC estimate but not for a difference between split conditions. We used a seed-level paired bootstrap (resample the five per-seed differences 1000 times, seed 0). This was decided before any analysis ran.
- `semantic_knn` implementation. sklearn's brute-force `NearestNeighbors` took about 45 minutes per cell, which would have been about 7.5 hours for HDFS alone. It was replaced by a batched GPU matrix multiply that computes the same quantity. Verified on `--selftest` to produce identical numbers, and re-verified later against a CPU re-implementation (Section 7).
- `llm_zeroshot` was skipped in the main run (Section 4).
- deeplog scoring was corrected after a verdict existed, under the protocol's rule for bug fixes found after a verdict, with the originals preserved (Section 10).

## 4. The skipped arm

Only `llm_zeroshot` (a `Qwen2.5-1.5B-Instruct` zero-shot scorer) was skipped, and only in the main run. It failed in sequence: a GPU out-of-memory error loading the 1.5B model on the 6 GB GPU, which is also the Windows display adapter and so shares its memory with the desktop; the same error in a fresh process with the `expandable_segments` setting, which is not supported on Windows; a CPU fp32 fallback that hit `OSError: The paging file is too small`; and a multi-candidate fallback (1.5B GPU, then 0.5B GPU, then 0.5B CPU) that worked once on a 30-sample selftest but took about 33 minutes for 30 generations and failed again on the pagefile error. This is a resource limit of the machine, not a defect in the arm, and it exceeded the protocol's "two attempts and one simplification" rule for marking an arm skipped. It does not affect H1 to H5, which are defined only over the six non-LLM arms.

## 5. Results most worth independent scrutiny

1. HDFS_v1 shows negative AUPRC drops for pca, iforest and ocsvm (chronological beats random), the opposite of H1's direction, on one dataset only. The post-hoc prevalence and AUROC analysis (Section 7) attributes most of this to the test-set anomaly rate nearly halving between conditions. The stratified-by-time random split that would settle it was not run.
2. `semantic_knn` scores below chance (AUROC 0.34) on BGL/CHRONO, a surprising failure of the arm we expected to be most robust to template drift. The within-dataset drift test (Section 7) supports the mechanism inside each class but does not explain the below-chance result; a second factor is needed. See `results/ANOMALIES.md` item 2.
3. The H4 and H5 confidence intervals rest on five seeds. A paired bootstrap over five values is a thin basis for a 95% CI. The NOT CONFIRMED verdicts are reasonably solid, since the point estimates were also far from the bar in most cases, but any borderline CI in the detail tables should be read with caution.

## 6. Weaknesses a reviewer would raise

- Two datasets, both heavily studied and possibly atypical of real deployments.
- BGL's window label ("anomalous if any line in the 100-line window carries an alert tag") is coarse and likely explains iforest's below-chance AUROC on BGL/RANDOM (`results/ANOMALIES.md` item 3), a confound unrelated to the paper's question.
- One Drain3 configuration (similarity 0.4, depth 4). A different parser or settings would change the template vocabulary and with it `unseen_template_rate`.
- The LLM arm has only a single-seed, 200-session-per-cell result, far below the five-seed scale of the other six arms, so the paper can offer only a suggestive data point on zero-shot LLM scoring.
- ocsvm's 50k-session training subsample (prescribed by the protocol) is about 15% of HDFS's normal training pool, which may understate its performance relative to arms trained on the full set.
- The chronological condition is a single split per dataset in the main run. The rolling-origin check (Section 7) shows BGL's result is stable across cutoffs and HDFS_v1's is not.

## 7. Post-hoc analyses

These were run after the pre-registered experiments, with `results/full/` frozen except where Section 10 says otherwise. Detail is in `results/audit/AUDIT.md`.

Verification of the main run:

- BGL parse re-verified by re-reading all 4,747,963 lines: zero dropped, 348,460 alert-labelled lines (7.34%), 1,346 templates. The empty-content template has exactly 34,470 occurrences, the same count the original parser bug dropped, which confirms the fix is complete.
- The `semantic_knn` GPU implementation was compared with a CPU brute-force re-implementation on 5,000 HDFS/CHRONO sessions: maximum score difference 5.4e-7, AUPRC 0.97793 versus 0.97792, AUROC identical at 0.99968.
- A fresh-clone reproducibility test found two real bugs: a `requirements.txt` pin (`torch==2.6.0+cu124`) that does not exist on PyPI, and no script to fetch the selftest fixture files. Both were fixed and every selftest passes on a fresh clone.
- A selftest-contamination check regenerated `summary.csv`, `paper/tables.md` and all figures from the frozen per-seed files and found them identical to the committed versions. One latent bug was fixed: `make_figures.py` wrote to `figures/` even in selftest mode; it now writes to `figures/selftest/`.

Interpretation checks:

- Prevalence and AUROC. HDFS_v1's test-set anomaly prevalence is 2.96% under RANDOM and 1.46% under CHRONO; BGL's is 9.73% and 8.49%. AUROC, which does not depend on prevalence, is nearly flat across split conditions for the HDFS_v1 arms whose scoring was not revised, while AUPRC swings by 0.09 to 0.33. BGL's AUROC drops track its AUPRC drops closely. So HDFS_v1's "chronological splitting helps" result is mostly a prevalence artifact, and BGL's degradation is a real loss of ranking quality. Files: `results/full/addendum/prevalence.csv`, `auroc_drops.md`.
- Within-dataset drift test. On BGL/CHRONO, a session's unseen-template rate correlates positively with its `semantic_knn` score inside both the normal and the anomalous class (rho 0.54 and 0.62), as the mechanism predicts, but both classes have high unseen rates (76.8% and 87.5%), so unseen rate alone is at chance for separating them (AUROC 0.50) and cannot explain the below-chance result. On HDFS_v1/CHRONO, normal sessions have exactly 0% unseen templates, anomalous sessions average 9.9%, and unseen rate alone scores AUROC 0.72. File: `results/audit/drift_mechanism/REPORT.md`.
- Rolling-origin check. All six non-LLM arms were re-run at chronological cutoffs of 50% and 70% (seed 0) alongside the main 60%. BGL's degradation holds at every cutoff (AUPRC 0.06 to 0.34 throughout). HDFS_v1's does not: pca's AUPRC is 0.485 at 50%, 0.920 at 60% and 0.994 at 70%, with `unseen_template_rate` at zero throughout. File: `results/full/addendum/rolling_origin/rolling_origin_table.md`.
- Unseen-template sanity check. Jaccard token similarity of every unseen BGL/CHRONO template against every training template: only 1.55% of unseen occurrences have a close (above 0.8) training match, so the drift is mostly new log content, with a few high-frequency exceptions that are Drain3-fragmented variants. File: `results/full/addendum/unseen_templates_bgl.md`.
- Baseline sanity table. Absolute RANDOM-split metrics per arm sit next to a literature column that is marked n/a throughout, because we did not verify exact published values against a source and preferred not to quote unverified numbers. File: `results/full/addendum/baseline_sanity.md`.

Paper work: the headline was reframed to "dataset-dependent and tracks template drift"; the HDFS_v1 negative drops got their own paragraph with the prevalence evidence; H3 is worded "consistent with"; the single-split limitation is stated with the rolling-origin results as mitigation; the abstract was cut to 196 words; LogAnomaly, LogBERT, the Landauer survey and He et al. (ISSRE 2016) were added to related work, with bib fields we did not verify marked `% CHECK`; internal references were removed from visible prose.

## 8. The LLM arm, completed later

A retry with `Qwen2.5-0.5B-Instruct` (fp16, GPU) completed all four dataset x split cells on 200 sessions each (100 per class), seed 0. AUROC was 0.44 and 0.37 on HDFS_v1 (random, chrono) and 0.75 and 0.65 on BGL, the opposite difficulty pattern from most of the six main arms. It is reported in `paper/draft.md` Section 5.7 and `results/audit/llm_table.md` and is not part of any ranking or verdict.

## 9. Runtime and compute

Hardware: NVIDIA GeForce GTX 1660 Ti (6 GB, shared with the display), 12 logical CPUs, about 19.85 GB RAM, Windows (`results/hardware.json`).

Script execution time for the main run, reconstructed from script logs and commit timestamps, excluding idle time between work sessions:

| Stage | Approximate wall-clock |
|---|---|
| Environment setup | 25 min |
| Data download (HDFS_v1 1.58 GB, BGL 743 MB, with retries) | 15 min |
| Drain3 parsing (both datasets, about 16M lines) | 6 min |
| Pipeline development and selftests | 30 min |
| Full experiment matrix, first attempt (stopped for the `semantic_knn` performance fix; 21 of 124 cells kept) | 90 min |
| Full experiment matrix, second attempt (GPU out of memory from an oversized batch) | 15 min |
| Full experiment matrix, third attempt (all 120 non-LLM cells) | 142 min |
| `llm_zeroshot` attempts | 45 min |
| Robustness checks | 23 min |
| Analysis, figures, tables | 2 min |
| Total | about 6.1 hours, inside the protocol's compute budget |

## 10. The deeplog scoring correction

A full-strength DeepLog check (HDFS_v1, seed 0, 1M-window cap, up to 15 epochs with early stopping, hidden size 128) showed RANDOM-split AUPRC falling from 0.779 to 0.380 while CHRONO barely moved, which looked like a reversal of DeepLog's RANDOM-to-CHRONO gap. That result was not accepted until its cause was found.

The diagnosis retrained both the main-run configuration (hidden 64, 100k windows, 5 fixed epochs) and the full-strength configuration on both HDFS_v1 splits, keeping the checkpoints this time (neither earlier run had saved one). Findings:

- Top-9 scoring is saturated. Between 97.5% and 99.2% of test sessions score exactly zero at k = 9 in every configuration, including the one that matches the main run. A sweep over k in {1, 2, 3, 5, 7, 9} shows AUROC and AUPRC varying substantially and non-monotonically with k.
- No loading or early-stopping bug. State-dict hashes are identical before and after loading and after scoring for all four models; early stopping used normal-only validation sessions.
- Mean negative log-likelihood of the true next template, a continuous score, gives AUROC 0.97 to 0.99 in every configuration, against 0.70 to 0.88 for top-9.

Because the main-run configuration itself was saturated far above the 80% threshold set for this diagnosis, the main-run deeplog cells were treated as defective under the protocol's rule for bugs found after a verdict:

1. The original deeplog cells (both datasets, both splits, five seeds) and a snapshot of `summary.csv` were copied to `results/superseded/deeplog_v1/`.
2. `src/arms/deeplog.py` was corrected: mean NLL is the primary score, top-9 is kept as a secondary diagnostic field, and early stopping on normal-only validation loss (patience 3, up to 15 epochs) replaced the fixed 5-epoch schedule.
3. All 20 deeplog cells were re-run; no other arm was touched.
4. `results/full/summary.csv` was regenerated. `results/full/VERDICTS.md` was left as the pre-registered record and the verdicts were recomputed into `results/full/VERDICTS_v2.md`.

None of the five verdicts changed. The evidence under them did move in places: HDFS_v1's deeplog drop went from indistinguishable from zero (0.014, CI spanning zero) to 0.074 with CI [0.046, 0.106], which clears H1's bar for one arm; HDFS_v1's ranking stability went from tau 1.00 to 0.73; BGL's tau went from 0.20 to 0.47; H3's rho went from 0.83 to 0.80; and HDFS_v1's H5 gap went from 0.061 to 0.099, still just under the 0.10 bar. The apparent "stronger model reverses the gap" result did not survive: with NLL scoring, deeplog on HDFS_v1 behaves like most other arms, with RANDOM modestly ahead of CHRONO. Full comparison: `results/full/addendum/deeplog_diagnosis/DIAGNOSIS.md`.

A consistency sweep then regenerated every derived artefact from the corrected `summary.csv` and updated every deeplog-dependent number in the paper, giving the pre-registered value alongside the corrected one. Two secondary observations came out of it, neither changing a verdict: deeplog's own threshold-leakage gap under CHRONO grew from 0.003 to 0.067 on BGL and from 0.013 to 0.246 on HDFS_v1, an expected consequence of replacing a near-constant score with a continuous one; and the corrected deeplog's BGL rolling-origin numbers move more across cutoffs than any other arm (AUPRC 0.118, 0.243, 0.076 at 50%, 60%, 70%), which we attribute to single-seed training variance since it stays inside the 0.06 to 0.34 band reported for BGL as a whole.

## 11. Seeding

deeplog was the one arm whose training was not fully seeded. `src/arms/deeplog.py` now seeds random, numpy, torch and cuDNN from the cell's seed and sets cuDNN to deterministic mode. Two separate-process runs on the same selftest split and seed now give byte-identical AUROC, AUPRC and F1 (`results/audit/deeplog_determinism_check.json`). The released deeplog cells were trained before this fix, so re-running them reproduces each result within seed-to-seed variance (across-seed AUPRC standard deviation 0.0055 to 0.0903 depending on dataset and split) but not bit for bit. Every other arm, and any deeplog cell trained after the fix, is exactly reproducible given the seed. The released cells were not re-run, because the recorded scores are valid draws and re-running would not be expected to change any verdict.

## 12. Suggested next experiments

- A time-bucket-stratified random split on HDFS_v1, to separate anomaly clustering from template coverage as the cause of its negative drops.
- Identify the second factor behind `semantic_knn`'s below-chance AUROC on BGL/CHRONO, for example by testing which specific unseen templates dominate normal sessions' embeddings.
- Re-run iforest on BGL with a stricter alert-fraction label instead of "any alert line".
- A third Loghub dataset with a drift profile between HDFS_v1 and BGL, to see whether the unseen-template relationship (H3) holds across more than two points.
- Run the LLM arm at full scale on a machine with a dedicated GPU.
- A rolling-origin design averaged over several chronological cutoffs as the main evaluation rather than a single cut.
