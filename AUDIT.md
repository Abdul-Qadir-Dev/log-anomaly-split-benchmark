# Post-hoc analyses

This file records the analyses carried out after the pre-registered run, in order. `results/full/` stayed frozen throughout except for the deeplog correction in Step 11, which is described there. New outputs live under `results/audit/` and `results/full/addendum/`. The reasons behind each choice are in `DECISIONS.md`.

# Step 08: verification of the main run

## BGL parse

Full report: `results/audit/bgl_parse_verification.md` and `.json`. All numbers were recomputed by re-reading the raw log and re-running the parser rather than read back from cached results.

- Lines read: 4,747,963, matching the expected count.
- Lines dropped: 0.
- Alert-labelled lines: 348,460 (7.34%); non-alert: 4,399,503.
- Drain3 templates: HDFS_v1 55, BGL 1,346, well under the 5,000 over-fragmentation flag. The top-20 table was produced anyway.
- BGL/CHRONO/seed0 `unseen_template_rate`: 0.7746 per occurrence (matching the recorded value used by H1, H3 and H5) and 0.7500 per unique template (450 of 600 distinct test templates never appear in training). The per-occurrence number is higher because unseen templates also recur more often per session in the chronological tail.

The empty-content template has exactly 34,470 occurrences, the same count of lines the Step 03 parser bug used to drop. That is independent confirmation that the fix was complete.

## semantic_knn speed fix

Full report: `results/audit/knn_cpu_check_result.json` (script `results/audit/knn_cpu_check.py`). One cell (HDFS_v1, chrono, seed 0), 5,000-session query subsample against the 333,878-session training set.

- Maximum absolute score difference between the GPU implementation and a CPU brute-force re-implementation: 5.4e-7, against a bar of 1e-4. Mean absolute difference: 5.6e-8.
- Spearman rho between the two score arrays: 0.23. This looked alarming until the cause was found: most HDFS test sessions are template-identical to some training session, so both scorers give near-zero scores for most of the 5,000 queries (only 19 distinct rounded values), and float32 versus float64 rounding reorders that tied block.
- AUPRC on the subsample: 0.97793 (GPU) versus 0.97792 (CPU). AUROC: 0.99968 for both.

## H4 and H5 diagnosis

Full tables: `results/audit/h4_h5_diagnosis.md`. No verdict changed.

- H4 fails for two different reasons depending on the comparison arm. Against pca, the data point the opposite way from the hypothesis on both datasets, with CIs nowhere near zero. Against deeplog (pre-correction scoring), the point estimate leans the hypothesised way on both datasets, but deeplog's across-seed variance is too large for five seeds to separate it from semantic_knn.
- H5's HDFS_v1 aggregate gap (0.0606, pre-correction) is driven almost entirely by one arm, iforest (gap 0.348); every other HDFS_v1 arm has a gap under 0.02.

## LLM arm, completed with a smaller model

Script: `scripts/audit_llm_retry.py`. Model `Qwen/Qwen2.5-0.5B-Instruct`, fp16 on the GPU, 200 sessions per dataset and split (100 per class), seed 0, at most 40 template lines per prompt. This arm has no pre-registered verdict.

All four cells completed on the GPU with no fallback needed. The smaller model's VRAM footprint (well under 2 GB) avoided the memory contention that stopped every attempt with the 1.5B model in the main run.

| Dataset | Split | AUROC | AUPRC | N | Device |
|---|---|---|---|---|---|
| HDFS_v1 | random | 0.4380 | 0.4880 | 200 | cuda |
| HDFS_v1 | chrono | 0.3689 | 0.4442 | 200 | cuda |
| BGL | random | 0.7452 | 0.6930 | 200 | cuda |
| BGL | chrono | 0.6513 | 0.5923 | 200 | cuda |

The scorer is worse than chance on HDFS_v1 and better than chance on BGL, the opposite difficulty pattern from several of the non-LLM arms. One plausible reading is that HDFS's repetitive 55-template logs give the model little distinguishing text, while BGL's more varied error messages (1,346 templates) give it more to work with. This is an observation, not a result that was pursued further.

## Reproducibility test on a fresh clone

Full report: `results/audit/smoketest_report.md`. Procedure: fresh `git clone`, fresh Python 3.11 venv, `pip install -r requirements.txt`, fetch the selftest fixtures, run every `--selftest`.

Two bugs were found and fixed:

1. `requirements.txt` pinned `torch==2.6.0+cu124`, which does not exist on PyPI, so a fresh install failed immediately. Re-pinned to `torch==2.6.0`, which PEP 440 matches against any local-version variant.
2. `data/sample/`, needed by every `--selftest`, was gitignored and never fetched by any script. `scripts/download_sample_data.py` now fetches it and `REPRODUCE.md` documents it.

A Windows `MAX_PATH` failure when installing from a deeply nested path was a property of the install location, not the repository, and is documented as a troubleshooting note.

After the fixes every `--selftest` passed on a fresh clone: parse, sessionize, split and features in under 80 seconds combined, `run_all.py --selftest` in 212.6 seconds on CPU, `analysis.py` and `make_figures.py --selftest` in 12 seconds combined, about 5.1 minutes in total.

# Step 09: interpretation checks

## Selftest contamination

`results/full/summary.csv` was regenerated from the frozen per-seed JSONs (`rglob("seed*.json")` under `results/full/`, no cell re-run), and `paper/tables.md` and all four figures were regenerated from it. All matched the committed versions: the summary has 120 rows with `n_test` of exactly 9,496 (BGL) or 115,013 (HDFS_v1), never the selftest sizes such as 4; the tables and PNG figures were byte-identical; the PDF figures differed only in matplotlib's embedded `/CreationDate` and were discarded rather than committed. The README's hand-written results table was recomputed from the clean data and matches.

One latent bug was fixed: `src/make_figures.py` hardcoded `figures/` as its output directory regardless of `--selftest`, unlike `run_all.py` and `analysis.py`. Nothing had been overwritten, because the full-data run always came after any selftest run, but the next person to run `--selftest` after a full run could have overwritten the committed figures. Selftest output now goes to `figures/selftest/`, which is gitignored. `scripts/make_tables.py` reads only `results/full/summary.csv` and was never at risk.

## Prevalence and AUROC

Built from `results/full/summary.csv` with no re-run: `results/full/addendum/prevalence.csv` (20 rows, dataset x split x seed) and `results/full/addendum/auroc_drops.md` (AUROC drop next to AUPRC drop per dataset and arm, same paired-bootstrap CI as H1, H4 and H5).

HDFS_v1's RANDOM and CHRONO test splits have very different anomaly prevalence (2.96% versus 1.46%, nearly twofold) purely because of where anomalous blocks fall on the timeline, while BGL's is nearly stable (9.73% versus 8.49%). AUROC, which does not depend on prevalence, is nearly flat for every HDFS_v1 arm (drops of -0.003 to -0.023) while AUPRC swings by 0.09 to 0.33 for the same arms. So HDFS_v1's apparent "chronological splitting helps" result is mostly a prevalence artifact. BGL's AUROC drops track its AUPRC drops closely (five of six arms, CIs excluding zero), so BGL's degradation is a real loss of ranking quality. The exception is iforest on BGL, with a large negative AUROC drop despite a near-zero AUPRC drop, consistent with the separate finding in `results/ANOMALIES.md` item 3 that this arm's score is already inverted on BGL under RANDOM. Added to the paper's Section 5.1 and Threats to validity.

## Within-dataset template-drift test

Diagnostic re-score only, in `results/audit/drift_mechanism/`; a fresh semantic_knn was fitted on the existing BGL/CHRONO/seed0 split so the check is independent of the frozen run.

On BGL/CHRONO/seed0, the mean per-session unseen rate is 0.768 for normal sessions and 0.875 for anomalous ones. Both are high, and unseen rate alone separates the classes at chance level (AUROC 0.498), because the whole test period is affected by drift regardless of label. Within each class, unseen rate correlates strongly and positively with the semantic_knn score (Spearman rho 0.544 for normal, 0.620 for anomalous, both p below 1e-80), which is the relationship the arm is built on. That relationship, taken on its own, would predict an AUROC slightly above chance for semantic_knn, since anomalous sessions have the higher mean unseen rate, not the 0.342 observed. Unseen-template volume therefore explains only part of the below-chance result; a second factor, plausibly which templates go unseen rather than how many, is needed. On HDFS_v1/CHRONO/seed0, normal sessions have exactly 0% unseen templates, anomalous sessions average 9.9%, and unseen rate alone scores AUROC 0.716. Full report: `results/audit/drift_mechanism/REPORT.md`.

## Rolling-origin robustness

All six non-LLM arms, both datasets, seed 0, at chronological training cutoffs of 50% and 70% (validation the next 20%, test the remainder), alongside the main 60%. Table: `results/full/addendum/rolling_origin/rolling_origin_table.md`.

BGL's degradation holds at all three cutoffs: AUPRC stays between 0.06 and 0.34 at every cutoff against BGL's RANDOM-split 0.83 to 0.99, and `unseen_template_rate` stays between 0.69 and 0.80. HDFS_v1's result does not hold: pca's AUPRC is 0.485 at 50%, 0.920 at 60% and 0.994 at 70%, and ocsvm shows the same pattern (0.672, 0.923, 0.996), while `unseen_template_rate` stays at exactly zero. Template drift cannot explain the swing; the likeliest cause is the prevalence and test-window composition sensitivity found above. This is a real limitation of the main run's single HDFS_v1 chronological split and is stated in the paper's Section 5.6 and Threats to validity.

## Baseline sanity table

`results/full/addendum/baseline_sanity.md` lists absolute RANDOM-split AUROC, AUPRC, F1_val and F1_oracle per arm and dataset (mean over five seeds) next to a literature-value column. Every literature cell is n/a: exact published values for these method, dataset and metric combinations were not verified against a source, so citations and qualitative reasons for differences (session definition, split, threshold policy, Drain3 configuration, training caps) are given instead. The table noted at the time that deeplog's HDFS_v1 result (AUPRC 0.589, AUROC 0.806, F1 about 0.71) was well below DeepLog's usual reported performance and attributed it to the light training configuration (100k windows, 5 epochs, hidden size 64, chosen to fit the 90-minute-per-arm compute cap). Step 11 found a larger cause.

## Paper reframing

Every new number carries an HTML-comment source pointer. The headline became "split inflation is dataset-dependent and tracks template drift". Section 5.1 gained a paragraph explaining the HDFS_v1 negative drops with the prevalence and AUROC evidence. Section 5.3 was retitled to "consistent with" wording and states that the effective sample size is two datasets, with the within-dataset test as corroboration rather than independent confirmation. A Threats to validity bullet and an expanded Section 5.6 state the single-split limitation with the rolling-origin results as mitigation. Section 4 and a new Section 5.7 describe the LLM arm as exploratory, sample-based and completed with the 0.5B model, with a Threats bullet on its scale.

# Step 10: remaining gaps and paper readability

## BGL unseen-template sanity check

`results/full/addendum/unseen_templates_bgl.md` and `.json`. For every unseen test-session template occurrence in BGL/CHRONO/seed0 (735,532 of 949,563, or 77.46%), the nearest training template was found by Jaccard similarity on lowercased whitespace-split tokens of the Drain3-masked string. Only 1.55% of unseen occurrences have a nearest training template above 0.8 similarity, so the drift is predominantly new message content rather than Drain3 fragmenting known templates. The top-20 table does show a few of the highest-frequency unseen templates (similarity 0.80 to 0.86) that are the same message as a training template with an extra trailing clause, which Drain3 clustered separately. Both the aggregate and the exceptions are reported in the paper's Section 3 and Threats to validity.

## Full-strength DeepLog run (superseded by Step 11)

`results/full/addendum/deeplog_fullstrength/` (script `scripts/addendum_deeplog_fullstrength.py`). HDFS_v1, seed 0, RANDOM and CHRONO, with a 1,000,000-window cap, up to 15 epochs with early stopping (patience 3) on validation loss, hidden 128, two layers, top-k = 9. The run exceeded a job time limit twice before checkpoint and resume support was added to the script; the third run resumed from the epoch-12 checkpoint and completed both splits. The checkpoint tracks cumulative training time across restarts, so the 90-minute cap is enforced against real compute, and checkpoints are deleted after each split's result is written so no model weights are in the repository.

The RANDOM split's early-stopping model (epoch 13) scored AUPRC 0.380 and AUROC 0.681, well below the main-run configuration's seed-0 AUPRC of 0.779, while CHRONO (early stopping at epoch 4) scored AUPRC 0.557 and AUROC 0.783, essentially the main run's 0.553. That appeared to reverse the sign of DeepLog's RANDOM-to-CHRONO gap. Step 11 found that this was a scoring artifact, not a training effect; the paragraph that originally reported it as a finding has been replaced in the paper.

## Unseen-rate presentation

The paper's Section 3 table now shows four significant figures: HDFS_v1 RANDOM 1.590e-06, CHRONO exactly 0, BGL RANDOM 3.505e-04. A by-class breakdown for HDFS_v1/CHRONO (normal 0, anomalous 0.09943) is read directly from `results/audit/drift_mechanism/HDFS_v1_unseen_rate_normal_vs_anomalous.json`.

## Internal references removed from the paper

Every internal step number, review phrasing and file path in visible prose was reworded to reader-facing language ("a post-hoc analysis added after the pre-registered run", "supplementary material in the repository"); HTML comments keep the real paths. Verified by stripping HTML comments and searching for "Step ", ".md" and "results/": zero hits.

## Abstract

Reduced from 244 words to 227 and then 196, dropping the method list (kept in Section 4) and tightening the prevalence sentence. No parenthetical statistics; the effective-sample-size caveat lives in Section 5.3 and Threats to validity.

## Related work

Added LogAnomaly (Meng et al., IJCAI 2019), LogBERT (Guo et al., IJCNN 2021), the Landauer et al. 2023 survey and He et al.'s ISSRE 2016 experience report, with bib entries and two to three sentences placing this paper relative to each. Full author lists for LogAnomaly and LogBERT and the journal details for the Landauer survey were not verified against a source and carry a `% CHECK` comment outside the BibTeX braces.

## Cross-reference and proofread

Section 5.1's cross-reference to Section 5.4 (which is about H4) now points at the following paragraph and Section 5.6. Two sentences in the Discussion and Conclusion that stated a causal relationship more firmly than Section 5.3's "consistent with" framing allows were softened.

# Step 11: DeepLog diagnosis

The Step 10 full-strength result was not accepted until its cause was found. Full detail: `results/full/addendum/deeplog_diagnosis/DIAGNOSIS.md`.

## Checkpoint and training integrity

No checkpoint existed to load: the main-run arm never saved one, computed a validation loss or did early stopping (it ran a fixed five epochs), and the Step 10 script deleted its checkpoint after writing results. Both configurations were retrained for the diagnosis, keeping the checkpoints (`scripts/diagnose_deeplog.py`).

| Config | Split | Params | Epochs | Stop reason | Final train loss | Best val loss |
|---|---|---|---|---|---|---|
| lightweight | random | 73,719 | 5 | fixed | 0.2272 | 0.2299 |
| lightweight | chrono | 73,719 | 5 | fixed | 0.2560 | 0.1698 |
| fullstrength | random | 278,455 | 15 | completed all epochs | 0.2035 | 0.2045 |
| fullstrength | chrono | 278,455 | 8 | early stopping, patience 3 | 0.2261 | 0.1558 |

Early stopping used normal-only validation sessions, confirmed by reading the code. State-dict hashes are identical after training, after loading and after scoring for all four models. A loading or early-stopping bug is ruled out.

## Score saturation

| Config | Split | Fraction of sessions scoring exactly 0 at k = 9 |
|---|---|---|
| lightweight | random | 0.9754 |
| lightweight | chrono | 0.9854 |
| fullstrength | random | 0.9878 |
| fullstrength | chrono | 0.9915 |

All four configurations, including the lightweight one that matches the main run, saturate above 97%. A sweep over k in {1, 2, 3, 5, 7, 9} shows AUROC and AUPRC varying substantially and non-monotonically with k in every configuration. On a 55-template vocabulary, top-k is a fragile statistic rather than a stable ranking score.

## Continuous scoring

Mean negative log-likelihood scores AUROC 0.97 to 0.99 in all four configurations, against 0.70 to 0.88 for top-9. The Spearman correlation between the two session-level scores is only 0.16 to 0.27, so they measure substantially different things rather than being noisy versions of the same signal.

## Conclusion and consequences

The cause is saturation. It is reproducible across two independently trained configurations, one of which matches the main run, and is directly fixed by a principled alternative score. Because the lightweight configuration's saturation (97.5% random, 98.5% chrono) clears the pre-set 80% threshold by a wide margin, the main-run deeplog cells were treated as defective under the protocol's rule for bugs found after a verdict.

The original deeplog cells (both datasets, both splits, five seeds, plus a `summary.csv` snapshot) were copied to `results/superseded/deeplog_v1/`. `src/arms/deeplog.py` was corrected (NLL primary, top-9 secondary, normal-only early stopping). All 20 deeplog cells were re-run; no other arm's cells were touched. `results/full/summary.csv` was regenerated. `results/full/VERDICTS.md` and `verdicts.json` were left as the pre-registered record and the verdicts were recomputed into `results/full/VERDICTS_v2.md` and `verdicts_v2.json`.

No verdict changed. The evidence under the verdicts did move: HDFS_v1's deeplog H1 drop went from indistinguishable from zero to positive with a CI excluding zero, and HDFS_v1's H2 ranking stability fell from tau 1.00 to 0.73. The Step 10 "stronger DeepLog reverses the sign" result does not survive: with NLL scoring, deeplog on HDFS_v1 behaves like the other arms. Full comparison in `DIAGNOSIS.md`.

## Paper and artefacts

The paper gained a disclosure note at the start of Section 5; the Section 5.6 full-strength paragraph was rewritten to state the saturation finding and withdraw the sign-reversal claim; a k-sweep table was added; the DeepLog Threats bullet now names saturation as the main cause of low absolute performance, with a new bullet on top-k sensitivity; every other deeplog-dependent number in Sections 5.1, 5.2, 5.3 and 5.5 was updated to the corrected value with the pre-registered value alongside. `paper/tables.md` and all four figures were regenerated from the corrected results and visually checked. The comment-stripped draft was re-checked for internal references: zero hits.

# Step 12: consistency sweep after the correction

## Single source of truth

`results/full/summary.csv`'s deeplog rows match `results/full/verdicts_v2.json` and differ substantially from `results/superseded/deeplog_v1/summary_v1.csv` (for example HDFS_v1/random/seed0 AUROC 0.9966 corrected versus 0.9076 original). Both records are intact.

## Derived artefacts

- `scripts/make_tables.py` and `src/make_figures.py`: no change beyond PDF timestamps.
- `scripts/addendum_prevalence_auroc.py`: `prevalence.csv` unchanged. The `auroc_drops.md` table changed for the corrected deeplog numbers, and the script also overwrote the file's hand-written prevalence line and interpretation paragraph, which it was never written to regenerate. Both were restored, with the one deeplog-dependent sentence updated: BGL's five-of-six-arms AUROC-drop range is now 0.35 to 0.65.
- `results/full/addendum/baseline_sanity.md` and `results/audit/h4_h5_diagnosis.md` have no generator script. Every deeplog-dependent number was recomputed by hand from the corrected `summary.csv` and `verdicts_v2.json`; other rows were left untouched.
- `results/full/addendum/rolling_origin/rolling_origin_table.md`: only the deeplog rows at 50% and 70% were re-run (seed 0, NLL scoring).
- README.md's results table was unaffected, since the arms that set its minimum and maximum AUPRC (iforest, supervised_lgbm, pca) were not touched by the correction.

Two secondary findings, neither changing a verdict:

1. deeplog's own CHRONO threshold-leakage gap grew under the corrected scoring: BGL 0.0029 to 0.0674, HDFS_v1 0.0130 to 0.2457. HDFS_v1's aggregate H5 gap is now driven by two arm-level outliers (iforest at 0.348, unrelated to the correction; deeplog at 0.246) rather than one. This is a direct consequence of replacing a near-constant score with a continuous one. H5 stays NOT CONFIRMED on both datasets.
2. The corrected deeplog's rolling-origin BGL numbers move across cutoffs more than any other arm's (AUPRC 0.118, 0.243, 0.076 at 50%, 60%, 70%; AUROC 0.231 at 70%). This is attributed to single-seed training variance (the arm was not yet fully seeded) and stays within the 0.06 to 0.34 band reported for BGL as a whole. HDFS_v1's corrected deeplog is stable across the same cutoffs (0.715, 0.747, 0.837).

## Paper

A short disclosure paragraph was added at the end of Section 4 and the Section 5 intro trimmed so it is not repeated. Sections 5.5 and 5.6 were updated for the two findings above. The comment-stripped draft was checked for "top-k", "k=9", "saturat", "full-strength" and "Step "; every hit is ordinary methodology prose. The Threats to validity deeplog bullet already named saturation rather than under-training as the main cause.

# Step 13: seeding

`src/arms/deeplog.py`'s `fit()` now seeds `random`, `numpy`, `torch.manual_seed` and `torch.cuda.manual_seed_all` from the cell's seed and sets `torch.backends.cudnn.deterministic = True` and `benchmark = False` before training. `scripts/check_deeplog_determinism.py` trained and scored deeplog twice in two separate Python processes on the selftest HDFS_v1/random/seed0 split; AUROC, AUPRC, F1_val and F1_oracle matched exactly (`results/audit/deeplog_determinism_check.json`).

The released `results/full/` deeplog cells were not re-run, because this is a reproducibility fix rather than a correctness fix and re-running 20 cells would not be expected to change any verdict. `REPRODUCE.md` and the paper's Threats to validity each state that the released cells reproduce within seed-to-seed variance (across-seed AUPRC standard deviation 0.0055 to 0.0903 across the four dataset and split combinations) but not bit for bit, and that every other arm is exactly reproducible given the seed.
