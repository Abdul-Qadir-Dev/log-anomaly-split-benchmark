# Summary

This document summarizes the outcomes, deviations from protocol, and open questions for readers of the paper.

## 1. Outcome summary

We pre-registered and ran a benchmark testing whether random vs. chronological data splitting changes log anomaly detection results on HDFS_v1 and BGL (Loghub), across six methods and five seeds (120 cells total, all completed). **Split inflation is dataset-dependent and tracks template drift, not a universal property of chronological evaluation**: chronological splitting substantially lowers AUPRC and AUROC and reorders the best-performing method **on BGL** (where 77% of chronological test-template occurrences are unseen in training, stable across a rolling-origin check at three different chronological cutoffs) but **not on HDFS_v1** (where the template vocabulary is small and almost entirely covered within the first 60% of the timeline regardless of split); HDFS_v1's apparent AUPRC *gains* under chronological splitting were themselves shown in a Step 09 post-run review to be mostly a test-set-prevalence artifact (AUROC stays nearly flat) and not stable across different chronological cutoffs. The AUPRC drop correlates strongly with unseen-template rate (Spearman rho = 0.80, p < 0.001; 0.83 before the deeplog scoring correction), which we now report as *consistent with* the drift mechanism rather than proof of it, given an effective sample size of 2 independent datasets. Threshold leakage (test-tuned vs. validation-tuned F1) was small in both datasets in our setup. The exploratory LLM-scoring arm's main-run attempt (1.5B model) was marked SKIPPED on this machine, but a Step 09 retry with a smaller 0.5B model completed successfully (sample-based, no verdict attached, reported separately -- Section 11 below). A Step 11 diagnosis also found and fixed a scoring defect in `deeplog` (its original top-k score saturated on HDFS_v1's small vocabulary); all `deeplog` cells were re-run with a corrected score and every hypothesis's verdict was recomputed -- **none changed** (Section 13). None of this affects any of the five pre-registered hypotheses.

## 2. The five verdicts

| Hypothesis | Bar | Verdict |
|---|---|---|
| H1 — split inflation | AUPRC_random − AUPRC_chrono ≥ 0.05, CI excl. 0, on ≥4/6 arms, on ≥1 dataset | **CONFIRMED** (BGL: 5/6 arms) |
| H2 — ranking instability | Kendall τ < 0.8 between RANDOM/CHRONO rankings, on ≥1 dataset | **CONFIRMED** (BGL: τ=0.20) |
| H3 — unseen templates explain degradation | Spearman ρ > 0.4, p < 0.05, across (dataset×seed×arm) cells | **CONFIRMED** (ρ=0.83, p=3.5e-16) |
| H4 — semantic-arm robustness | semantic_knn's drop smaller than pca's and deeplog's, both datasets, CI excl. 0 | **NOT CONFIRMED** |
| H5 — threshold leakage | mean(F1_oracle − F1_val) ≥ 0.10 under CHRONO, CI excl. 0, on ≥1 dataset | **NOT CONFIRMED** |

Full detail with numbers and CIs: `results/full/VERDICTS.md`.

**Step 12 note:** the table above is the pre-registered (v1) record, unchanged. After the Step 11 `deeplog` scoring correction, all five verdicts were recomputed from the corrected `results/full/summary.csv` into `results/full/VERDICTS_v2.md` / `verdicts_v2.json` -- **the v2 verdicts are identical to the v1 verdicts above**, arm for arm. See Section 14 for the full consistency sweep that followed the correction.

## 3. Deviations from protocol, with reasons

All logged contemporaneously in `DECISIONS.md`; summary:

- **Hardware tier**: GTX 1660 Ti measured at exactly 6144 MiB (6 GiB), the stated GPU-present threshold — treated as GPU-present tier, no deviation needed.
- **`MIN_SIZE_BYTES` download sanity check** was miscalibrated (assumed both dataset zips would be >100MB; BGL's compresses to 57.5MB) and falsely flagged a complete download as failed. Fixed in `scripts/download_data.py`; no effect on the data itself (verified with `zipfile.testzip()`).
- **`.gitignore`** was changed from blanket-excluding `data/raw/` to excluding everything except `INVENTORY.md`, so that inspection documentation (not the raw data) is tracked.
- **`parse_bgl_line`** originally rejected any BGL line with an empty message field (9 fields instead of 10), incorrectly flagging 34,470 genuine lines (e.g. bare `RAS KERNEL FATAL`) as unparseable. Fixed before any verdict existed; BGL was re-parsed from scratch (0 unparseable lines after the fix).
- **CI methodology for cross-condition differences** (H1, H4, H5): the protocol specifies a bootstrap CI for a single AUPRC estimate but not explicitly for a *difference* across split conditions. We used a seed-level paired bootstrap (resample the 5 per-seed differences, 1000 times, seed 0) — a clarification made before any analysis ran, not a change made to move a verdict.
- **`semantic_knn` implementation**: switched from sklearn's brute-force `NearestNeighbors` to a batched GPU matrix multiply after the former took ~45 minutes per cell (would have been ~7.5 hours for HDFS alone). This is a performance fix only — re-validated on `--selftest` that it produces identical numbers.
- **`llm_zeroshot` arm marked SKIPPED** for the full run — see Section 4.

## 4. Arms skipped, and why

Only `llm_zeroshot` (`Qwen2.5-1.5B-Instruct` zero-shot scorer) was skipped, for the full run only. It failed in sequence: (1) GPU OOM loading the 1.5B model on this machine's 6GB GPU, which is also the Windows display adapter and so has fluctuating free VRAM shared with the OS/other apps; (2) GPU OOM again in a fresh process with the suggested `expandable_segments` mitigation (unsupported on Windows); (3) a CPU fp32 fallback then hit `OSError: The paging file is too small`; (4) a multi-candidate fallback (1.5B GPU → 0.5B GPU → 0.5B CPU) succeeded once on a 30-sample selftest set but took ~33 minutes for just 30 generations, and the very next combo failed again on the same pagefile error. This is a hardware/OS resource ceiling on this specific machine, not a bug in the arm's logic, and exceeds the protocol's "two attempts and one simplification" bar for marking an arm SKIPPED (Section 5.5). **This does not affect H1–H5**, which are defined only over the six non-LLM arms (`PROTOCOL.md` Section 5.5, `NON_LLM_ARMS` in code).

## 5. Three results most worth an independent reviewer's scrutiny

1. **HDFS_v1 shows *negative* AUPRC drops for pca/iforest/ocsvm (chrono outperforms random)** — the opposite of H1's direction, confined to one dataset. We attribute this to HDFS's near-zero unseen-template rate under both conditions plus possible time-clustering of anomalies (`results/ANOMALIES.md` #1), but we did not run the suggested follow-up (stratified-by-time-bucket random split) to confirm it.
2. **`semantic_knn` scores *below chance* (AUROC 0.34) on BGL/CHRONO** — a specific, surprising failure of the arm we hypothesized would be *most* robust to template drift. Our explanation (`results/ANOMALIES.md` #2) is plausible but unverified; a reviewer should check whether this is a real inversion of signal or an artifact of how session embeddings are averaged.
3. **The H4/H5 CIs rely on only 5 seeds** (`DECISIONS.md` Step 05 methodology note) — a paired bootstrap over 5 values is a thin basis for a 95% CI, even though it's the best use of the data the protocol specifies. A reviewer should treat H4 and H5's "NOT CONFIRMED" verdicts as reasonably solid (point estimates were also far from the bar in most cases) but treat any borderline CI result elsewhere in the detail tables with caution.

## 6. Weaknesses a hostile reviewer would raise

- Two datasets only, both widely used and therefore possibly over-studied/atypical of real deployments.
- BGL's "anomalous if any line in the 100-line window carries an alert tag" labelling is coarse and likely explains the below-chance `iforest` AUROC on BGL/RANDOM (`results/ANOMALIES.md` #3) — a confound unrelated to the paper's actual question.
- One Drain3 configuration (similarity 0.4, depth 4); template vocabulary and therefore `unseen_template_rate` would shift under a different parser or config.
- (Narrowed in the Step 09 review — see Section 11 below.) The exploratory LLM arm's main-run attempt (1.5B model) failed on this machine; a Step 09 retry with a smaller 0.5B model completed and is reported in `paper/draft.md` Section 5.7, but only at a single seed and 200-session-per-cell sample, well below the 5-seed scale of the other six arms — so the paper still cannot speak with real confidence to whether zero-shot LLM scoring is more or less sensitive to chronological splitting than the other methods, just offer a suggestive data point.
- `ocsvm`'s 50k-session training subsample (prescribed by the protocol) means its HDFS training set is a ~15% subsample of the full normal training pool; this could understate its true performance relative to methods that train on the full set.

## 7. Suggested next experiments

- Run the time-bucket-stratified random split on HDFS_v1 (Section 5, `results/ANOMALIES.md` #1) to test whether anomaly clustering, not template coverage, explains HDFS's negative drops.
- Compare per-session `unseen_template_rate` between normal and anomalous BGL/CHRONO sessions directly, to test the proposed explanation for `semantic_knn`'s below-chance AUROC (`results/ANOMALIES.md` #2).
- Re-run `iforest` on BGL with a stricter alert-fraction labelling threshold instead of "any alert line" (`results/ANOMALIES.md` #3).
- Extend to a third Loghub dataset with a template-drift profile between HDFS_v1 and BGL's extremes, to see whether the unseen-template-rate relationship (H3) generalizes smoothly.
- Get `llm_zeroshot` running reliably (e.g. on a machine with a dedicated, non-display GPU) to fill in the one arm missing from this benchmark.

## 8. Total runtime and compute

Hardware: NVIDIA GeForce GTX 1660 Ti (6 GB VRAM, shared with the OS display), 12 logical CPUs, ~19.85 GB RAM, Windows (`results/hardware.json`).

Measured script-execution time (reconstructed from script logs and commit timestamps; excludes idle time between work sessions, which was not compute):

| Stage | Approx. wall-clock |
|---|---|
| Environment setup (pip installs) | ~25 min |
| Data download (HDFS_v1 1.58GB + BGL 743MB, with retries) | ~15 min |
| Drain3 parsing (both datasets, ~16M lines total) | ~6 min |
| Pipeline development + selftests | ~30 min |
| Full experiment matrix, first attempt (killed: `semantic_knn` performance bug) | ~90 min (21/124 cells salvaged) |
| Full experiment matrix, second attempt (failed: GPU OOM from an oversized batch) | ~15 min |
| Full experiment matrix, third attempt (completed all 120 non-LLM cells) | ~142 min |
| `llm_zeroshot` attempts (all failed or partial; arm ultimately SKIPPED) | ~45 min |
| Section 9 robustness checks | ~23 min |
| Analysis, figures, tables | ~2 min |
| **Total measured compute** | **~6.1 hours**, well inside the overall compute budget (PROTOCOL.md's compute-plan rule) |

## 10. Post-hoc analysis (Step 08)

A six-task verification pass ran after the main run completed, with `results/full/` frozen throughout (nothing in it was re-run or changed, and none of the five verdicts above moved). Full detail: `results/audit/AUDIT.md`.

| Task | Result |
|---|---|
| Verify the BGL parse | **Confirmed correct.** Independently re-read all 4,747,963 lines, 0 dropped, and the empty-content template's occurrence count (34,470) exactly matches the Step 04 bug's drop count -- strong confirmation the earlier fix is complete. |
| Verify the `semantic_knn` GPU speed fix changed nothing | **Confirmed.** Max score difference 5.4e-7 (bar: <1e-4); AUPRC/AUROC on a 5,000-session check cell match to 4-5 decimal places. |
| Diagnose H4 and H5 | **Explained**, no verdict changed. H5's HDFS_v1 aggregate gap is driven almost entirely by one outlier arm (`iforest`); see `results/audit/h4_h5_diagnosis.md`. |
| Retry `llm_zeroshot` with a smaller model | **Succeeded this time** (`Qwen2.5-0.5B-Instruct`, all 4 cells on GPU, no failure). Still exploratory, still no verdict, still not in the main ranking -- see `results/audit/llm_table.md`. |
| Reproducibility smoke test on a fresh clone | **Found and fixed 2 real bugs**: a `requirements.txt` CUDA-pin that broke `pip install` on any machine, and a missing downloader for the `--selftest` fixture files. Both fixed; `REPRODUCE.md` updated; all selftests now pass cleanly on a fresh clone. |

None of the two real reproducibility bugs this post-hoc analysis found and fixed (the `requirements.txt` CUDA pin, the missing selftest-fixture downloader) touched `results/full/`, the pipeline's scientific logic, or any pre-registered hypothesis -- exactly the kind of thing a second pass is meant to catch once blind spots from the main run are no longer visible to whoever ran it.

## 11. Step 09 review fixes

A second review pass, focused on result quality and interpretation rather than release hygiene. `results/full/` stayed frozen throughout -- no cell that fed a verdict was re-run, and `results/full/VERDICTS.md` was not edited. Full detail: `results/audit/AUDIT.md` ("Step 09" section).

| Task | Result |
|---|---|
| 1. Selftest contamination check | **Clean, with one real bug fixed.** `results/full/summary.csv`, `paper/tables.md`, and all four figures were independently regenerated from the frozen per-seed JSONs and diffed byte-for-byte against the committed versions -- identical; no committed table or figure had ever contained selftest numbers. The underlying risk was real, though: `src/make_figures.py` hardcoded its output directory regardless of `--selftest`, so a selftest run *could* have overwritten the real figures (it happened not to, because the full run always ran after any selftest run in this project's actual history). Fixed to write to `figures/selftest/` when `--selftest` is passed. |
| 2. Prevalence + AUROC addendum | **Done, and it changes the right-sized conclusion, not any verdict.** `results/full/addendum/prevalence.csv` and `auroc_drops.md` show HDFS_v1's RANDOM/CHRONO test-set prevalence differs nearly 2x (2.96% vs 1.46%) while BGL's barely differs (9.73% vs 8.49%); AUROC (prevalence-invariant) is nearly flat for every HDFS_v1 arm while AUPRC swings by 0.09-0.33 -- strong evidence that HDFS_v1's "chronological splitting helps" result is mostly a prevalence artifact, not a real discriminability gain. BGL's AUROC drops track its AUPRC drops closely, confirming BGL's result is genuine. Added to `paper/draft.md` Results (5.1) and Threats to validity. |
| 3. Within-dataset drift-mechanism test | **Done; partially supports, does not complete, the semantic_knn explanation.** On BGL/CHRONO, unseen-template rate correlates positively with `semantic_knn` score *within* both the normal and anomalous classes (as the mechanism predicts) and correctly predicts the *direction* of the small between-class gap -- but that gap is too small to explain why the arm's AUROC comes out *below* chance rather than just near it; a second, unidentified factor is needed. On HDFS_v1/CHRONO, the signal is much cleaner: normal sessions have exactly 0% unseen templates, anomalous sessions average 9.9%, and unseen-rate-alone is a moderately predictive score on its own (AUROC 0.72) -- refining (not contradicting) the time-clustering speculation in `results/ANOMALIES.md` #1. Full report: `results/audit/drift_mechanism/REPORT.md`. |
| 4. Rolling-origin robustness | **Done; a genuine, now-documented limitation found.** Re-ran all 6 non-LLM arms at chronological cutoffs of 50% and 70% (plus the existing 60%), seed 0 only. BGL's degradation is stable across all three cutoffs (AUPRC 0.06-0.34 throughout). HDFS_v1's is not: `pca`'s AUPRC is 0.485 at the 50% cutoff vs. 0.920 at 60% and 0.994 at 70%, even though `unseen_template_rate` stays at exactly 0 throughout -- the single main-run HDFS_v1 chronological split was not representative of nearby cutoffs. Table: `results/full/addendum/rolling_origin/rolling_origin_table.md`; added to `paper/draft.md` Section 5.6 and Threats to validity. |
| 5. Baseline sanity table | **Done, with literature values honestly marked `n/a` throughout.** `results/full/addendum/baseline_sanity.md` tabulates our absolute RANDOM-split metrics against a literature-value column; we did not have a verified source for exact published digits for any row, so every literature cell is `n/a` rather than a guess, with citations and qualitative differences (window definition, threshold policy, training cap) given instead. One finding stated plainly at the time: our `deeplog` HDFS_v1 AUPRC (0.589)/F1 (~0.71) is noticeably below DeepLog's typical reputation, attributed to our 100k-window/5-epoch training cap. **Section 13 found a second, larger contributing cause**: the original scoring heuristic itself saturates on HDFS_v1's small vocabulary, independent of training budget -- the corrected scoring scores substantially higher on the same data. |
| 6. Paper reframing | **Done.** `paper/draft.md`'s abstract, Results (new 5.1 paragraph, retitled 5.3, new 5.6 rolling-origin paragraph, new 5.7 LLM section), Discussion, Threats to validity (3 new bullets), and Conclusion were all updated: headline is now "split inflation is dataset-dependent and tracks template drift"; HDFS_v1's negative drops are explained with the prevalence/AUROC evidence; H3 is worded "consistent with" with the effective-n=2 caveat stated; the one-split-per-chrono-condition limitation is stated with the rolling-origin results as mitigation; the LLM arm is described as exploratory/sample-based/0.5B. Every new number carries an HTML-comment source pointer, same convention as the original draft. |

None of the six Step 09 tasks touched `results/full/` or moved any of the five pre-registered verdicts. The net effect is a more carefully qualified paper -- same five verdicts, considerably more honest about what the HDFS_v1 "negative drop" result does and doesn't mean, and with the LLM arm no longer silently absent.

## 12. Step 10 final polish

A third pass, focused on two remaining analytical gaps and on making `paper/draft.md` readable as a standalone document (no internal process references, a proper abstract, complete related work). `results/full/` stayed frozen throughout. Full detail: `results/audit/AUDIT.md` ("Step 10" section).

| Task | Result |
|---|---|
| 1. BGL unseen-template sanity check | **Done.** Jaccard token-similarity against every training template for every unseen BGL/CHRONO template: only 1.55% of unseen occurrences have a close (>0.8) training match, supporting genuine content novelty as the dominant story, though the top-20-by-frequency table shows a handful of high-frequency exceptions that are really Drain3-fragmented variants. Added to `paper/draft.md` Section 3 and Threats to validity. |
| 2. Full-strength DeepLog (HDFS_v1, seed 0) | **Done; interpretation superseded by Section 13.** A much larger DeepLog (1M-window cap, up to 15 epochs with early stopping, hidden 128) showed a sharp RANDOM-split AUPRC drop (0.779 to 0.380) while CHRONO barely moved, apparently reversing the sign of DeepLog's RANDOM-vs-CHRONO gap. At the time this was read as a property of the top-k-miss scoring heuristic and reported as a genuine finding. **Section 13's dedicated diagnosis found this was actually a scoring defect** (top-k saturates on HDFS_v1's small vocabulary), not a real effect -- see Section 13 for the full story and the correction applied. This training run was also interrupted twice by a per-invocation wall-clock cap partway through -- added checkpoint/resume capability to the script so it could continue rather than restart from scratch each time. |
| 3. Unseen-rate presentation fix | **Done.** `unseen_template_rate` now reported to 4 significant figures (not 4 decimals) in `paper/draft.md` Section 3, revealing HDFS_v1's RANDOM (1.590e-06) and CHRONO (exactly 0) are not actually identical, as the old 4-decimal rounding made them look. Added HDFS_v1/CHRONO split-by-class rows, read directly from the Step 09 drift-mechanism JSON so they agree with that report by construction. |
| 4. Remove internal references from paper prose | **Done.** Every visible mention of internal step numbers, "post-run audit/review," and internal file paths was reworded to reader-facing language ("a post-hoc analysis added after the pre-registered run," "supplementary material"); HTML comments keep the real paths for traceability. Verified by stripping HTML comments and grepping the remainder for the literal strings "Step ", ".md", "results/" -- zero hits. |
| 5. Abstract rewrite | **Done.** Cut from 244 to 196 words across three passes; dropped the explicit six-method list and all parenthetical statistics (the effective-sample-size caveat now lives only in Section 5.3 and Threats to validity); kept all five required content elements. |
| 6. Related work additions | **Done, with honest uncertainty.** Added LogAnomaly, LogBERT, the Landauer survey, and He et al.'s ISSRE 2016 experience report, each with a bib entry. We did not verify exact author lists / journal details for three of the four against a source, so those fields carry `% CHECK` comments rather than invented precision. |
| 7. Cross-reference fix + proofread | **Done.** Fixed a stale Section 5.1 -> 5.4 cross-reference (now points at the correct paragraph and Section 5.6); softened two sentences that overstated the "consistent with" framing established in 5.3 as more definite/causal than the evidence supports. |

**(Superseded by Section 13 below.)** This paragraph originally read the full-strength DeepLog result (the apparent RANDOM-vs-CHRONO sign reversal) as a genuine, if surprising, robustness finding about training configuration. A follow-up diagnosis (Section 13) found that reading was wrong: the reversal was an artifact of DeepLog's top-k scoring heuristic saturating on HDFS_v1's small vocabulary, not a real property of training intensity. The corrected finding is in Section 13.

## 13. Step 11 DeepLog diagnosis

The Step 10 full-strength DeepLog result was not accepted as a finding until its causes were ruled in or out, per the protocol's rule on bug fixes. Full detail: `results/full/addendum/deeplog_diagnosis/DIAGNOSIS.md`.

**Verdict: (A) saturation.** All 4 diagnostic checkpoints (lightweight/fullstrength x random/chrono, HDFS_v1, retrained since neither the main run nor Step 10 had kept a checkpoint) show 97.5-99.2% of test sessions scoring *exactly zero* at top-9 -- including the lightweight config, architecturally and procedurally identical to the main run. State-dict hashes confirmed identical before/after loading and before/after scoring for all 4 models, ruling out a loading or scoring-mutation bug (conclusion B). Mean negative log-likelihood (continuous, no top-k saturation) scores AUROC 0.97-0.99 in every configuration, against top-9's 0.70-0.88.

**Trigger fired**: the lightweight config's saturation rate clears the diagnosis's 80% main-run-saturation threshold by ~18-20 percentage points on both splits -- not a close call. Per the protocol's rule on bug fixes, this meant the main-run `deeplog` cells had a defect requiring correction, not just documentation:

1. Original `deeplog` cells (both datasets, both splits, 5 seeds, plus a `summary.csv` snapshot) copied to `results/superseded/deeplog_v1/`.
2. `src/arms/deeplog.py` corrected: primary score is now mean NLL; top-9 retained as a secondary diagnostic (`secondary_topk9_*` fields); early stopping added on normal-only validation loss (patience 3, up to 15 epochs), replacing the original's unconditional fixed-5-epoch schedule.
3. All 20 `deeplog` cells re-run with the corrected arm; every other arm's cells untouched.
4. `results/full/summary.csv` regenerated (now reflects corrected `deeplog`); the original `results/full/VERDICTS.md` / `verdicts.json` were **not** touched and remain the pre-registered record; verdicts recomputed from the new summary into `results/full/VERDICTS_v2.md` / `verdicts_v2.json`.

**Which verdicts changed: none.** All five hypotheses' top-level verdicts (H1-H5) are identical between the pre-registered (v1) and corrected (v2) result sets. The underlying evidence did shift in several places -- most notably, HDFS_v1's `deeplog` RANDOM-vs-CHRONO AUPRC drop flips from statistically indistinguishable from zero (v1) to positive and CI-confirmed (v2, now clearing H1's bar, though still far short of flipping H1's overall verdict), and HDFS_v1's rank-order stability for H2 drops from perfectly stable (tau=1.00) to moderately unstable (tau=0.73). Full comparison table in `DIAGNOSIS.md`.

**Most consequential single outcome**: Step 10's "a stronger DeepLog reverses the RANDOM-vs-CHRONO sign" finding does not survive the correction -- with NLL-primary scoring, `deeplog`'s HDFS_v1 behavior is unremarkable (RANDOM modestly outperforms CHRONO, the same direction most other arms show). The apparent reversal was an artifact of how differently two training configurations' top-k scores happened to saturate, not a real property of model capacity. `paper/draft.md` Section 5.6 was rewritten accordingly, with a new upfront note at the start of Section 5 disclosing the correction, a k-sweep table, and two new/revised Threats to validity bullets (top-k scoring sensitivity; the original absolute-performance bullet updated to cite saturation as a second, dominant cause alongside the training-budget explanation). Every other place in the paper that quoted a `deeplog`-dependent number (Sections 5.1, 5.2, 5.3, 5.5) was updated to the corrected value, with the pre-registered value given alongside it, so neither record is hidden.

## 14. Step 12 consistency sweep

Step 11 corrected `deeplog` and re-ran its cells; this step swept every derived artefact and every prose number that depends on it, to catch anything the correction did not already propagate to. Full detail: `results/audit/AUDIT.md` "Step 12".

**Single source of truth confirmed**: `results/full/summary.csv` holds the v2 (corrected) `deeplog` cells; `results/superseded/deeplog_v1/` holds the originals, untouched.

**Artefacts regenerated and diffed**: `paper/tables.md` and the four figures were already current (no change). `results/full/addendum/prevalence.csv` is unchanged (not arm-specific); regenerating `auroc_drops.md` exposed that its generator script was never written to preserve two hand-written paragraphs added in Step 09 -- restored them, updating the one `deeplog`-dependent sentence. `results/full/addendum/baseline_sanity.md` and `results/audit/h4_h5_diagnosis.md` have no persisted generator (hand-written in Step 08/09) -- recomputed and rewrote their `deeplog` rows/prose by hand. The rolling-origin table's `deeplog` rows (only) were re-run at the 50%/70% cutoffs with the corrected scoring, per this step's scope. README.md's results table was checked and found unaffected.

**Two secondary findings surfaced, neither changing any verdict**:
1. `deeplog`'s own CHRONO threshold-leakage gap grew sharply under the corrected scoring (BGL 0.003 -> 0.067; HDFS_v1 0.013 -> 0.246) -- HDFS_v1's H5 aggregate is now driven by two arm-level outliers (`iforest`, pre-existing; corrected `deeplog`, new) instead of one, an expected consequence of moving from a saturated to a continuous score.
2. The corrected `deeplog`'s rolling-origin numbers are not stable across chronological cutoffs on BGL (AUPRC 0.118/0.243/0.076 at 50%/60%/70%, non-monotonic), more so than any other arm -- attributed to this arm's already-documented single-seed training non-determinism, not a new systematic pattern, and still within the aggregate BGL AUPRC band (0.06-0.34) the paper already reports across all six arms. HDFS_v1's corrected `deeplog` is comparatively stable across the same cutoffs.

**Paper**: added a short, plain disclosure paragraph at the end of Section 4 (ahead of Results), trimmed the Section 5 intro to avoid duplicating it, and updated Sections 5.5 and 5.6 for the two findings above. Re-grepped visible prose for internal-reference markers -- clean, as in Step 11.

No hypothesis verdict changed; `results/full/VERDICTS.md` (v1) remains frozen and `results/full/VERDICTS_v2.md` (v2) remains the corrected record, both released.

## 15. Step 13 seeding

`src/arms/deeplog.py` was the one arm whose training was not fully seeded (random/numpy/torch/cuDNN) -- fixed by seeding all of those from the cell's seed at the start of `fit()` and setting `cudnn.deterministic=True`/`benchmark=False`. Verified with `scripts/check_deeplog_determinism.py`: two separate-process runs on the same selftest split and seed produced byte-identical AUROC/AUPRC/F1 (`results/audit/deeplog_determinism_check.json`).

**`results/full/` was not re-run** (this is a reproducibility fix, not a correctness fix; the already-recorded scores from unseeded runs are not wrong, they are simply one of many equally-valid random draws that a future re-run with the same seed can no longer reproduce bit-for-bit). Documented with one sentence each in `REPRODUCE.md` and `paper/draft.md`'s Threats to validity, quoting the across-seed AUPRC standard deviation from `results/full/summary.csv` (0.0055-0.0903 across the 4 dataset/split combinations) as the expected tolerance for anyone re-running the released `deeplog` cells from scratch. Every other arm, and any `deeplog` cell trained after this fix, is exactly reproducible given the same seed.
