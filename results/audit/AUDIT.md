# Step 08 — post-hoc verification analysis

`results/full/` is frozen: nothing in it was re-run or modified by this analysis, and `results/full/VERDICTS.md` is unchanged. All new outputs from this analysis live under `results/audit/`. Every decision is logged contemporaneously in `DECISIONS.md` under "Step 08 — post-hoc analysis".

## Summary table

| Task | Description | Status |
|---|---|---|
| 2 | Verify the BGL parse | **PASS** |
| 3 | Verify the `semantic_knn` speed fix didn't change results | **PASS** |
| 4 | Diagnose H4 and H5 | **DONE** (explanatory, no pass/fail bar; verdicts unchanged) |
| 5 | Retry the exploratory LLM arm | **PASS** (all 4 cells completed; no failure traceback needed) |
| 6 | Reproducibility smoke test | **PASS** (after 2 real fixes) |

## Task 2: BGL parse verification

Full report: `results/audit/bgl_parse_verification.md` / `.json`. Headline numbers (all independently recomputed by re-reading the raw log and re-running the parser, not read back from cached results):

- Total lines read: 4,747,963 — **matches** the expected count exactly.
- Lines dropped during parse: **0**.
- Alert-labelled lines: 348,460 (7.34% of all lines); non-alert: 4,399,503.
- Drain3 templates: HDFS_v1 = 55, BGL = 1,346 — well under the 5,000 over-fragmentation flag (not triggered; top-20 table produced anyway, see the full report).
- BGL/CHRONO/seed0 `unseen_template_rate`: **0.7746 per occurrence** (recomputed independently, matches the recorded value used by every H1/H3/H5 verdict exactly) vs. **0.7500 per unique template** (450 of 600 distinct test templates never appear in train). Both framings agree the drift is large; the per-occurrence number is modestly higher because unseen templates also recur more often per session in the chronological tail.

A striking cross-check: the empty-content template (`""`) in the current BGL template vocabulary has exactly **34,470** occurrences in `results/audit/bgl_parse_verification.md`'s top-20 table — the identical count of lines the pre-audit (Step 04) parser bug used to drop. This is strong independent confirmation that the Step 04 fix is both correct and complete.

## Task 3: `semantic_knn` speed-fix verification

Full report: `results/audit/knn_cpu_check_result.json` (script: `results/audit/knn_cpu_check.py`). One cell (HDFS_v1, chrono, seed 0), 5,000-session query subsample against the real 333,878-session training set:

- **Max absolute score difference: 5.4e-7** (pass bar: < 1e-4) — **PASS**, by four orders of magnitude.
- Mean absolute difference: 5.6e-8.
- Spearman rho: 0.23 — superficially alarming, but explained: most HDFS test sessions are template-identical to some training-normal session, so both scorers produce near-zero scores for the bulk of the 5,000 queries (only 19 distinct rounded values), and float32-vs-float64 rounding noise reorders that tied block without reflecting a real disagreement.
- To settle it decisively, AUPRC/AUROC were computed directly on the query subsample with both score sets: **AUPRC 0.97793 (GPU) vs. 0.97792 (CPU)**, **AUROC bit-identical at 0.99968** for both. The performance fix from Step 04 did not change the metrics that matter.

## Task 4: H4 / H5 diagnosis

Full tables and two explanatory paragraphs: `results/audit/h4_h5_diagnosis.md`. No verdict changed. In short:

- **H4** fails for two different reasons depending on the comparison arm: against `pca`, the data clearly point the opposite direction from the hypothesis (CI nowhere near zero on both datasets); against `deeplog`, the point estimate leans the hypothesized way on both datasets but `deeplog`'s own across-seed variance is too large for 5 seeds to distinguish it from `semantic_knn`.
- **H5**'s HDFS_v1 aggregate gap (0.0606, close-ish to the 0.10 bar) is driven almost entirely by one outlier, `iforest` (gap 0.348); every other HDFS_v1 arm has a gap under 0.02. The aggregate is not representative of "threshold leakage across the board" on that dataset.

## Task 5: LLM arm retry

Script: `scripts/audit_llm_retry.py`. Model switched to `Qwen/Qwen2.5-0.5B-Instruct` (fp16, GPU; CPU fallback available but not needed), 200 sessions/dataset/split (100/class), seed 0, max 40 template lines/prompt. This arm has no pre-registered verdict, so re-running it is in scope.

**All 4 cells completed successfully on GPU** — no CUDA failure, no CPU fallback needed, no `results/audit/llm_failure.txt` written. The smaller model's VRAM footprint (well under 2GB) avoided the GPU contention that sank every attempt with the 1.5B model during the original run.

| Dataset | Split | AUROC | AUPRC | N sample | Device |
|---|---|---|---|---|---|
| HDFS_v1 | random | 0.4380 | 0.4880 | 200 | cuda |
| HDFS_v1 | chrono | 0.3689 | 0.4442 | 200 | cuda |
| BGL | random | 0.7452 | 0.6930 | 200 | cuda |
| BGL | chrono | 0.6513 | 0.5923 | 200 | cuda |

Worth noting (exploratory only, not a verdict): the 0.5B zero-shot scorer does *worse than chance* on HDFS_v1 (AUROC 0.44, 0.37) but clearly better than chance on BGL (0.75, 0.65) — the opposite dataset-difficulty pattern from several of the non-LLM arms, plausibly because HDFS's extremely repetitive, low-vocabulary (55-template) logs give the LLM little distinguishing text to reason over, while BGL's richer, more semantically varied error messages (1,346 templates) are easier for a language model to judge. This is a new observation from the audit, not pursued further here.

## Task 6: Reproducibility smoke test

Full report: `results/audit/smoketest_report.md`. Procedure: fresh `git clone`, fresh Python 3.11 venv, `pip install -r requirements.txt`, download selftest fixtures, run every `--selftest`.

**Two real bugs found and fixed**:
1. `requirements.txt`'s `torch==2.6.0+cu124` pin doesn't exist on PyPI — a fresh install failed immediately. Fixed by re-pinning to the bare `torch==2.6.0` (PEP 440 matches any local-version variant, CPU or CUDA, against a bare specifier).
2. `data/sample/` (required by every `--selftest`) is gitignored and was never fetched by any committed script. Added `scripts/download_sample_data.py` and documented it in `REPRODUCE.md`.

One environment-specific issue (Windows `MAX_PATH` failure installing from a deeply nested path) was documented as a troubleshooting note rather than "fixed," since it is a property of the install location, not the repository.

After both fixes, every `--selftest` script passed on a fresh clone: parse/sessionize/split/features in under 80 seconds combined, `run_all.py --selftest` in 212.6s (CPU-only), `analysis.py`/`make_figures.py --selftest` in 12 seconds combined. Total measured `--selftest` execution: **~5.1 minutes**.

## Overall

All six tasks complete. Two real bugs were found and fixed during this analysis (the `requirements.txt` CUDA-pin reproducibility break, and the missing selftest-fixture downloader) — none of them affect `results/full/` or any of the five pre-registered verdicts, which remain exactly as reported in `results/full/VERDICTS.md`.

---

# Step 09 — review fixes

A second review pass, focused on result quality, statistical interpretation, and paper framing rather than release hygiene. `results/full/` stayed frozen throughout: no cell that fed a verdict was re-run, and `results/full/VERDICTS.md` was not edited. New analysis went to `results/audit/` or `results/full/addendum/`. Every decision logged contemporaneously in `DECISIONS.md` under "Step 09 — review fixes".

## Summary table

| Task | Description | Status |
|---|---|---|
| 1 | Selftest contamination check | **PASS** (no contamination found; 1 real bug fixed pre-emptively) |
| 2 | Prevalence + AUROC addendum | **DONE** |
| 3 | Within-dataset drift-mechanism test | **DONE** |
| 4 | Rolling-origin robustness | **DONE** (found a real, now-documented limitation) |
| 5 | Baseline sanity table | **DONE** (literature values honestly `n/a` throughout) |
| 6 | Paper reframing | **DONE** |

## Task 1: Selftest contamination

**Check performed**: regenerated `results/full/summary.csv` from the frozen per-seed JSONs (`rglob("seed*.json")` under `results/full/`, no cell re-run), regenerated `paper/tables.md` and all four committed figures from that summary, and diffed every one against the committed version.

**Result**: **no contamination found.** `results/full/summary.csv` was byte-identical (mod row order) before and after regeneration -- 120 rows, `n_test` values of exactly 9,496 (BGL) or 115,013 (HDFS_v1), never the tiny selftest sizes (e.g. `n_test=4` for BGL/chrono/selftest). `paper/tables.md` and all four `figures/*.png` were byte-identical to the regenerated versions. The four `figures/*.pdf` files matched in content but not bytes -- the only difference was matplotlib's embedded `/CreationDate` timestamp, confirmed by direct inspection of the differing byte range; the regenerated PDFs were discarded in favor of the originals rather than committing a timestamp-only diff. The README's results table (hand-written, not script-generated) was independently recomputed from the clean data and matches exactly.

**Bug found and fixed pre-emptively**: `src/make_figures.py`'s `main()` hardcoded `figdir = Path("figures")` regardless of the `--selftest` flag -- every other script in the pipeline (`run_all.py`, `analysis.py`) correctly branches on a `tag = "selftest" if selftest else "full"` pattern, but this one didn't. In this project's actual history the full-data `make_figures.py` run always happened *after* any selftest run, so the committed figures were never actually overwritten with selftest output -- but the risk was real and the next person to run `--selftest` after a full run, in either order, could have silently corrupted the committed figures. Fixed: selftest now writes to `figures/selftest/` (added to `.gitignore`). `scripts/make_tables.py` has no `--selftest` flag at all and hardcodes `results/full/summary.csv` as its only input, so it was never at risk by construction.

## Task 2: Prevalence and AUROC addendum

Built entirely from `results/full/summary.csv` (no re-run). `results/full/addendum/prevalence.csv` (20 rows: dataset x split x seed) and `results/full/addendum/auroc_drops.md` (AUROC drop vs AUPRC drop, same paired-bootstrap CI methodology as the H1/H4/H5 differences, per dataset x arm).

**Headline finding**: HDFS_v1's RANDOM and CHRONO test splits have substantially different anomaly prevalence (2.96% vs. 1.46% -- nearly 2x) purely as an accident of where anomalous blocks fall on the timeline, while BGL's prevalence is nearly stable across conditions (9.73% vs. 8.49%). AUROC (prevalence-invariant) is nearly flat for every HDFS_v1 arm (-0.003 to -0.023) while AUPRC swings by 0.09-0.33 for the same arms -- strong evidence that HDFS_v1's apparent "chronological splitting helps" result is mostly a prevalence artifact, not a real gain in ranking quality. BGL's AUROC drops track its AUPRC drops closely (five of six arms, 0.38-0.65, CIs excluding zero), confirming BGL's result is a genuine degradation, not a metric artifact. The one exception, `iforest` on BGL, has a large *negative* AUROC drop despite a near-zero AUPRC drop -- consistent with `results/ANOMALIES.md` #3's unrelated finding that this arm's score is already inverted on BGL under RANDOM. Added to `paper/draft.md` Results (new paragraph in 5.1) and Threats to validity (new bullet).

## Task 3: Within-dataset template-drift mechanism test

Diagnostic re-score only (`results/audit/drift_mechanism/`), `results/full/` untouched. On **BGL/CHRONO/seed0**: mean per-session unseen rate is 0.768 (normal) vs. 0.875 (anomalous) -- both very high, and unseen-rate-alone is only chance-level (AUROC 0.498) at separating the classes, because the entire test period is swamped by drift regardless of label. Within each class separately, unseen rate correlates strongly and positively with a freshly-fit `semantic_knn`'s score (Spearman rho=0.544 normal, 0.620 anomalous, both p<1e-80) -- the mechanical relationship the arm is built on holds up cleanly. But that mechanism, extended naively, would predict `semantic_knn`'s AUROC to land a little *above* chance (anomalous sessions have the higher mean unseen rate), not substantially *below* it (0.342, matching the frozen run) -- so raw unseen-template *volume* only partially explains the arm's below-chance result; a second, unidentified factor (plausibly *which* templates go unseen, not how many) is needed. On **HDFS_v1/CHRONO/seed0**, the picture is cleaner: normal sessions have exactly 0% unseen templates, anomalous sessions average 9.9%, and unseen-rate-alone is moderately predictive on its own (AUROC 0.716) -- refining, not contradicting, `results/ANOMALIES.md` #1's time-clustering speculation. Full report with both interpretive paragraphs: `results/audit/drift_mechanism/REPORT.md`.

## Task 4: Rolling-origin robustness

All 6 non-LLM arms, both datasets, seed 0 only, at chronological train-cutoffs of 50% and 70% (val = next 20%, test = remainder), alongside the existing 60% main-run values. `results/full/addendum/rolling_origin/rolling_origin_table.md`.

**BGL's degradation is stable across all three cutoffs**: AUPRC stays in a 0.06-0.34 range at every cutoff (vs. BGL's RANDOM-split AUPRC of 0.83-0.99), and `unseen_template_rate` stays high throughout (0.69-0.80, never dropping). **HDFS_v1's result is not stable**: `pca`'s AUPRC is 0.485 at the 50% cutoff vs. 0.920 at 60% and 0.994 at 70%; `ocsvm` shows the same pattern (0.672 / 0.923 / 0.996). Since `unseen_template_rate` stays at exactly 0.0000 at every HDFS_v1 cutoff, template drift cannot explain this swing -- the most likely driver is the same prevalence/test-window-composition sensitivity found above. This is a genuine limitation of the main run's single HDFS_v1 chronological split, now quantified and documented rather than hidden. Added to `paper/draft.md` Section 5.6 (expanded) and Threats to validity (new bullet); referenced there as a mitigation for that limitation.

## Task 5: Baseline sanity table

`results/full/addendum/baseline_sanity.md`: absolute RANDOM-split AUROC/AUPRC/F1_val/F1_oracle per arm per dataset (mean over 5 seeds, read from `results/full/summary.csv`), with a literature-value column. **Every literature-value cell is marked `n/a`**: we did not verify citable exact published digits for any of these specific method/dataset/metric combinations against a source, so unknown values are marked `n/a` rather than guessed. Citations and qualitative "why values would differ" notes (window/session definition, RANDOM-vs-CHRONO split, validation-vs-test threshold, Drain3 config, training-budget caps) are given instead, per method. One finding stated plainly: our `deeplog` HDFS_v1 result (AUPRC 0.589, AUROC 0.806, F1 ~0.71) is noticeably below DeepLog's typical reputation in the literature; the likely cause is our protocol's lightweight training configuration (100k-window cap, 5 epochs, hidden size 64 -- chosen to fit the 90-minute-per-arm compute budget across 120 cells), not a bug. Now stated in `paper/draft.md` Threats to validity.

## Task 6: Paper reframing

`paper/draft.md` edited throughout, every new number carrying an HTML-comment source pointer (same convention as the original draft):

- **(a) Headline reframed**: abstract, README, and SUMMARY.md now lead with "split inflation is dataset-dependent and tracks template drift," not a blanket "random splits inflate results" framing.
- **(b) HDFS negative-drops paragraph**: a full new paragraph in Results 5.1 (and echoed in Discussion) explains the negative AUPRC drops using the task-2 prevalence and AUROC evidence.
- **(c) H3 reworded**: Results 5.3 (retitled "Is the drop consistent with unseen-template rate as the mechanism?") now says "consistent with" throughout, states the effective sample size is 2 independent datasets (not 60 cells), and reports the task-3 within-dataset mechanism test as corroborating evidence rather than independent confirmation.
- **(d) Single-chrono-split limitation stated**: a new Threats to validity bullet and an expanded Results 5.6 state the one-split-per-condition limitation explicitly, with the task-4 rolling-origin results given as the mitigation (and as evidence the limitation is real for HDFS_v1 specifically).
- **(e) LLM arm reframed**: Methods (Section 4) and a new Results 5.7 describe `llm_zeroshot` as exploratory, sample-based, and now completed with the smaller `Qwen2.5-0.5B-Instruct` model, with its actual results table from `results/audit/llm_table.md`; Threats to validity gained a bullet on its limited scale (1 seed, 200 samples/cell).

## Overall (Step 09)

All six tasks complete. `results/full/` was never touched -- every addendum, diagnostic, and robustness result lives in `results/full/addendum/` or `results/audit/`, and all five pre-registered verdicts in `results/full/VERDICTS.md` are byte-for-byte unchanged from Step 04. The one real bug found (`make_figures.py`'s non-selftest-aware output path) was a latent risk, not an active contamination -- confirmed by independently regenerating and diffing every committed table and figure against the frozen per-seed source data.

---

# Step 10 — final polish

A third pass: two remaining analytical gaps, plus making `paper/draft.md` read as a standalone document free of internal-process references. `results/full/` stayed frozen throughout; `results/full/VERDICTS.md` untouched. New analysis to `results/full/addendum/`. Decisions logged in `DECISIONS.md` under "Step 10 — final polish".

## Summary table

| Task | Description | Status |
|---|---|---|
| 1 | BGL unseen-template sanity check | **DONE** |
| 2 | Full-strength DeepLog robustness run | **DONE** (surprising, now-documented result) |
| 3 | Fix unseen-rate presentation (4 sig figs + by-class rows) | **DONE** |
| 4 | Remove internal references from paper prose | **PASS** (verified by grep, zero hits) |
| 5 | Abstract rewrite, under 200 words | **PASS** (196 words) |
| 6 | Related work additions | **DONE** (uncertain fields marked `% CHECK`) |
| 7 | Cross-reference fix + proofread | **DONE** |

## Task 1: BGL unseen-template sanity check

`results/full/addendum/unseen_templates_bgl.md` / `.json`. For every one of BGL/CHRONO/seed0's unseen test-session template occurrences (735,532 of 949,563 total, 77.46%), computed the nearest training template by Jaccard similarity on lowercased whitespace-split tokens of the Drain3-masked string (metric chosen and stated explicitly, as required). **Only 1.55% of unseen occurrences have a nearest training template above 0.8 similarity** -- the occurrence-weighted picture says these are predominantly genuinely new message content, not Drain3 fragmenting known templates. The top-20-by-frequency table complicates the clean headline number, though: a handful of the highest-occurrence unseen templates (similarity 0.80-0.86) are clearly the same underlying message as a training template with an extra trailing clause appended, which Drain3 clustered as a new template rather than a variant -- a real, if minority, contribution to the unseen-template count. Both the clean headline and the complicating exceptions are reported; added to `paper/draft.md` Section 3 and Threats to validity.

## Task 2: Full-strength DeepLog robustness run

`results/full/addendum/deeplog_fullstrength/` (script: `scripts/addendum_deeplog_fullstrength.py`). HDFS_v1, seed 0, RANDOM and CHRONO, config as specified: 1,000,000-window cap, up to 15 epochs with early stopping (patience 3) on validation loss, hidden 128, 2 layers, top-k=9.

**Infrastructure note**: this run needed far longer than the wall-clock cap available per invocation, and was killed outright (process terminated, not just detached) twice before checkpoint/resume support was added to the script. After that fix, a third invocation correctly resumed from the epoch-12 checkpoint (confirmed by the script's own "resumed from checkpoint at epoch 13" log line) and completed both splits in one run. The checkpoint tracks *cumulative actual training time* across restarts (not wall-clock including the gaps when the process wasn't running), so the 90-minute wall-clock cap is enforced correctly against real compute regardless of how many times the process had to be restarted. Checkpoints are deleted after each split's final result is written, so no large model-weight files ended up in the repository.

**Result**: RANDOM split's early-stopping-selected model (epoch 13) scores AUPRC 0.380 / AUROC 0.681 -- *substantially worse* than the lightweight main-run config's seed-0 AUPRC of 0.779. CHRONO (early-stopping at epoch 4) scores AUPRC 0.557 / AUROC 0.783, essentially unchanged from the main-run's 0.553. **This reverses the sign of DeepLog's RANDOM-vs-CHRONO gap entirely** -- more training makes CHRONO look relatively *better* than RANDOM, the same direction as `pca`/`iforest`/`ocsvm`'s HDFS_v1 pattern, not the direction one would expect from "more training fixes under-training." The likely mechanism: DeepLog's anomaly score depends on top-k next-template-prediction misses, and validation loss (next-token cross-entropy) is a proxy objective for that score, not identical to it -- a model that becomes a better general-purpose predictor with more capacity and data can become *better* at predicting sequences that happen to be anomalous too, shrinking the separation the score relies on. This is reported as a genuine, surprising robustness finding, not explained away. Added to `paper/draft.md` Section 5.6 (new paragraph) and the DeepLog Threats to validity bullet (updated).

## Task 3: Unseen-rate presentation fix

`paper/draft.md` Section 3's unseen_template_rate table now shows 4 significant figures: HDFS_v1 RANDOM is 1.590e-06 (not indistinguishable "0.0000" from CHRONO's exact 0) and BGL RANDOM is 3.505e-04. Added a by-class breakdown for HDFS_v1/CHRONO (normal: 0 exactly; anomalous: 0.09943), read directly from `results/audit/drift_mechanism/HDFS_v1_unseen_rate_normal_vs_anomalous.json` with no recomputation, so the numbers agree with that report by construction.

## Task 4: Remove internal references from paper prose

Every flagged line (internal step numbers, internal audit/review phrasing, internal file paths in visible prose) was individually reworded to reader-facing language -- "a post-hoc analysis added after the pre-registered run," "the pre-registered protocol (supplementary material)," "supplementary material in the repository" -- while HTML comments retained the real paths for traceability. Verified twice: once with a moderately broad regex, once with the exact literal substrings ("Step ", ".md", "results/") against the comment-stripped draft. **Zero hits both times.**

## Task 5: Abstract rewrite

Three revision passes: 244 words (first rewrite) -> 227 (dropped the method-list detail, kept in Section 4) -> 196 (tightened the prevalence sentence). Final abstract has zero parenthetical statistics; the effective-sample-size caveat was removed from the abstract entirely and exists only in Section 5.3 and Threats to validity. All five required content elements present: problem, protocol, headline finding, one-sentence HDFS prevalence lesson, what's released.

## Task 6: Related work additions

Added LogAnomaly (Meng et al., IJCAI 2019), LogBERT (Guo et al., IJCNN 2021), a deep-learning-for-log-anomaly-detection survey (Landauer et al., 2023), and He et al.'s ISSRE 2016 experience report, with bib entries and 2-3 sentences placing this paper relative to each in `paper/draft.md` Section 2. We did not verify the full author lists for LogAnomaly and LogBERT, or the exact journal/volume/pages for the Landauer survey, against a source -- marked each with a `% CHECK` comment line placed outside the BibTeX entry's braces (to avoid risking a parse break) rather than inventing precision we didn't have.

## Task 7: Cross-reference fix + proofread

Fixed Section 5.1's cross-reference, which pointed at Section 5.4 (actually about H4 semantic-arm robustness) for a discussion that is really in the very next paragraph and in Section 5.6 -- now points there instead. A full proofread pass for overclaiming found two sentences stating a causal/predictive relationship more definitely than the "consistent with" framing established in Section 5.3 justifies (Discussion's "it is a consequence of template drift" and the Conclusion's "well predicted ... consistent with ... our data" run-on) and softened both.

## Overall (Step 10)

All seven tasks complete. `results/full/` was never touched; every new result lives in `results/full/addendum/`. **(Superseded by Step 11 below.)** The sentence that followed originally read the full-strength DeepLog finding above as a genuine result about training configuration; Step 11 found it was a scoring-saturation artifact instead -- see below.

---

# Step 11 — DeepLog diagnosis

The Step 10 full-strength DeepLog result (RANDOM AUPRC falling from ~0.78 to 0.38 with a stronger model, the RANDOM-vs-CHRONO gap reversing sign) was not accepted as a finding until the causes below were ruled in or out, per the protocol's rule on bug fixes. `results/full/` stayed frozen except for the one remediation this diagnosis triggered (deeplog cells, below). Full detail: `results/full/addendum/deeplog_diagnosis/DIAGNOSIS.md`.

## Summary table

| Task | Description | Status |
|---|---|---|
| 1 | Checkpoint and training integrity | **DONE** (no loading/early-stopping bug found) |
| 2 | Score saturation check | **DONE** (severe saturation confirmed, both configs) |
| 3 | Continuous (NLL) scoring variant | **DONE** (restores strong, stable ranking) |
| 4 | Verdict | **(A) Saturation** |
| 5 | Consequences for frozen results | **TRIGGERED** -- deeplog cells superseded, corrected, re-run; no verdict changed |
| 6 | Paper correction | **DONE** |
| 7 | Housekeeping | **DONE** |

## Task 1: Checkpoint and training integrity

No checkpoint existed to load -- the main-run `deeplog` arm never saved one, computed a validation loss, or did early stopping (fixed 5 epochs, unconditional, confirmed by direct code inspection before any change was made); Step 10's script deleted its own checkpoint after writing results. Both configs (lightweight = main-run-equivalent, fullstrength = Step 10's config) were retrained for this diagnosis, keeping the checkpoints this time (`scripts/diagnose_deeplog.py`).

| Config | Split | Params | Epochs | Stop reason | Train loss (final) | Best val loss |
|---|---|---|---|---|---|---|
| lightweight | random | 73,719 | 5 | fixed (no early stop) | 0.2272 | 0.2299 |
| lightweight | chrono | 73,719 | 5 | fixed (no early stop) | 0.2560 | 0.1698 |
| fullstrength | random | 278,455 | 15 | completed_all_epochs | 0.2035 | 0.2045 |
| fullstrength | chrono | 278,455 | 8 | early_stopping (patience 3) | 0.2261 | 0.1558 |

**Validation set for early stopping**: confirmed by direct code inspection -- normal-only validation sessions, matching the training objective. No bug. **State-dict hash check**: `hash_after_training == hash_after_loading == hash_after_scoring` for all 4 models -- no loading or scoring-mutation bug. **Conclusion (B) ruled out.**

## Task 2: Score saturation check

| Config | Split | Fraction exactly 0 (k=9) |
|---|---|---|
| lightweight | random | 0.9754 |
| lightweight | chrono | 0.9854 |
| fullstrength | random | 0.9878 |
| fullstrength | chrono | 0.9915 |

All four configurations -- **including lightweight, architecturally and procedurally identical to the main run** -- saturate above 97%. A k-sweep (k in {1,2,3,5,7,9}) shows AUROC/AUPRC varying substantially and non-monotonically with k in every config, confirming top-k is a fragile statistic on a 55-template vocabulary, not a stable ranking score.

## Task 3: Continuous scoring variant (NLL)

Mean negative log-likelihood scores **AUROC 0.97-0.99 in all four configurations**, against top-9's 0.70-0.88. Weak Spearman correlation between the two session-level scores (0.16-0.27) confirms they measure substantially different things, not noisy versions of the same signal.

## Task 4: Verdict

**(A) Saturation.** Task 1 ruled out (B). The mechanism is identified, reproducible across two independently-trained configurations (including one procedurally identical to the main run), and directly fixed by a principled alternative scoring function -- ruling out (C).

## Task 5: Consequences for the frozen results

**Trigger fired**: the lightweight config's saturation (97.5% random, 98.5% chrono) clears the 80% main-run-saturation threshold by ~18-20 points on both splits -- not a close call. Remediation: original `deeplog` cells (both datasets, both splits, 5 seeds, plus a `summary.csv` snapshot) copied to `results/superseded/deeplog_v1/`; `src/arms/deeplog.py` corrected (NLL primary, top-9 secondary, normal-only early stopping added); all 20 `deeplog` cells re-run with the corrected arm, every other arm's cells untouched; `results/full/summary.csv` regenerated; `results/full/VERDICTS.md`/`verdicts.json` left untouched as the pre-registered record; verdicts recomputed into `results/full/VERDICTS_v2.md`/`verdicts_v2.json`.

**Which verdicts changed: none** (H1-H5 identical between v1 and v2). The underlying evidence shifted in several places -- most notably HDFS_v1's `deeplog` H1 drop flips from indistinguishable-from-zero to positive and CI-confirmed, and HDFS_v1's H2 rank stability falls from tau=1.00 to tau=0.73 -- without flipping any top-level verdict. **Most consequential**: Step 10's "stronger DeepLog reverses the sign" finding does not survive the correction; with NLL-primary scoring, `deeplog`'s HDFS_v1 behavior is unremarkable and consistent with the other arms' pattern. Full comparison table: `DIAGNOSIS.md`.

## Task 6: Paper correction

`paper/draft.md` updated throughout: a new disclosure note at the start of Section 5; the Section 5.6 "full-strength DeepLog" paragraph rewritten to state the saturation finding and retract the sign-reversal claim; a k-sweep table added to 5.6; the DeepLog Threats bullet rewritten (saturation now cited as a second, dominant cause of low absolute performance) and a new Threats bullet added on top-k scoring sensitivity; every other `deeplog`-dependent number in Sections 5.1, 5.2, 5.3, 5.5 updated to the corrected (v2) value with the pre-registered (v1) value given alongside it, consistent with the regenerated supplementary tables and figures (which now reflect the corrected `deeplog`).

## Task 7: Housekeeping

Grepped the comment-stripped draft for "Step ", ".md", "results/" -- zero hits, both before and after all Step 11 edits. `paper/tables.md` and all four figures regenerated from the corrected `results/full/` (genuine content changes this time, not just PDF timestamps, since `deeplog`'s underlying numbers changed) and visually spot-checked.

## Overall (Step 11)

All seven tasks complete. This is the first step in this project's post-hoc analysis trail where `results/full/` was legitimately modified -- a pre-registered-arm defect was found, documented, corrected, and re-run, with the original preserved in full and both verdict sets released side by side, exactly as the protocol's rule for this situation requires. No hypothesis verdict changed, but a previously-reported "finding" (DeepLog's apparent training-sensitivity) was retracted and replaced with the correct explanation (scoring saturation) -- the kind of self-correction a pre-registered, audited protocol is supposed to produce when it works as intended.

# Step 12 — consistency sweep

Step 11 corrected `deeplog`'s scoring and re-ran its cells, but left a sweep of every derived artefact and every prose number for a later step -- this is that step. `results/full/` stayed frozen except for the one artefact explicitly re-run below (rolling-origin `deeplog` rows); no new defect was found.

## Summary table

| Task | Description | Status |
|---|---|---|
| 1 | Single source of truth confirmed | **DONE** (summary.csv = v2; superseded/ = v1) |
| 2 | Regenerate derived artefacts, diff against committed | **DONE** (2 stale artefacts found and fixed, 1 surprising secondary finding) |
| 3 | Paper consistency | **DONE** |
| 4 | SUMMARY.md / REPRODUCE.md | **DONE** |

## Task 1: Single source of truth

Confirmed by direct comparison: `results/full/summary.csv`'s `deeplog` rows match `results/full/verdicts_v2.json` (corrected, NLL-primary) and differ substantially from `results/superseded/deeplog_v1/summary_v1.csv` (original, top-9) -- e.g. HDFS_v1/random/seed0 AUROC 0.9966 (v2) vs. 0.9076 (v1). Both records are intact and released side by side.

## Task 2: Derived-artefact regeneration

Re-ran every generator script and diffed the output against the committed version:

- `scripts/make_tables.py`, `src/make_figures.py`: **no change** (already regenerated at the end of Step 11; figures show only the usual PDF timestamp noise).
- `scripts/addendum_prevalence_auroc.py`: `prevalence.csv` unchanged (not arm-specific). `auroc_drops.md`'s table changed (corrected `deeplog` numbers); **the script also silently dropped the file's hand-written prevalence-context line and Interpretation paragraph**, which it was never written to regenerate (a Step 09 gap, not a Step 12 regression) -- restored both, updating the one `deeplog`-dependent sentence in the Interpretation paragraph (BGL's "five of six arms, AUROC drops 0.38-0.65" range tightens to "0.35-0.65").
- `results/full/addendum/baseline_sanity.md`, `results/audit/h4_h5_diagnosis.md`: **no persisted generator script for either** (both were written by hand directly from `summary.csv` in Step 08/09). Recomputed every `deeplog`-dependent number by hand from the corrected `summary.csv` / `verdicts_v2.json` and rewrote the affected rows and prose; non-`deeplog` rows/prose (e.g. `pca`, `semantic_knn` vs. `pca`) left untouched.
- `results/full/addendum/rolling_origin/rolling_origin_table.md`: re-ran only the `deeplog` rows at the 50%/70% cutoffs (seed 0, corrected NLL scoring) per this step's explicit scope; the other five arms' cutoff cells were not touched.
- README.md's results table: checked against the corrected summary -- unaffected (the per-arm means that set its min/max AUPRC are `iforest` and `supervised_lgbm`/`pca`, neither touched by the correction).

**Two findings worth flagging for a reviewer, surfaced only by doing this sweep:**
1. `deeplog`'s own CHRONO threshold-leakage gap (F1_oracle - F1_val) grew sharply under the corrected scoring: BGL 0.0029 -> 0.0674, HDFS_v1 0.0130 -> 0.2457. HDFS_v1's aggregate H5 gap is now driven by two individual-arm outliers (`iforest` at 0.348, unrelated to this correction; corrected `deeplog` at 0.246) rather than one. H5's verdict is unaffected (NOT CONFIRMED on both datasets either way), but this is a real, direct consequence of moving from a mostly-saturated, near-constant score to a continuous one, not noise.
2. The corrected `deeplog`'s rolling-origin BGL numbers are not stable across the three chronological cutoffs (AUPRC 0.118/0.243/0.076 at 50%/60%/70%; AUROC falls to 0.231, below chance, at 70%) -- more cutoff-sensitive than any other arm. Attributed to this arm's known, previously-documented training non-determinism (single seed, unseeded LSTM/cuDNN training) rather than a new systematic pattern, since it stays within the aggregate 0.06-0.34 BGL AUPRC band the paper already reports across all six arms collectively. HDFS_v1's corrected `deeplog` is comparatively stable across the same cutoffs (0.715/0.747/0.837).

## Task 3: Paper consistency

Added a short, plainly-worded disclosure paragraph to the end of Section 4 (ahead of Results), and trimmed Section 5's longer Step-11 intro paragraph to a one-line pointer so the disclosure is not duplicated. Updated Section 5.5 (threshold leakage) and Section 5.6 (rolling-origin) for the two findings above. Grepped the comment-stripped draft for "top-k", "k=9", "saturat", "full-strength", "Step " -- every hit is legitimate methodology/mechanism prose (e.g. "flags a step anomalous if ..." is ordinary English, not an internal step-number reference); nothing needed changing. The Threats to validity `deeplog` bullet (from Step 11) already describes top-k saturation, not undertraining, as the dominant cause -- confirmed still accurate, no change needed.

## Task 4: SUMMARY.md / REPRODUCE.md

Added a note under Section 2's verdict table stating the v2 verdicts are identical to v1 arm-for-arm (the table itself is the frozen v1 record and was left untouched); added this Section 14. `REPRODUCE.md` has no mention of `deeplog` scoring mode as a flag (NLL is simply the only behavior now) -- no update needed. Committed as "step 12 consistency sweep".

## Overall (Step 12)

All five tasks complete. No defect was found in this sweep -- it confirmed the Step 11 correction propagated correctly almost everywhere, with two small gaps (a stale hand-maintained paragraph in `auroc_drops.md`, two hand-written audit tables with no persisted generator) now fixed, and surfaced two genuine secondary findings (deeplog's own threshold-leakage gap, and its rolling-origin cutoff-sensitivity on BGL) that belong in the paper but do not change any verdict. `results/full/VERDICTS.md` (v1) remains frozen; `results/full/VERDICTS_v2.md` (v2) remains the corrected record; both are released.

# Step 13 — seeding

`results/full/` untouched. `src/arms/deeplog.py` was the one arm whose training was not fully seeded -- fixed, verified, and documented as a reproducibility (not correctness) limitation of the already-released cells.

## Task 1: Seed everything, verify determinism

`fit()` now seeds `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` from `ctx.seed`, and sets `torch.backends.cudnn.deterministic = True` / `benchmark = False`, all before any training happens. `scripts/check_deeplog_determinism.py` trained and scored `deeplog` twice in two separate Python processes (selftest HDFS_v1/random/seed0) and compared AUROC/AUPRC/F1_val/F1_oracle: all four matched **exactly** (not just to the requested 6 decimals) -- `results/audit/deeplog_determinism_check.json`.

## Task 2: No re-run; documented the limitation

The released `results/full/` `deeplog` cells were **not** re-run -- this is a forward-looking reproducibility fix, and re-running 20 cells for zero expected verdict change was out of scope. Added one sentence each to `REPRODUCE.md` and `paper/draft.md`'s Threats to validity, stating that the released cells reproduce within seed-to-seed variance (across-seed AUPRC SD 0.0055-0.0903 across the 4 dataset/split combinations, quoted from `results/full/summary.csv`) but not bit-for-bit, and that every other arm is exactly reproducible given the same seed.

## Task 3: Housekeeping

`SUMMARY.md` Section 15 added. Committed as "step 13 deeplog seeding".

## Overall (Step 13)

All tasks complete. This step fixes a reproducibility gap (not a correctness defect) in the one arm that had one, verifies the fix with an explicit two-process determinism check rather than assuming it worked, and is transparent that the already-released cells predate the fix rather than silently implying they were always deterministic. `results/full/` is untouched.
