# DeepLog diagnosis (Step 11)

The Step 10 full-strength DeepLog result (RANDOM AUPRC falling from ~0.78 to 0.38 with a stronger model, the RANDOM-vs-CHRONO gap reversing sign) was not accepted as a finding until the causes below were ruled in or out, per the pre-registered protocol's rule on bug fixes. This document gives the verdict, with the numbers that decide it.

## Checkpoint and training integrity

No checkpoint from Step 10 (or the main run) existed to load -- the main-run `deeplog` arm never saved one, computed a validation loss, or did early stopping (fixed 5 epochs, unconditionally; confirmed by direct code inspection of `src/arms/deeplog.py` before any change was made), and Step 10's full-strength script deleted its checkpoint after writing each split's final result. Both configs were therefore retrained for this diagnosis (`scripts/diagnose_deeplog.py`), seed 0, HDFS_v1, both splits, keeping the checkpoints this time.

| Config | Split | Params | Hidden | Layers | Epochs completed | Stop reason | Final train loss | Best val loss |
|---|---|---|---|---|---|---|---|---|
| lightweight | random | 73,719 | 64 | 2 | 5 | completed_all_epochs (fixed, no early stop) | 0.2272 | 0.2299 |
| lightweight | chrono | 73,719 | 64 | 2 | 5 | completed_all_epochs (fixed, no early stop) | 0.2560 | 0.1698 |
| fullstrength | random | 278,455 | 128 | 2 | 15 | completed_all_epochs | 0.2035 | 0.2045 |
| fullstrength | chrono | 278,455 | 128 | 2 | 8 | early_stopping_at_epoch_8_patience_3 | 0.2261 | 0.1558 |

Full per-epoch validation-loss trajectories: `HDFS_v1_*_*_meta.json` in this directory.

**Validation set used for early stopping**: confirmed by direct code inspection of `scripts/addendum_deeplog_fullstrength.py` (Step 10) and reproduced identically in `scripts/diagnose_deeplog.py` -- `val_normal = val[val["label"] == 0]`, i.e. **normal validation sessions only**, matching the training objective. No bug here; this hypothesis under conclusion (B) is ruled out.

**State-dict hash check** (`hash_checks.json`): for all 4 models, `hash_after_training == hash_after_loading == hash_after_scoring`. The checkpoint that was trained is the checkpoint that was loaded and the checkpoint that was scored; scoring does not mutate the model. This rules out a loading/scoring-integrity bug.

**Conclusion so far: no evidence of a loading or early-stopping bug (conclusion B ruled out).**

## Score saturation check

`score_saturation.json`, `k_sweep.json` / `k_sweep_table.md` (full tables in this directory).

| Config | Split | Fraction scoring exactly 0 (k=9) | Mean score, normal sessions | Mean score, anomalous sessions |
|---|---|---|---|---|
| lightweight | random | **0.9754** | 0.0001 | 0.0791 |
| lightweight | chrono | **0.9854** | 0.0004 | 0.1106 |
| fullstrength | random | **0.9878** | 0.0000 | 0.0597 |
| fullstrength | chrono | **0.9915** | 0.0000 | 0.1085 |

All four configurations -- including **lightweight, which is architecturally and procedurally identical to what the main run used** -- have 97.5-99.2% of test sessions scoring *exactly zero* at k=9. The k-sweep (k in {1,2,3,5,7,9}) shows AUROC/AUPRC varying substantially and non-monotonically with k for every config/split (e.g. fullstrength/random: AUROC 0.854 at k=1 falling to 0.704 at k=9), which is itself a symptom of a scoring heuristic that is not a stable, well-behaved ranking statistic on a 55-template vocabulary -- with so few templates, "is the true next template in the top-k of 55?" saturates quickly as k grows, collapsing most sessions' scores to 0 regardless of label.

**Conclusion: severe saturation, confirmed in both the lightweight and fullstrength configs.**

## Continuous scoring variant (NLL)

`nll_vs_topk_table.md` (full table in this directory):

| Config | Split | NLL AUROC | NLL AUPRC | Top9 AUROC | Top9 AUPRC | Spearman(NLL, top9) |
|---|---|---|---|---|---|---|
| lightweight | random | **0.9910** | 0.7600 | 0.8823 | 0.7443 | 0.2655 |
| lightweight | chrono | **0.9906** | 0.7110 | 0.8047 | 0.5631 | 0.2082 |
| fullstrength | random | **0.9658** | 0.8266 | 0.7035 | 0.4233 | 0.1903 |
| fullstrength | chrono | **0.9927** | 0.8125 | 0.7767 | 0.5558 | 0.1594 |

Mean per-position NLL (continuous, no top-k saturation) scores **AUROC 0.97-0.99 across all four configs** -- a large, consistent improvement over top-9's 0.70-0.88, and the gap is largest exactly where saturation is worst (fullstrength/random: 0.97 vs 0.70). The weak Spearman correlation between the two session-level scores (0.16-0.27) confirms they are measuring substantially different things, not just noisy versions of the same signal: top-9's miss/no-miss binary-per-position signal throws away almost all the information NLL retains once nearly every session has only 0-2 misses out of many positions.

**Conclusion: NLL restores sensible, consistently strong ranking in every configuration tested.**

## Verdict

**(A) Saturation.** The strong model's (and, critically, the lightweight/main-run-equivalent model's) top-9 score collapses toward zero for the overwhelming majority of sessions (97.5-99.2%, above), and the NLL score restores sensible, strong ranking in every case tested (AUROC 0.97-0.99, above). The checkpoint-integrity check above ruled out (B): no loading bug, no early-stopping-on-wrong-validation-set bug, no checkpoint-integrity issue. (C) is ruled out because the mechanism is identified, reproducible across two independently-trained configurations, and directly fixed by a principled alternative scoring function.

## Consequences for the frozen results

**Trigger fires.** The lightweight config's saturation rate (97.54% random, 98.54% chrono) clears the 80%-of-test-sessions-scoring-exactly-0 threshold by a wide margin, on a configuration that is architecturally and procedurally equivalent to the main run's `deeplog` arm (same hidden size 64, 2 layers, 100k-window cap, same seed; retrained only because the original checkpoint was never persisted). This is not a close call sensitive to the training non-determinism noted in `DECISIONS.md` -- both splits clear 80% by roughly 18-20 percentage points. **The main-run `deeplog` cells (both datasets) have a defect**, per the protocol's rule on bug fixes discovered after a verdict is written.

**Remediation carried out:**
1. `results/full/HDFS_v1/{random,chrono}/deeplog/` and `results/full/BGL/{random,chrono}/deeplog/` (all 5 seeds, plus a snapshot of the pre-correction `summary.csv`) copied to `results/superseded/deeplog_v1/` before anything was changed.
2. `src/arms/deeplog.py` corrected: primary score is now mean NLL; top-9 miss-rate retained as a secondary score (`secondary_topk9_*` fields in every deeplog cell's JSON); early stopping added on normal-only validation loss (patience 3, up to 15 epochs), replacing the original unconditional fixed-5-epoch schedule.
3. All 20 `deeplog` cells (2 datasets x 2 splits x 5 seeds) re-run with the corrected arm; every other arm's cells were never touched.
4. `results/full/summary.csv` regenerated (now reflects corrected `deeplog` + unchanged everything else); `results/full/VERDICTS.md` / `verdicts.json` were **not** regenerated and remain the original v1 record; verdicts recomputed from the new summary into `results/full/VERDICTS_v2.md` / `verdicts_v2.json`.

**Which verdicts changed**: **none of the five top-level verdicts changed.**

| Hypothesis | v1 verdict | v2 verdict |
|---|---|---|
| H1 | CONFIRMED | CONFIRMED |
| H2 | CONFIRMED | CONFIRMED |
| H3 | CONFIRMED | CONFIRMED |
| H4 | NOT CONFIRMED | NOT CONFIRMED |
| H5 | NOT CONFIRMED | NOT CONFIRMED |

The underlying evidence did shift in several places worth flagging for a reviewer, even though no top-level verdict moved:

- **H1**: HDFS_v1's `deeplog` RANDOM-vs-CHRONO AUPRC drop flips from indistinguishable-from-zero in v1 (mean 0.014, CI [-0.125, 0.152], not meeting the bar) to positive and significant in v2 (mean 0.074, CI [0.046, 0.106], **now meeting the 0.05 bar**). HDFS_v1's arm-count-meeting-the-bar rises from 0/6 (v1) to 1/6 (v2) -- still nowhere near the 4/6 needed for HDFS_v1 to independently confirm H1, so the overall verdict (driven by BGL's 5/6) is unaffected. BGL's `deeplog` drop itself shrinks somewhat (0.875 to 0.782) but remains large and clears the bar either way.
- **H2**: HDFS_v1's RANDOM-vs-CHRONO rank correlation falls from perfectly stable in v1 (Kendall tau = 1.00) to moderately unstable in v2 (tau = 0.73, now also below the 0.8 threshold) -- the verdict was already CONFIRMED via BGL alone in both versions, but v2 shows both datasets exhibiting some ranking instability, not just one.
- **H3**: pooled Spearman correlation is essentially unchanged (0.828 to 0.803, both p < 1e-13).
- **H4**: the specific `semantic_knn`-vs-`deeplog` sub-comparison on HDFS_v1 flips from inconclusive in v1 (mean diff 0.0016, CI straddling zero) to a CI-confirmed finding in v2 (mean diff 0.062, CI [0.033, 0.093] -- `semantic_knn` genuinely degrades less than the corrected `deeplog` on HDFS_v1). The overall H4 verdict stays NOT CONFIRMED because the `semantic_knn`-vs-`pca` sub-comparison (unaffected by this correction) still fails on both datasets, and H4 requires both comparisons on both datasets.
- **H5**: HDFS_v1's mean threshold-leakage gap under CHRONO rises from 0.061 (v1) to 0.099 (v2) -- now within a hair of the pre-registered 0.10 bar (point estimate still just short, so the verdict stays NOT CONFIRMED), compared to v1 being comfortably below it.
- **Most importantly for the paper's narrative**: Step 10's "a stronger DeepLog reverses the sign of the RANDOM-vs-CHRONO gap" finding does not survive the correction. With NLL-primary scoring, `deeplog`'s HDFS_v1 behavior is unremarkable and consistent with the same-direction pattern most other arms show (RANDOM outperforms CHRONO, by a modest, CI-confirmed margin) -- the "reversal" reported in Step 10 was an artifact of the top-k metric saturating differently at different training intensities, not a real property of model capacity interacting with chronological splitting. This is corrected in `paper/draft.md` (see Section 5.6).

Per the protocol's rule on documenting bug fixes, the paper reports both the pre-registered v1 verdicts and the corrected v2, with the defect stated plainly.
