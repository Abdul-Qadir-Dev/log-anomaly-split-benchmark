# H4 and H5 diagnosis (Step 08 audit)

Explanation only -- the pre-registered verdicts in `results/full/VERDICTS.md` are unchanged. All numbers below are read from `results/full/summary.csv` and `results/full/verdicts.json` (frozen, not re-run).

**Step 12 update:** `results/full/summary.csv` now holds the Step 11 corrected (mean-NLL) `deeplog` cells; the `deeplog` rows and the `pca`/`deeplog` comparison below are updated to match (pulled from `results/full/verdicts_v2.json`, not the frozen `verdicts.json`). `semantic_knn`/`pca` numbers are untouched. The original (pre-correction) `deeplog` numbers are preserved in `results/superseded/deeplog_v1/`, and the top-level H4/H5 verdicts did not change either way.

## H4: RANDOM->CHRONO AUPRC drop, semantic_knn vs pca vs deeplog

| Dataset | Arm | AUPRC random (mean, 5 seeds) | AUPRC chrono (mean, 5 seeds) | Drop (random - chrono) | Drop SD across seeds |
|---|---|---|---|---|---|
| HDFS_v1 | pca | 0.5936 | 0.9200 | -0.3264 | 0.0036 |
| HDFS_v1 | deeplog | 0.8068 | 0.7325 | 0.0743 | 0.0390 |
| HDFS_v1 | semantic_knn | 0.9979 | 0.9854 | 0.0124 | 0.0002 |
| BGL | pca | 0.8292 | 0.0802 | 0.7491 | 0.0116 |
| BGL | deeplog | 0.9581 | 0.1757 | 0.7824 | 0.0924 |
| BGL | semantic_knn | 0.9382 | 0.0639 | 0.8742 | 0.0072 |

Pre-registered comparison (difference of drops; positive means semantic_knn's drop is smaller / semantic_knn is more robust), from `results/full/verdicts.json` H4:

| Dataset | Comparison | Mean difference | 95% CI | Point favors semantic_knn? | CI excludes 0? |
|---|---|---|---|---|---|
| HDFS_v1 | pca - semantic_knn | -0.3388 | [-0.3420, -0.3357] | No | No |
| HDFS_v1 | deeplog - semantic_knn | +0.0618 | [+0.0332, +0.0934] | Yes | **Yes** |
| BGL | pca - semantic_knn | -0.1252 | [-0.1306, -0.1198] | No | No |
| BGL | deeplog - semantic_knn | -0.0919 | [-0.1610, -0.0228] | No | No |

**Paragraph (Step 12: updated for the corrected `deeplog`; `pca` rows unchanged).** `semantic_knn` is not more robust to chronological splitting than `pca` on either dataset, and the effect is not close: on HDFS_v1, `pca` actually *improves* under CHRONO (-0.326 "drop", i.e. a gain) while `semantic_knn` loses a small 0.012, so `pca`'s drop is 0.34 points smaller -- the opposite of H4's hypothesized direction, with a confidence interval nowhere near zero. On BGL, both arms lose heavily under CHRONO (0.75-0.88 AUPRC), and `pca` still loses noticeably less than `semantic_knn` (0.125 points less). Against the corrected `deeplog`, the picture now differs by dataset: on HDFS_v1, `semantic_knn`'s drop is now clearly, CI-confirmed smaller than `deeplog`'s corrected drop (+0.062, CI [0.033, 0.093] entirely above zero) -- the hypothesized direction, with enough signal this time because the corrected scoring's lower across-seed variance (SD 0.039, vs. 0.16 pre-correction) tightens the CI. On BGL, the comparison flips the other way: `deeplog`'s corrected drop (0.782) is now smaller than `semantic_knn`'s (0.874), so the difference is negative and CI-confirmed (-0.092, CI [-0.161, -0.023]) -- `deeplog` is the more robust arm on BGL, not `semantic_knn`. So H4 fails for two different reasons depending on the comparison: against `pca` it fails on both datasets because the data clearly point the *other* way; against `deeplog` it now fails because the one dataset where the hypothesized direction holds (HDFS_v1) is outweighed by the other (BGL) going the opposite way with equally high confidence -- H4 requires both datasets both comparisons, so one CI-confirmed dataset in each direction is not enough. See `results/full/addendum/deeplog_diagnosis/DIAGNOSIS.md` for the correction that produced this.

## H5: F1_oracle - F1_val under CHRONO, per arm

(Full table already in `paper/tables.md` Table 2; reproduced here with the aggregate used by the H5 verdict.)

| Dataset | Arm | F1_val | F1_oracle | Gap |
|---|---|---|---|---|
| BGL | deeplog | 0.3016 | 0.3690 | 0.0674 |
| BGL | iforest | 0.1565 | 0.1769 | 0.0204 |
| BGL | ocsvm | 0.1708 | 0.1769 | 0.0061 |
| BGL | pca | 0.1744 | 0.1854 | 0.0110 |
| BGL | semantic_knn | 0.1788 | 0.1794 | 0.0006 |
| BGL | supervised_lgbm | 0.2851 | 0.2898 | 0.0047 |
| HDFS_v1 | deeplog | 0.4414 | 0.6871 | 0.2457 |
| HDFS_v1 | iforest | 0.0428 | 0.3908 | 0.3481 |
| HDFS_v1 | ocsvm | 0.8684 | 0.8684 | 0.0000 |
| HDFS_v1 | pca | 0.8684 | 0.8684 | 0.0000 |
| HDFS_v1 | semantic_knn | 0.9767 | 0.9770 | 0.0003 |
| HDFS_v1 | supervised_lgbm | 0.9973 | 0.9994 | 0.0021 |

Aggregate used by the H5 verdict (mean gap across the 6 arms x 5 seeds = 30 cells, under CHRONO; **Step 12: recomputed from `results/full/verdicts_v2.json`, corrected `deeplog`**): BGL = 0.0184 [CI 0.0080, 0.0322]; HDFS_v1 = 0.0994 [CI 0.0469, 0.1524]. (Pre-correction, v1: BGL = 0.0076 [0.0053, 0.0102]; HDFS_v1 = 0.0606 [0.0169, 0.1082].)

**Paragraph (Step 12: updated; `deeplog`'s gap grew substantially under the corrected scoring, on both datasets).** Threshold leakage remains small for most arms on both datasets -- four of six BGL arms and three of six HDFS_v1 arms still have a gap under 0.02. But `deeplog`'s corrected gap is no longer negligible: BGL's `deeplog` gap rose from 0.0029 (v1) to 0.0674 (v2), and HDFS_v1's rose from 0.0130 (v1) to 0.2457 (v2) -- now the *second*-largest HDFS_v1 gap, close behind `iforest`'s 0.3481. This is a direct consequence of switching `deeplog`'s primary score from a saturated top-9 miss-rate (mostly exact zeros, so a validation-chosen threshold and a test-chosen threshold land in nearly the same flat region) to continuous mean-NLL, which separates sessions well enough that the choice of threshold now matters more. The aggregate mean gap still clears neither dataset's 0.10 bar outright: BGL's point estimate (0.0184) is still well below it, and HDFS_v1's point estimate (0.0994) is now within a hair of the bar but its CI (0.047 to 0.152) still straddles it both ways, so the verdict stays NOT CONFIRMED. HDFS_v1's aggregate is now driven by two outliers rather than one: `iforest` (0.348, because its validation-chosen threshold sits in a bad spot for that arm's score distribution -- unrelated to this correction) and the corrected `deeplog` (0.246, a direct and expected effect of moving to a less-saturated continuous score). Drop both and the HDFS_v1 mean gap would be roughly 0.001 across the remaining four arms -- i.e. the result is still "threshold leakage is negligible almost everywhere, with two arm-dataset combinations (one pre-existing, one introduced by the Step 11 scoring correction) where it is not." A reviewer interested in threshold leakage should look at `iforest` and the corrected `deeplog` specifically rather than treating either dataset's aggregate as representative of all six arms.
