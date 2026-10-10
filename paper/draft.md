Abdul Qadir
Independent researcher, Lahore, Pakistan
aq76301@gmail.com

# Does log anomaly detection performance depend on how the data is split?

## Abstract

Published results for log-based anomaly detection on HDFS and BGL often report F1 scores above 0.95. Many of those evaluations shuffle sessions randomly before splitting into train and test, and tune the decision threshold on the test set itself. We pre-registered a protocol that compares a random split against a chronological one (train on the past, test on the future), with the threshold chosen on a held-out validation set. We evaluate six classic and deep anomaly detection methods on HDFS and BGL with five seeds each, plus an exploratory small-LLM scorer reported separately. The main finding is that split inflation depends on the dataset and follows template drift; it is not a general property of chronological evaluation. The effect is large on BGL, where the chronological test period contains many log templates never seen in training, and nearly absent on HDFS, whose small vocabulary barely changes over time. The size of the effect rises with the fraction of test vocabulary unseen in training, which fits template drift as the mechanism. On HDFS, the apparent improvement under chronological splitting turns out to be mostly a test-set prevalence artifact rather than a real gain in ranking quality, which is a warning against reading AUPRC alone. Threshold leakage was small on both datasets. We release the pre-registered protocol, code and full results.

## 1. Introduction

Log-based anomaly detection is usually evaluated by taking a fixed log file, splitting its sessions into train and test sets, fitting a model, and reporting precision, recall, F1 or AUROC and AUPRC on the test set. Two choices in this recipe can inflate the reported numbers without being visible in the headline metric:

1. **How the split is drawn.** If sessions are shuffled before splitting, a session's near-duplicate (same block, same time window, same operator action) can land in both train and test. That would not matter if a log were i.i.d. over time, but system logs usually are not: one incident produces a burst of similar lines, and the set of templates a system emits changes as its software or workload changes. A chronological split, training on the earliest sessions and testing on the latest, removes this leakage and is closer to how a detector is deployed (trained on history, run on the future).
2. **How the decision threshold is chosen.** Many evaluations report F1 at the threshold that maximises F1 *on the test set itself*. Nothing in the number reveals this; it just means the reported F1 is an upper bound on what a deployed system, which has to pick a threshold without seeing the future, could reach.

This paper asks a narrow question: on two standard public log datasets, how much do these two choices change measured performance and the ranking of methods, and is the mechanism the expected one (new, unseen log templates in the test period)? We pre-registered five hypotheses before running any experiment, each with a numeric bar and a CI-based confirm / not-confirm / inconclusive rule (the full protocol is in the supplementary material), so that this is a confirmatory study rather than an exploratory one written up afterwards.

## 2. Related work

**Datasets.** HDFS and BGL are distributed as part of Loghub, a collection of system log datasets for log analytics research <!-- zhu2023loghub --> that was later extended with parsing benchmarks <!-- jiang2024loghub2 -->. He et al. set out the experience-report approach of applying classic anomaly detectors to parsed HDFS and BGL style logs, which much of this line of work, including this paper, still follows <!-- he2016experience -->. HDFS is a set of logs from a private-cloud Hadoop deployment labelled by block id; BGL is a BlueGene/L supercomputer log labelled per line by an alert tag.

**Methods.** Xu et al. introduced PCA on log-event count vectors for anomaly detection in console logs <!-- xu2009detecting -->. DeepLog treats anomaly detection as next-template prediction with an LSTM and flags a step as anomalous when the true next template falls outside the model's top-k predictions <!-- du2017deeplog -->. LogAnomaly adds a count-based branch to the same sequence idea <!-- meng2019loganomaly -->, and LogBERT adapts a BERT-style masked-token objective to log templates <!-- guo2021logbert -->. We include neither as a full arm; our `deeplog` and `semantic_knn` arms stand in for the sequence and embedding families they belong to. Isolation Forest <!-- liu2008isolation --> is a general-purpose unsupervised detector that we apply to the same count vectors. Drain is the parser we use to turn raw lines into templates <!-- he2017drain -->. Sentence-BERT style embeddings <!-- reimers2019sentencebert --> let us place a never-before-seen template in the same space as the training templates, which is the premise of the semantic k-NN arm. Landauer et al. survey the deep learning literature on log anomaly detection, including most of the methods above <!-- landauer2023deep -->.

**Re-evaluation studies.** Le and Zhang re-examined several published deep-learning log anomaly detectors and found that reported results are sensitive to data processing and evaluation choices, including how the train/test split is built <!-- le2022far -->. This paper follows the same spirit but narrows the question to one pre-registered comparison (random versus chronological splitting, validation versus test-tuned thresholds) with confidence intervals and a decision rule fixed in advance. Where Le and Zhang re-run existing deep methods under varied processing choices, we hold six method implementations fixed and vary only the split and the threshold policy, so that any measured difference is attributable to one of those two choices rather than to re-implementation differences.

## 3. Datasets and protocol

We use HDFS_v1 and BGL as distributed by Loghub (record 8196385). Both are parsed with Drain3 (similarity threshold 0.4, depth 4; block ids, IPs, hex values and integers are masked before clustering). <!-- data/raw/INVENTORY.md -->

HDFS sessions are defined per block id (one session is every log line that mentions that block), labelled from the dataset's own `anomaly_label.csv`. BGL sessions are fixed, non-overlapping windows of 100 consecutive lines, labelled anomalous if any line in the window carries an alert tag other than `-`. <!-- PROTOCOL.md Section 5.1 -->

| Dataset | Sessions | Anomaly rate | Distinct templates |
|---|---|---|---|
| HDFS_v1 | 575,061 | 2.93% | 55 |
| BGL | 47,480 | 10.12% | 1,346 |
<!-- results/data_stats.json -->

We compare two ways of splitting each dataset's sessions 60/20/20 into train, validation and test, with five seeds each:

- **RANDOM**: sessions are shuffled before the 60/20/20 cut.
- **CHRONO**: sessions are sorted by timestamp and cut 60/20/20 in time order. The split code asserts that the latest training timestamp precedes the earliest validation timestamp, which in turn precedes the earliest test timestamp, so no future session can leak into training.

For every (dataset, split, seed) cell we also record `unseen_template_rate`: the fraction of test-session template *occurrences* whose template id never appears in that cell's training sessions.

| Dataset | Split | Unseen template rate |
|---|---|---|
| HDFS_v1 | random | 1.590e-06 |
| HDFS_v1 | chrono | 0 (exactly) |
| BGL | random | 3.505e-04 |
| BGL | chrono | 0.7746 |
<!-- paper/tables.md Table 3, from results/full/summary.csv -->

Breaking HDFS_v1's CHRONO test sessions down by class (seed 0) shows that the near-zero aggregate hides a clean split by label. Normal test sessions have an unseen rate of exactly 0, while anomalous test sessions average 0.09943: almost all of the small amount of template novelty in HDFS_v1 sits in the anomalous class.

| Dataset | Split | Class | Unseen template rate (per-session mean) |
|---|---|---|---|
| HDFS_v1 | chrono | normal | 0 (exactly) |
| HDFS_v1 | chrono | anomalous | 0.09943 |
<!-- results/audit/drift_mechanism/HDFS_v1_unseen_rate_normal_vs_anomalous.json -->

This table previews the main result. HDFS has almost no unseen-template problem in aggregate under either split (its 55-template vocabulary is nearly all used within the first 60% of the timeline), while 77.46% of BGL's chronological test occurrences are unseen templates, a large distribution shift that random shuffling hides completely. A check for Drain3 over-fragmentation (a post-hoc analysis, supplementary material) found that only 1.55% of BGL's unseen test occurrences have a nearest training template above 0.8 token-level similarity, so most of this gap is new log content rather than the parser splitting near-duplicate messages into separate templates.

The protocol, including the five hypotheses and their numeric bars, was fixed and recorded before any arm was trained (supplementary material).

## 4. Methods (arms)

Six methods ("arms") are compared on both datasets, plus a supervised reference and an exploratory small-LLM scorer:

- **pca**: count vector of templates per session, PCA with enough components to explain 95% of variance, reconstruction error as the anomaly score.
- **iforest**: Isolation Forest (200 trees) on the same count vectors.
- **ocsvm**: One-Class SVM (RBF kernel) on TF-IDF count vectors, training subsampled to 50k sessions.
- **deeplog**: a 2-layer LSTM (hidden size 64) that predicts the next template from the previous 10, with training capped at 100k windows and early stopping on normal-only validation loss (up to 15 epochs, patience 3). The anomaly score is the mean per-position negative log-likelihood of the true next template. An earlier version of this arm scored sessions by the fraction of positions where the true next template fell outside the top-9 predictions, with a fixed 5-epoch schedule and no early stopping. A post-hoc review (labelled as such throughout) found that heuristic saturates almost completely on HDFS_v1's small vocabulary and corrected it; see the note at the end of this section and Section 5.6.
- **semantic_knn**: each distinct template is embedded once with `all-MiniLM-L6-v2`; a session's vector is the count-weighted mean of its template embeddings; the score is the mean cosine distance to the 10 nearest training (normal) sessions.
- **supervised_lgbm**: LightGBM on count vectors trained with true labels. This is a ceiling reference, not a competitor for H1, H2 or H4.
- **llm_zeroshot** (exploratory, sample-based): a small chat-tuned LLM is shown a session's template sequence (truncated to 40 lines) and asked for an anomaly probability from 0 to 100; the number is parsed as the score. The main-run attempt with `Qwen2.5-1.5B-Instruct` was marked SKIPPED after repeated GPU-memory contention and a Windows pagefile limit made it unreliable on the machine used. A post-hoc analysis (labelled as such throughout) retried it with the smaller `Qwen2.5-0.5B-Instruct` (fp16, GPU), which completed on 200 sessions (100 anomalous, 100 normal) per dataset per split, seed 0 only <!-- results/audit/llm_table.md -->. This arm has no pre-registered hypothesis and is reported separately in Section 5.7; it is never part of the six-arm ranking (H1, H2, H4) or any verdict.

For the five unsupervised and one-class arms, training uses only sessions labelled normal in the training split, which is standard practice for this family of methods (DeepLog among them); `supervised_lgbm` trains on all labels.

**A note on a post-hoc scoring correction to `deeplog`.** The pre-registered run scored `deeplog` by whether the true next template fell outside its top-9 predictions. A post-hoc check found that this score is exactly zero for 97 to 99% of HDFS_v1 test sessions, so `deeplog` was re-scored with mean negative log-likelihood and all of its cells were re-run. The pre-registered verdicts did not change. Both the original and the corrected result sets are released in full; Section 5.6 gives the diagnosis.

## 5. Results

All numbers below are the mean and standard deviation across five seeds, computed on the test split. They are reproduced in the supplementary tables, which are generated from the full results so that every number here can be traced to its source.

**Reading the `deeplog` numbers.** Following the correction in Section 4, every number below that involves `deeplog` is given as the corrected (mean-NLL) value first, with the original pre-registered (top-9) value alongside. Numbers that do not involve `deeplog` are the same under both versions. Section 5.6 has the full diagnosis.

### 5.1 AUPRC, random vs. chronological

<!-- figures/fig1_random_vs_chrono.png -->

On **BGL**, five of six arms lose a large amount of AUPRC under CHRONO: pca drops by 0.749, ocsvm by 0.747, deeplog by 0.782 (corrected scoring; 0.875 with the original scoring, Section 5.6), semantic_knn by 0.874 and supervised_lgbm by 0.742, all with 95% CIs excluding zero. Only iforest's drop (0.022) fails the pre-registered 0.05 bar. <!-- results/full/VERDICTS.md H1 detail; results/full/verdicts_v2.json for the corrected deeplog number -->

On **HDFS_v1**, only one arm meets the bar: the corrected deeplog drops by 0.074 (95% CI [0.046, 0.106]; with the original scoring the drop was 0.014 with a CI spanning zero). Three arms (pca, iforest, ocsvm) score *higher* AUPRC under CHRONO than under RANDOM; pca's AUPRC, for example, rises from 0.594 to 0.920. The next paragraph and Section 5.6 explain why. <!-- results/full/verdicts_v2.json H1 detail -->

**H1 (split inflation) verdict: CONFIRMED.** The pre-registered bar required at least 4 of 6 arms to show a drop of at least 0.05 AUPRC (CI excluding zero) on *at least one* dataset; BGL meets this with 5 of 6 arms, under both scoring versions. <!-- results/full/verdicts.json H1; results/full/verdicts_v2.json H1 -->

**Why HDFS_v1's apparent AUPRC gains under CHRONO are not what they look like.** A post-hoc analysis computed the same RANDOM-to-CHRONO drop on AUROC, which, unlike AUPRC, does not depend on the positive-class rate. For `pca`, `iforest` and `ocsvm` on HDFS_v1, the large negative AUPRC "drops" (apparent gains of 0.09 to 0.33 under CHRONO) shrink to between -0.003 and -0.023 on AUROC, and AUROC is close to flat across split conditions for every HDFS_v1 arm whose scoring was not revised. <!-- results/full/addendum/auroc_drops.md --> The cause is a prevalence artifact of where this dataset's single chronological cut falls: HDFS_v1's test-set anomaly prevalence is 2.96% under RANDOM but 1.46% under CHRONO. The positive rate nearly halves because of how anomalous blocks happen to be spread over the timeline, not because of anything about the split method. <!-- results/full/addendum/prevalence.csv --> Since AUPRC moves with the positive-class rate and AUROC does not, most of HDFS_v1's apparent "chronological splitting helps" result is better read as this prevalence shift than as a gain in discriminability. BGL's prevalence shift between conditions is much smaller (9.73% random versus 8.49% chrono, about 13% relative), and its AUROC drops track its AUPRC drops closely: five of six arms show positive AUROC drops of 0.38 to 0.65 with CIs excluding zero. BGL's split-inflation result is therefore a real loss of ranking quality, not a prevalence artifact. <!-- results/full/addendum/auroc_drops.md --> The one BGL exception is `iforest`, whose AUROC drop is large and *negative* despite a near-zero AUPRC drop. A separate post-hoc note found that this arm's score is already inverted on BGL under RANDOM, for reasons unrelated to chronological drift.

### 5.2 Does the ranking of methods change?

<!-- figures/fig4_bump_chart.png -->

With the corrected `deeplog` scoring (Section 5.6), the rank order of the six arms by AUPRC on BGL is weakly correlated between RANDOM and CHRONO (Kendall tau = 0.47). `supervised_lgbm` and `deeplog` hold the top two places in both conditions, but `semantic_knn` falls from 3rd (random) to last (chrono), and `ocsvm`, `pca` and `iforest` reorder beneath them. On HDFS_v1 the ranking is no longer identical between conditions (tau = 0.73, also below the 0.8 bar): `deeplog` moves from 3rd (random) to 5th (chrono) while the other five arms keep their relative order. <!-- results/full/verdicts_v2.json H2 detail --> With the original pre-registered `deeplog` scoring, BGL's tau was 0.20 and HDFS_v1's was 1.00 (supplementary material); the conclusion that BGL's ranking is unstable holds under both versions.

**H2 (ranking instability) verdict: CONFIRMED.** The bar required tau below 0.8 on at least one dataset; BGL clears it under both the pre-registered and the corrected scoring. <!-- results/full/verdicts.json H2; results/full/verdicts_v2.json H2 -->

### 5.3 Is the drop consistent with unseen-template rate as the mechanism?

<!-- figures/fig2_unseen_vs_drop.png -->

Pooling all 60 (dataset x seed x arm) cells, the per-cell AUPRC drop from RANDOM to CHRONO correlates strongly with that cell's CHRONO-split unseen_template_rate (Spearman rho = 0.80, p = 1.2e-14, n = 60; rho = 0.83, p = 3.5e-16 with the original pre-registered `deeplog` scoring, so the correction in Section 5.6 barely moves this relationship). <!-- results/full/verdicts_v2.json H3; results/full/verdicts.json H3 for the pre-registered number --> We report this as **consistent with** unseen-template rate being part of the mechanism, not as proof of it. Although there are 60 cells, the effective sample size for generalising across *datasets* is 2: the 5 seeds x 6 arms within a dataset are correlated draws from the same split, not 30 independent observations. The correlation mostly describes a clean separation between one dataset with near-zero drift (HDFS_v1) and one with heavy drift (BGL), not a smooth relationship verified across many datasets.

A post-hoc analysis tested the mechanism within a single dataset and split. On BGL/CHRONO, a session's unseen-template rate correlates positively with its `semantic_knn` score within the normal class and within the anomalous class separately (Spearman rho = 0.54 and 0.62, both p < 1e-80), which is the relationship the drift explanation predicts. Unseen rate alone, though, is at chance level (AUROC 0.50) for telling normal and anomalous BGL sessions apart, because *both* classes have high unseen rates under heavy chronological drift (76.8% and 87.5%). On HDFS_v1/CHRONO, where overall drift is near zero, the picture is cleaner: normal test sessions have exactly 0% unseen templates, anomalous sessions average 9.9%, and unseen rate alone is a moderately useful anomaly score on its own (AUROC 0.72). <!-- results/audit/drift_mechanism/REPORT.md --> Both findings fit the drift mechanism as hypothesised; neither is independent confirmation on a third dataset.

**H3 (unseen templates, consistent with explaining degradation) verdict: CONFIRMED.** The pre-registered bar was rho above 0.4 with p below 0.05, which the pooled correlation clears; see the caveat above on what n = 60 represents.

### 5.4 Is the semantic embedding arm more robust to unseen templates?

We hypothesised that `semantic_knn`, which can embed a template it has never seen in training, would degrade less than `pca` or `deeplog` under CHRONO. Against `pca` it does not: on both datasets `semantic_knn`'s AUPRC drop is *larger* than `pca`'s (mean difference -0.339 on HDFS_v1, -0.125 on BGL; these comparisons are unaffected by the `deeplog` correction). Against the corrected `deeplog` the result is mixed. On HDFS_v1, `semantic_knn` degrades less than `deeplog` (mean difference 0.062, 95% CI [0.033, 0.093]), but on BGL the opposite holds (mean difference -0.092, 95% CI [-0.161, -0.023]). Because H4 requires both comparisons to favour `semantic_knn` on both datasets, the split result does not change the verdict. <!-- results/full/verdicts_v2.json H4 detail -->

**H4 (semantic robustness) verdict: NOT CONFIRMED.** <!-- results/full/verdicts.json H4 -->

An embedding that *can* represent a novel template did not produce a more robust anomaly score in this setup. One surprising case, `semantic_knn` scoring *below chance* on BGL/CHRONO, is examined further, with a candidate explanation and the within-dataset test described in Section 5.3, in the supplementary material. <!-- results/ANOMALIES.md item 2; results/audit/drift_mechanism/REPORT.md -->

### 5.5 Threshold leakage

<!-- figures/fig3_oracle_vs_val.png -->

Under CHRONO, the mean gap between the test-tuned F1 (F1_oracle) and the validation-tuned F1 (F1_val) across the six arms is 0.018 on BGL and 0.099 on HDFS_v1 with the corrected `deeplog` scoring (0.0076 and 0.0606 with the original scoring). Both versions are below the pre-registered 0.10 bar, though HDFS_v1 is now just under it. BGL's 95% CI does not reach the bar under either version. `deeplog`'s own gap grew under the corrected scoring (BGL: 0.003 to 0.067; HDFS_v1: 0.013 to 0.246). This is an expected side effect of moving from a near-constant, saturated top-9 score, where the validation and test thresholds land in the same flat region, to a continuous score that separates sessions well enough for the threshold choice to matter. HDFS_v1's aggregate is now driven by two arm-level outliers rather than one: `iforest` (gap 0.348, from a very low validation-chosen F1 of 0.043 against a test-tuned F1 of 0.391, unrelated to the `deeplog` correction) and the corrected `deeplog` (gap 0.246). That is still an arm-level outlier pattern, not the aggregate the hypothesis was registered against. <!-- paper/tables.md Table 2; results/full/verdicts_v2.json H5; results/audit/h4_h5_diagnosis.md -->

**H5 (threshold leakage) verdict: NOT CONFIRMED.** <!-- results/full/verdicts.json H5 -->

### 5.6 Robustness checks

These variants are reported for context and were never used to decide H1 to H5. They were either fixed in the pre-registered protocol before being run or, for the rolling-origin check, added as a post-hoc analysis scoped as "robustness only, no verdict".

- **BGL, 1-hour time windows instead of fixed 100-line windows**: the same qualitative pattern holds, with large RANDOM-to-CHRONO differences and a still-high (0.79) CHRONO unseen-template rate, though the direction and size of individual arms' changes differ in detail from the fixed-window analysis.
- **HDFS_v1, 70/15/15 split instead of 60/20/20**: results are close to the 60/20/20 analysis for every arm (for example `supervised_lgbm` AUPRC 0.999 under random and 1.000 under chrono), consistent with HDFS_v1's near-zero unseen-template rate being a property of the dataset rather than of the split proportions.
- **semantic_knn with k = 5 and k = 20** (instead of k = 10): AUPRC is stable across k on both datasets and both split conditions (HDFS_v1/chrono: 0.985 at k = 5 and 0.985 at k = 20), so the choice of k = 10 is not driving the H4 result.

<!-- results/full/robustness.json -->

**Rolling-origin check (chronological cutoffs at 50% and 70%, in addition to the main 60%).** Section 7 notes that the chronological result is a single train/test split per dataset, which is a real limitation for a time-ordered claim. As a mitigation, we re-ran all six non-LLM arms at two further chronological cutoffs (train on the first 50% or 70%, validate on the next 20%, test on the remainder), seed 0 only; the `deeplog` rows use the corrected NLL scoring. <!-- results/full/addendum/rolling_origin/rolling_origin_table.md --> The two datasets respond differently. **BGL's degradation is stable across all three cutoffs when the six arms are taken together**: AUPRC stays between 0.06 and 0.34 at every cutoff, far below BGL's RANDOM-split AUPRC of 0.83 to 0.99, and `unseen_template_rate` stays high throughout (0.69 to 0.80), so BGL's core finding does not depend on choosing 60%. One arm is an exception. The corrected `deeplog` swings more than any other arm across BGL's three cutoffs (AUPRC 0.118 at 50%, 0.243 at 60%, 0.076 at 70%; AUROC falls to 0.231 at 70%, below chance), and not monotonically in the cutoff. We attribute this to the single-seed training non-determinism already noted for this arm (Section 7) rather than to a new systematic pattern, since it stays within the aggregate 0.06 to 0.34 band either way. **HDFS_v1's result is not stable.** At the 50% cutoff, `pca`'s AUPRC falls to 0.485 (versus 0.920 at 60% and 0.994 at 70%) and `ocsvm`'s to 0.672 (versus 0.923 and 0.996), even though `unseen_template_rate` is exactly 0.0000 at every cutoff. Template drift cannot explain this swing, so the likeliest driver is the prevalence sensitivity discussed in Section 5.1: the size and composition of the test window change substantially across cutoffs (30% of the data at the 50% cutoff versus 10% at the 70% cutoff), and AUPRC responds to that in a way AUROC does not. The corrected `deeplog` is comparatively stable on HDFS_v1 across the same cutoffs (AUPRC 0.715 at 50%, 0.747 at 60%, 0.837 at 70%). This is a real limitation of the single main-run HDFS_v1 CHRONO split, and Section 7 records it.

**Full-strength DeepLog and the top-k scoring defect (HDFS_v1, seed 0).** The main-run `deeplog` arm used a deliberately light configuration (100k-window training cap, 5 fixed epochs, hidden size 64) to fit the compute budget (Section 7). A first check re-ran it on HDFS_v1 with a much larger budget (up to 1,000,000 training windows, up to 15 epochs with early stopping, hidden size 128) and found the full-strength model's RANDOM AUPRC *falling* sharply (0.779 to 0.380) while CHRONO stayed roughly flat, which looked like a reversal of DeepLog's RANDOM-to-CHRONO gap. That result did not hold up. A dedicated diagnosis found that DeepLog's top-k-miss score saturates almost completely on HDFS_v1's 55-template vocabulary: 97.5 to 99.2% of test sessions scored exactly zero in *both* the light and the full-strength configuration. This was confirmed by inspecting the score distributions directly, not only the summary AUROC and AUPRC, and checkpoint hashing ruled out a loading or scoring bug. A continuous alternative, the mean negative log-likelihood of the true next template, restored strong and stable ranking (AUROC 0.97 to 0.99) in every configuration tested, while a sweep over k showed top-k's AUROC and AUPRC both varying substantially and non-monotonically with k (supplementary material). Because the light configuration, which matches the main run in architecture and procedure, was itself saturated above the same threshold, we treated this as a defect in the main run's `deeplog` arm rather than a finding about model capacity. The arm's scoring was corrected (mean NLL as the primary score, top-9 kept as a secondary diagnostic, early stopping added), all `deeplog` cells were re-run, and all five verdicts were recomputed from scratch. **No verdict changed.** With the corrected scoring, `deeplog`'s HDFS_v1 RANDOM-to-CHRONO relationship is unremarkable: RANDOM modestly outperforms CHRONO (mean AUPRC drop 0.074, 95% CI [0.046, 0.106], which now clears H1's 0.05 bar, where the original scoring had shown a difference indistinguishable from zero). That is the same-direction pattern most other HDFS_v1 arms show, not a reversal. The apparent "stronger model reverses the sign" result was an artifact of how differently the two configurations' top-k scores saturated, not a property of training intensity interacting with chronological splitting.

| Config | Split | k=1 | k=2 | k=3 | k=5 | k=7 | k=9 |
|---|---|---|---|---|---|---|---|
| lightweight | random (AUROC) | 0.872 | 0.853 | 0.889 | 0.909 | 0.903 | 0.882 |
| lightweight | chrono (AUROC) | 0.946 | 0.825 | 0.804 | 0.807 | 0.808 | 0.805 |
| fullstrength | random (AUROC) | 0.854 | 0.820 | 0.799 | 0.744 | 0.729 | 0.704 |
| fullstrength | chrono (AUROC) | 0.938 | 0.808 | 0.802 | 0.787 | 0.786 | 0.777 |
<!-- results/full/addendum/deeplog_diagnosis/k_sweep_table.md (also has AUPRC) -->

AUROC is neither monotonic in k nor consistent in the direction of its trend across configurations (lightweight/random *rises* with k through k = 5 and then falls; fullstrength/random falls steadily from k = 1). On a 55-template vocabulary, top-k is a fragile statistic whose value depends on the choice of k, not a robust ranking score.

### 5.7 Exploratory: zero-shot LLM scoring

`llm_zeroshot` has no pre-registered hypothesis and is never part of the six-arm ranking above; it is reported for interest. The main-run attempt with `Qwen2.5-1.5B-Instruct` was marked SKIPPED (Section 4). A post-hoc analysis retried it with the smaller `Qwen2.5-0.5B-Instruct` (fp16, GPU) on 200 sessions per dataset per split (100 anomalous, 100 normal), seed 0 only:

| Dataset | Split | AUROC | AUPRC |
|---|---|---|---|
| HDFS_v1 | random | 0.438 | 0.488 |
| HDFS_v1 | chrono | 0.369 | 0.444 |
| BGL | random | 0.745 | 0.693 |
| BGL | chrono | 0.651 | 0.592 |
<!-- results/audit/llm_table.md -->

The small zero-shot scorer does *worse than chance* on HDFS_v1 (AUROC 0.37 to 0.44) but clearly better than chance on BGL (AUROC 0.65 to 0.75), the opposite of the difficulty pattern most of the six main arms show. One plausible reading is that HDFS_v1's repetitive 55-template vocabulary gives the model little distinguishing text to work with (most sessions look nearly identical as token sequences), while BGL's more varied error messages (1,346 templates) give it more to go on. With one seed and a 200-session sample per cell this is an observation, not a result we draw conclusions from.

## 6. Discussion

**For practitioners**, three recommendations follow from Section 5. First, report a chronological-split result alongside any random-split result, because the gap can be very large and depends on the dataset. Second, report `unseen_template_rate` for the test window; it is a strong, measurable predictor of how much a chronological split will hurt a method, and costs nothing beyond a parse. Third, choose the decision threshold on a validation window that ends before the test window starts, not on the test window itself. Threshold leakage was small in our setup, but it is a methodological error whatever its size in any one experiment, and `iforest` on HDFS_v1 shows it can be large for an individual method.

**On method choice**, our results do not support "use a semantic or embedding method because it survives template drift better" as a general recommendation. H4 was not confirmed, and in one case (`semantic_knn` on BGL/CHRONO) the embedding arm scored *worse than chance*. The one method that was robust across both splitting conditions was the supervised reference, `supervised_lgbm`. That is unsurprising, since it trains with labels, but it is also the method practitioners usually cannot deploy, because labelled anomalies are precisely what is scarce in production.

**The HDFS result shows that "chronological splitting lowers scores" is not a general law.** The degradation on BGL fits template drift over time, and HDFS_v1's small, largely time-invariant vocabulary (55 templates, about 0% unseen under either split) means that mechanism does not apply there. A dataset-specific diagnostic such as `unseen_template_rate` generalises better than a blanket rule. The HDFS_v1 "chronological splitting helps" finding should itself be read with caution, for two reasons found in post-hoc analyses: it is largely a prevalence artifact rather than a gain in discriminability (Section 5.1's AUROC comparison), and it is not stable across where the chronological cut falls (Section 5.6's rolling-origin check). BGL's degradation finding holds on both counts. The practical lesson is the same either way: do not trust a single split's AUPRC, chronological or random, without checking whether AUROC and a different cutoff tell the same story.

## 7. Threats to validity

- **Two datasets, and effectively two independent data points for H3.** HDFS_v1 and BGL are standard benchmarks but do not represent every logging environment; a third dataset with a different drift profile could shift the balance of evidence for H1, H2 and H3. The H3 correlation pools 60 (dataset x seed x arm) cells, but the 5 seeds x 6 arms within a dataset are correlated, not independent draws, so the real sample size for generalising the drift mechanism across datasets is 2, not 60. For this reason we report H3 as "consistent with" the drift mechanism (Section 5.3), not as a validated general law.
- **Session definition for BGL.** Fixed 100-line windows are a common but arbitrary choice. The 1-hour-window robustness check shows the qualitative pattern holds, but we did not search window definitions exhaustively.
- **Five seeds.** The CIs for cross-condition differences (H1, H4, H5) are bootstrap resamples of only five per-seed values (methodology note in the supplementary material); more seeds would tighten these intervals.
- **One parser configuration.** All results depend on one Drain3 similarity threshold (0.4) and depth (4). A different parser or configuration would produce a different template vocabulary and could change `unseen_template_rate`, and with it H1 and H3.
- **One chronological split per dataset in the main run.** H1 to H5 are evaluated at a single 60% train cutoff per dataset; the five seeds control downstream randomness, not the cutoff location. The post-hoc rolling-origin check (Section 5.6) at 50% and 70% cutoffs shows this matters: BGL's degradation is stable across all three cutoffs, but HDFS_v1's AUPRC varies substantially by cutoff (`pca` AUPRC 0.485 at 50%, 0.920 at 60%, 0.994 at 70%) even though `unseen_template_rate` stays at 0 throughout. HDFS_v1's specific numeric results, though not its qualitative "near-zero drift" story, are sensitive to where the single chronological split falls. A rolling-origin design averaged over several cutoffs would be more robust than our single-cutoff main run and is a natural extension of this protocol.
- **A small fraction of BGL's "unseen" templates are parser artifacts rather than new content.** A token-similarity check (Jaccard similarity against every training template) found that 98.45% of BGL/CHRONO's unseen test occurrences have no close training match (similarity at most 0.8), which supports new content as the dominant explanation. A handful of the highest-frequency unseen templates (similarity up to 0.86), though, are plainly the same message as a training template with an extra trailing clause, which Drain3 clustered as a distinct template rather than a variant. `unseen_template_rate` is therefore a slight overestimate of content novelty, by an amount we quantified only for this one dataset, split and seed. <!-- results/full/addendum/unseen_templates_bgl.md -->
- **AUPRC's sensitivity to test-set prevalence is a second confound, distinct from template drift.** HDFS_v1's RANDOM and CHRONO test splits have different anomaly prevalence (2.96% versus 1.46%) purely because of where anomalous blocks fall in the timeline. AUROC, which does not depend on prevalence, shows HDFS_v1's split effect is close to flat for every arm, while AUPRC shows large swings (Section 5.1). Any paper that reports only AUPRC for a chronological-versus-random comparison should check whether the two conditions' test-set prevalence matches before attributing the AUPRC gap to the split method.
- **`deeplog`'s absolute HDFS_v1 performance under the original pre-registered scoring is well below commonly cited DeepLog results.** The original RANDOM-split `deeplog` reaches AUPRC 0.589, AUROC 0.806 and F1 of about 0.71 on HDFS_v1 <!-- results/full/addendum/baseline_sanity.md -->, well below the numbers usually reported for DeepLog. Two causes contribute: (1) training windows were capped at 100k, with only 5 fixed epochs and a small (hidden size 64) LSTM, to fit a 90-minute-per-arm cap (the pre-registered compute plan, supplementary material) across 120 cells; and (2) the original top-k-miss score saturates severely on HDFS_v1's 55-template vocabulary (Section 5.6), independently of the training budget. The corrected scoring (mean negative log-likelihood, Section 5.6) scores much higher on the same data (RANDOM AUROC 0.97 to 0.99 across the configurations tested in the diagnosis), which indicates that the saturated top-k score, not under-training, was the main cause of the low absolute numbers.
- **Top-k scoring depends on the choice of k and saturates on small template vocabularies.** A sweep over k in {1, 2, 3, 5, 7, 9} (Section 5.6) shows `deeplog`'s AUROC and AUPRC on HDFS_v1 varying substantially and non-monotonically with k, and the fraction of sessions scoring exactly zero (no top-k miss anywhere in the session) exceeds 97% at k = 9 regardless of training configuration. Any top-k anomaly score should be checked for this failure mode on a small-vocabulary dataset before it is trusted. We did not check it for the original pre-registered run and only caught it in a later, deliberate review.
- **The released `deeplog` cells were trained before training was fully seeded.** `deeplog`'s training (random, numpy, torch and cuDNN) was not fully seeded when the released cells were produced, so re-running them reproduces each cell's result within ordinary seed-to-seed variance (its across-seed AUPRC standard deviation ranges from 0.0055 to 0.0903 depending on dataset and split) but not bit for bit. Every other arm is exactly reproducible given the same seed, as is any `deeplog` cell trained after the fix.
- **The exploratory LLM arm is a small, single-seed, 200-sample-per-cell check, not a full evaluation.** `llm_zeroshot`'s main-run attempt with the 1.5B model failed for machine-specific reasons (Section 4). The 0.5B retry succeeded but was never run at the five-seed, 400-session scale of the other arms, and has no pre-registered hypothesis or verdict. Its results (Section 5.7) are suggestive only and should not be compared on equal footing with the six main arms.
- **BGL label definition.** A BGL window is labelled anomalous if *any* line carries an alert tag. A supplementary note on this coarse labelling shows it likely contributes to `iforest`'s below-chance AUROC on BGL/RANDOM, a confound unrelated to the split-inflation question. <!-- results/ANOMALIES.md item 3 -->

## 8. Conclusion

Split inflation in log anomaly detection depends on the dataset and follows template drift; it is not a general property of chronological evaluation. Chronological splitting degraded both AUPRC and AUROC on BGL, where template drift between the training and test periods is large and holds across several chronological cutoffs (77% of test template occurrences unseen in training at the main 60% cutoff, and still high at 50% and 70%). It did not degrade HDFS_v1, where the template vocabulary is small and stable over time, and the modest AUPRC *gains* measured there turned out on closer inspection to be mostly a test-set prevalence artifact (AUROC stayed nearly flat) that does not survive moving the chronological cut. The size of the effect follows a cheap diagnostic, `unseen_template_rate`; our data fit that relationship, though with an effective sample size of two datasets we stop short of calling it proven. We recommend reporting it, together with both AUPRC and AUROC and more than one chronological cutoff, for any log anomaly detection result. Threshold leakage from test-set tuning was small in our setup but remains a methodological hazard independent of its measured size here. A later review also found and corrected a scoring defect in `deeplog` (its original top-k score saturated almost completely on HDFS_v1's small vocabulary); every verdict was recomputed after the fix and none changed, and both the pre-registered and the corrected result sets are released in full. An exploratory small-LLM zero-shot scorer, evaluated separately without a pre-registered hypothesis, showed the opposite difficulty pattern from most other arms (worse than chance on HDFS_v1, better than chance on BGL), which future work could investigate properly. We release the pre-registered protocol, code and full results, including the complete record of post-hoc analyses, so these findings can be checked, extended to other datasets, and re-run as new methods appear.

## References

See `references.bib` in the repository.
