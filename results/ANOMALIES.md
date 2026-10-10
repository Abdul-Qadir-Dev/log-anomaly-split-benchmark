# Anomalies

Surprising results from the full run (`results/full/summary.csv`, `results/full/VERDICTS.md`), with a best-effort explanation and a follow-up experiment for each. None of these change the pre-registered verdicts; they are reported as-is per the protocol's rule on honest reporting of negative/surprising results.

## 1. On HDFS_v1, three arms (pca, iforest, ocsvm) score *higher* AUPRC under CHRONO than RANDOM

Table 1 (`paper/tables.md`) and `results/full/VERDICTS.md` (H1 detail) show negative "drops" for `HDFS_v1/pca` (-0.326), `HDFS_v1/iforest` (-0.087), `HDFS_v1/ocsvm` (-0.228) -- i.e. chronological splitting *helped* these three arms on HDFS, the opposite of the split-inflation hypothesis.

**Best explanation:** HDFS's `unseen_template_rate` is ~0.0000 under *both* RANDOM and CHRONO (Table 3) -- with only 55 Drain3 templates total, a 60% chronological training prefix already sees essentially every template HDFS ever produces, so the usual mechanism for split inflation (training on templates the test set won't use) cannot operate here. What's left is session composition: HDFS anomalies cluster in particular stretches of the block timeline (e.g. batches of blocks affected by the same transient datanode issue), so a chronological test window can land on a stretch that is locally *easier* to separate by raw count-vector shape than a random 20% sample drawn uniformly from the whole log. This is a property of HDFS's specific anomaly distribution over time, not a general property of chronological evaluation.

**Follow-up experiment:** Plot anomaly rate in sliding time windows across the HDFS timeline; if anomalies cluster in bursts, a stratified-by-time-bucket random split (matching CHRONO's test window's local anomaly density) should reproduce RANDOM's higher score, confirming composition rather than drift is the driver.

## 2. BGL's `semantic_knn` AUROC is 0.3425 under CHRONO -- *worse than chance* (0.5)

An AUROC below 0.5 means the arm's anomaly score is, on average, *anti-correlated* with the true label: normal sessions get systematically higher scores than anomalous ones.

**Best explanation:** Under CHRONO, 77.5% of BGL test-session template occurrences are from templates never seen in training (Table 3). `semantic_knn`'s score is mean cosine distance to the nearest training (normal) session embeddings. If BGL's *anomalous* windows late in the timeline happen to reuse a narrower, more repetitive set of templates (alert bursts tend to repeat the same few KERNEL/RAS messages) while late *normal* windows accumulate more novel, never-before-seen routine-operation templates, then normal sessions would end up embedding farther from the training distribution than anomalous ones -- inverting the intended signal. This is plausible but not verified here.

**Follow-up experiment:** For CHRONO/BGL test sessions, compare the mean `unseen_template_rate` of normal vs. anomalous sessions directly; if normal sessions have a higher per-session unseen rate than anomalous ones, that confirms the mechanism above.

## 3. BGL `iforest` AUROC is 0.2519 under RANDOM -- also worse than chance, with no template drift to blame

Unlike anomaly 2, this happens on the RANDOM split, where `unseen_template_rate` is only 0.0004 -- so template drift cannot explain it.

**Best explanation:** Isolation Forest isolates points that are easy to separate with few random splits. BGL's fixed 100-line windows are labelled anomalous if *any* line in the window carries an alert tag (Section 5.1) -- so many "anomalous" windows are 99 ordinary lines plus one alert line, i.e. nearly identical in count-vector shape to a normal window. Meanwhile some *normal* windows (all 100 lines non-alert) may sit at the sparse tail of the template-count distribution (e.g. an unusually quiet or unusually busy stretch), making them easier for Isolation Forest to isolate than the alert-containing windows, which can dominate the count vector in high-frequency template counts and therefore look "dense"/normal to the forest. This would systematically invert the score.

**Follow-up experiment:** Re-run `iforest` on BGL with a session definition that requires a higher fraction of alert lines per window (e.g. >=10% alert lines) rather than "any alert line"; if AUROC rises back above 0.5, window purity is the driver.
