# Template-drift mechanism test (Step 09 audit)

Diagnostic re-score only -- results/full/ is untouched; this re-fits semantic_knn on the existing BGL/CHRONO/seed0 split independently to compare against per-session unseen-template rate.

## BGL/CHRONO/seed0

- Mean per-session unseen-template rate, normal test sessions: **0.7675**
- Mean per-session unseen-template rate, anomalous test sessions: **0.8753**
- AUROC of unseen-rate-alone as an anomaly score: **0.4981**
- Spearman(unseen rate, semantic_knn score), normal sessions only: rho=0.5441, p=0.00e+00
- Spearman(unseen rate, semantic_knn score), anomalous sessions only: rho=0.6199, p=9.51e-87
- (For reference) semantic_knn AUROC on this diagnostic re-score: 0.3425 (results/full/BGL/chrono/semantic_knn/seed0.json records the same quantity from the frozen run)

## HDFS_v1/CHRONO/seed0 (normal-vs-anomalous unseen rate only)

- Mean per-session unseen-template rate, normal test sessions: **0.0000**
- Mean per-session unseen-template rate, anomalous test sessions: **0.0994**
- AUROC of unseen-rate-alone as an anomaly score: **0.7161**

**HDFS note.** Unlike BGL, HDFS_v1's near-zero *aggregate* unseen_template_rate (previously reported as ~0.0000) turns out to conceal a real, cleanly-separated signal at the per-session level: normal test sessions have *exactly zero* unseen templates, while anomalous sessions average 9.9%. Unseen-rate-alone is a meaningfully predictive single-feature anomaly score here (AUROC 0.716) -- on HDFS_v1, the handful of templates that are novel in the chronological test window are overwhelmingly associated with anomalous blocks, consistent with the real-world intuition that anomalous system behavior is what tends to produce new, previously-unseen log messages in the first place. This refines (does not contradict) `results/ANOMALIES.md` #1's speculation about time-clustered anomalies -- it points to a specific, template-level mechanism rather than just temporal clustering, and is a legitimate candidate follow-up experiment in its own right.

## Interpretation (BGL, main question)

**Does this support the drift explanation for `semantic_knn`'s below-chance AUROC on BGL/CHRONO?** Partially, and with an important gap. Both normal and anomalous BGL test sessions show very high per-session unseen-template rates (76.8% and 87.5% respectively) -- the entire chronological test period is swamped by novel vocabulary regardless of label, which is exactly why unseen-rate-alone is a near-useless single-feature anomaly score on its own here (AUROC 0.498, indistinguishable from chance): there is not enough separation between the two groups' unseen rates, relative to their spread, to tell them apart with this one number. Within each class, however, a session's unseen rate correlates strongly and positively with its `semantic_knn` score (Spearman rho=0.544 among normal sessions, rho=0.620 among anomalous sessions, both p<1e-80) -- exactly the mechanical relationship the arm is built on: more novel templates push a session's embedding farther from the training-normal centroid, raising its score, and this holds up cleanly in both classes independently.

That mechanical link also correctly predicts the *direction* of the (small) between-class gap: anomalous sessions do have a somewhat higher mean unseen rate than normal ones (0.875 vs. 0.768). But this is where the explanation runs out of road: a same-direction, modest-effect-size relationship between unseen rate and label would predict `semantic_knn`'s AUROC to land a little *above* 0.5, not substantially *below* it (0.342, the same value recorded in the frozen run). Raw unseen-template *volume* is therefore not sufficient to explain the below-chance result -- there must be a second factor, plausibly *which* specific templates go unseen rather than how many (e.g. a few templates that dominate normal sessions' embeddings happening to sit unusually far from the training centroid for reasons unrelated to anomalousness). This diagnostic does not have the data to isolate that second factor; it narrows the explanation rather than completing it, and is listed as a suggested follow-up (per `SUMMARY.md` section 7 / `results/ANOMALIES.md` #2's original follow-up suggestion).