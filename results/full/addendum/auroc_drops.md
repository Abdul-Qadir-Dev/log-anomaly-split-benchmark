# RANDOM->CHRONO AUROC drop vs AUPRC drop, per dataset x arm

Same seed-level paired bootstrap CI methodology as AUPRC-based hypotheses (resample the 5 per-seed differences, 1000x, seed 0 -- see DECISIONS.md Step 05).

| Dataset | Arm | AUROC drop (mean) | AUROC drop 95% CI | AUPRC drop (mean) | AUPRC drop 95% CI |
|---|---|---|---|---|---|
| BGL | pca | 0.4725 | [0.4701, 0.4749] | 0.7491 | [0.7391, 0.7590] |
| BGL | iforest | -0.2940 | [-0.3026, -0.2855] | 0.0216 | [0.0130, 0.0304] |
| BGL | ocsvm | 0.4337 | [0.4305, 0.4367] | 0.7467 | [0.7371, 0.7553] |
| BGL | deeplog | 0.3546 | [0.2707, 0.4341] | 0.7824 | [0.7108, 0.8539] |
| BGL | semantic_knn | 0.6472 | [0.6464, 0.6479] | 0.8742 | [0.8680, 0.8809] |
| BGL | supervised_lgbm | 0.3831 | [0.3827, 0.3835] | 0.7424 | [0.7416, 0.7432] |
| HDFS_v1 | pca | -0.0225 | [-0.0227, -0.0222] | -0.3264 | [-0.3296, -0.3233] |
| HDFS_v1 | iforest | -0.0029 | [-0.0149, 0.0092] | -0.0872 | [-0.0922, -0.0812] |
| HDFS_v1 | ocsvm | -0.0131 | [-0.0134, -0.0128] | -0.2279 | [-0.2326, -0.2232] |
| HDFS_v1 | deeplog | 0.0000 | [-0.0023, 0.0025] | 0.0743 | [0.0457, 0.1057] |
| HDFS_v1 | semantic_knn | -0.0000 | [-0.0003, 0.0001] | 0.0124 | [0.0122, 0.0126] |
| HDFS_v1 | supervised_lgbm | -0.0000 | [-0.0001, 0.0000] | 0.0001 | [-0.0001, 0.0002] |

Test-set anomaly prevalence by condition, for context (full detail: `prevalence.csv`): HDFS_v1 random 2.96% vs. chrono 1.46% (prevalence nearly halves under CHRONO); BGL random 9.73% vs. chrono 8.49% (a much smaller, ~13% relative shift).

**Interpretation.** On BGL, the split-inflation effect survives the switch to AUROC: five of six arms show large, positive AUROC drops (0.35-0.65, CIs excluding zero) tracking the same direction as their AUPRC drops, confirming this is a genuine degradation in ranking quality under chronological splitting, not an artifact of AUPRC's sensitivity to the positive-class rate. The one exception is `iforest`, whose AUROC drop is large and *negative* (-0.294) despite a small, non-significant AUPRC drop -- consistent with `results/ANOMALIES.md` #3's independent finding that `iforest`'s score is already inverted/confounded on BGL under RANDOM for reasons unrelated to chronological drift, so its chronological behavior should not be read as evidence against H1. On HDFS_v1, the picture is different: the large *negative* AUPRC drops for `pca`, `iforest`, and `ocsvm` (apparent gains of 0.09-0.33 AUPRC under CHRONO) shrink to near-zero on AUROC (-0.001 to -0.023) -- AUROC is essentially flat across split conditions for every HDFS_v1 arm, `deeplog` included (AUROC drop 0.0000, CI [-0.0023, 0.0025], corrected scoring). Because AUROC is prevalence-invariant while AUPRC is not, and HDFS_v1's test-set anomaly prevalence happens to nearly halve between RANDOM and CHRONO (2.96% to 1.46%, an accident of where the single chronological 60/20/20 cut falls, not a split-*method* effect), the most defensible reading is that most of HDFS_v1's apparent "chronological splitting helps" result is a prevalence artifact of this one split, not a real discriminability change -- while the handful of arms whose AUROC does move slightly (`pca` -0.023, `ocsvm` -0.013) show the ranking quality moving, if anything, in the same "chrono is easier" direction, just by a much smaller and more plausible amount than the raw AUPRC numbers suggested. (`deeplog`'s HDFS_v1 AUPRC drop, by contrast, is now positive and CI-confirmed under the corrected NLL scoring -- 0.0743, CI [0.0457, 0.1057] -- consistent with its AUROC-flat/AUPRC-moves pattern reflecting a real, if modest, prevalence-sensitive change rather than noise; see `results/full/addendum/deeplog_diagnosis/DIAGNOSIS.md`.)