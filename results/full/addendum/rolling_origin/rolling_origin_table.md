# Rolling-origin robustness: chronological cutoffs at 50%, 60% (main run), 70%

Seed 0 only, all 6 non-LLM arms, both datasets. Robustness check only -- no verdict depends on this. `deeplog` rows reflect the Step 11/12 corrected (NLL) scoring; see `results/full/addendum/deeplog_diagnosis/DIAGNOSIS.md`.

| Dataset | Arm | Cutoff | n_test | unseen_template_rate | AUPRC | AUROC |
|---|---|---|---|---|---|---|
| HDFS_v1 | pca | 60% (main) | 115013 | 0.0000 | 0.9200 | 0.9987 |
| HDFS_v1 | pca | 50% | 172519 | 0.0000 | 0.4852 | 0.7834 |
| HDFS_v1 | pca | 70% | 57507 | 0.0000 | 0.9938 | 0.9999 |
| HDFS_v1 | iforest | 60% (main) | 115013 | 0.0000 | 0.1814 | 0.7725 |
| HDFS_v1 | iforest | 50% | 172519 | 0.0000 | 0.4176 | 0.7446 |
| HDFS_v1 | iforest | 70% | 57507 | 0.0000 | 0.4197 | 0.8560 |
| HDFS_v1 | ocsvm | 60% (main) | 115013 | 0.0000 | 0.9232 | 0.9988 |
| HDFS_v1 | ocsvm | 50% | 172519 | 0.0000 | 0.6723 | 0.8798 |
| HDFS_v1 | ocsvm | 70% | 57507 | 0.0000 | 0.9957 | 0.9999 |
| HDFS_v1 | deeplog | 60% (main) | 115013 | 0.0000 | 0.7465 | 0.9925 |
| HDFS_v1 | deeplog | 50% | 172519 | 0.0000 | 0.7146 | 0.9915 |
| HDFS_v1 | deeplog | 70% | 57507 | 0.0000 | 0.8368 | 0.9965 |
| HDFS_v1 | semantic_knn | 60% (main) | 115013 | 0.0000 | 0.9854 | 0.9998 |
| HDFS_v1 | semantic_knn | 50% | 172519 | 0.0000 | 0.9450 | 0.9988 |
| HDFS_v1 | semantic_knn | 70% | 57507 | 0.0000 | 1.0000 | 1.0000 |
| HDFS_v1 | supervised_lgbm | 60% (main) | 115013 | 0.0000 | 0.9997 | 1.0000 |
| HDFS_v1 | supervised_lgbm | 50% | 172519 | 0.0000 | 0.9999 | 1.0000 |
| HDFS_v1 | supervised_lgbm | 70% | 57507 | 0.0000 | 1.0000 | 1.0000 |
| BGL | pca | 60% (main) | 9496 | 0.7746 | 0.0802 | 0.4922 |
| BGL | pca | 50% | 14244 | 0.6924 | 0.1115 | 0.4683 |
| BGL | pca | 70% | 4748 | 0.8006 | 0.2604 | 0.6131 |
| BGL | iforest | 60% (main) | 9496 | 0.7746 | 0.0956 | 0.5459 |
| BGL | iforest | 50% | 14244 | 0.6924 | 0.1388 | 0.5532 |
| BGL | iforest | 70% | 4748 | 0.8006 | 0.1198 | 0.5615 |
| BGL | ocsvm | 60% (main) | 9496 | 0.7746 | 0.1049 | 0.5241 |
| BGL | ocsvm | 50% | 14244 | 0.6924 | 0.0970 | 0.4649 |
| BGL | ocsvm | 70% | 4748 | 0.8006 | 0.1400 | 0.5228 |
| BGL | deeplog | 60% (main) | 9496 | 0.7746 | 0.2433 | 0.7016 |
| BGL | deeplog | 50% | 14244 | 0.6924 | 0.1179 | 0.4383 |
| BGL | deeplog | 70% | 4748 | 0.8006 | 0.0755 | 0.2307 |
| BGL | semantic_knn | 60% (main) | 9496 | 0.7746 | 0.0639 | 0.3425 |
| BGL | semantic_knn | 50% | 14244 | 0.6924 | 0.1012 | 0.4504 |
| BGL | semantic_knn | 70% | 4748 | 0.8006 | 0.0801 | 0.2742 |
| BGL | supervised_lgbm | 60% (main) | 9496 | 0.7746 | 0.2523 | 0.6156 |
| BGL | supervised_lgbm | 50% | 14244 | 0.6924 | 0.3095 | 0.6233 |
| BGL | supervised_lgbm | 70% | 4748 | 0.8006 | 0.3354 | 0.6335 |