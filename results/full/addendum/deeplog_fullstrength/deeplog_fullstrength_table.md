# Full-strength DeepLog robustness run (Step 10)

HDFS_v1 only, seed 0, RANDOM and CHRONO. Config: window cap 1,000,000, up to 15 epochs with early stopping (patience 3) on validation loss, hidden 128, 2 layers, top-k=9 scoring. 90-minute wall-clock cap per split. Robustness only -- no verdict depends on this.

| Split | Config | AUROC | AUPRC | F1_val | F1_oracle | Epochs run | Stop reason | Fit time |
|---|---|---|---|---|---|---|---|---|
| random | full-strength | 0.6808 | 0.3797 | 0.5299 | 0.5302 | 13 | early_stopping_at_epoch_13_patience_3 | 1738s |
| random | main-run (100k windows, 5 epochs, hidden 64) | 0.9076 | 0.7786 | 0.8443 | 0.8445 | 5 | n/a (fixed) | 27s |
| chrono | full-strength | 0.7833 | 0.5569 | 0.6903 | 0.6906 | 4 | early_stopping_at_epoch_4_patience_3 | 510s |
| chrono | main-run (100k windows, 5 epochs, hidden 64) | 0.7896 | 0.5532 | 0.6320 | 0.6421 | 5 | n/a (fixed) | 32s |