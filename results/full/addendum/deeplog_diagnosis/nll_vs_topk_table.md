# NLL scoring vs top-9 scoring, both configs, both splits

| Config | Split | NLL AUROC | NLL AUPRC | NLL F1_val | NLL F1_oracle | Top9 AUROC | Top9 AUPRC | Top9 F1_val | Top9 F1_oracle | Spearman(nll,top9) |
|---|---|---|---|---|---|---|---|---|---|---|
| lightweight | random | 0.9910 | 0.7600 | 0.7016 | 0.7016 | 0.8823 | 0.7443 | 0.8377 | 0.8379 | 0.2655 |
| lightweight | chrono | 0.9906 | 0.7110 | 0.3458 | 0.6402 | 0.8047 | 0.5631 | 0.6142 | 0.6216 | 0.2082 |
| fullstrength | random | 0.9658 | 0.8266 | 0.7809 | 0.7820 | 0.7035 | 0.4233 | 0.5767 | 0.5767 | 0.1903 |
| fullstrength | chrono | 0.9927 | 0.8125 | 0.4950 | 0.7823 | 0.7767 | 0.5558 | 0.7022 | 0.7022 | 0.1594 |