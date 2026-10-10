# Verdicts

## H1: CONFIRMED

**Bar:** AUPRC_random - AUPRC_chrono >= 0.05, CI excl. 0, on >=4/6 arms, on >=1 dataset

**Interpretation:** Random splits substantially inflate AUPRC relative to chronological splits for most arms.

```json
{
  "n_meeting_bar_by_dataset_point": {
    "BGL": 5,
    "HDFS_v1": 0
  },
  "n_meeting_bar_by_dataset_ci": {
    "BGL": 5,
    "HDFS_v1": 0
  },
  "detail": {
    "BGL/pca": {
      "mean_drop": 0.7490540450463794,
      "ci_lo": 0.7390944213181112,
      "ci_hi": 0.7590136687746476,
      "point_ok": true,
      "ci_ok": true
    },
    "BGL/iforest": {
      "mean_drop": 0.0215577044278853,
      "ci_lo": 0.012956754165075388,
      "ci_hi": 0.030372090308714194,
      "point_ok": false,
      "ci_ok": false
    },
    "BGL/ocsvm": {
      "mean_drop": 0.7467141984764354,
      "ci_lo": 0.7370961254908535,
      "ci_hi": 0.7552562530355066,
      "point_ok": true,
      "ci_ok": true
    },
    "BGL/deeplog": {
      "mean_drop": 0.875334545063162,
      "ci_lo": 0.8701226697159011,
      "ci_hi": 0.880685742366644,
      "point_ok": true,
      "ci_ok": true
    },
    "BGL/semantic_knn": {
      "mean_drop": 0.8742384896622456,
      "ci_lo": 0.8679773698616527,
      "ci_hi": 0.8809181882992005,
      "point_ok": true,
      "ci_ok": true
    },
    "BGL/supervised_lgbm": {
      "mean_drop": 0.7423858393808874,
      "ci_lo": 0.741575559020332,
      "ci_hi": 0.7431963065312097,
      "point_ok": true,
      "ci_ok": true
    },
    "HDFS_v1/pca": {
      "mean_drop": -0.32640511787915677,
      "ci_lo": -0.32955377763063076,
      "ci_hi": -0.3233052760278725,
      "point_ok": false,
      "ci_ok": false
    },
    "HDFS_v1/iforest": {
      "mean_drop": -0.08722265965248108,
      "ci_lo": -0.09216919717425821,
      "ci_hi": -0.08118413058096083,
      "point_ok": false,
      "ci_ok": false
    },
    "HDFS_v1/ocsvm": {
      "mean_drop": -0.22789653389272518,
      "ci_lo": -0.2326162238448096,
      "ci_hi": -0.22317684394064088,
      "point_ok": false,
      "ci_ok": false
    },
    "HDFS_v1/deeplog": {
      "mean_drop": 0.014027298415102207,
      "ci_lo": -0.12458534656833216,
      "ci_hi": 0.15228514711008562,
      "point_ok": false,
      "ci_ok": false
    },
    "HDFS_v1/semantic_knn": {
      "mean_drop": 0.012439029553103054,
      "ci_lo": 0.012242221017214195,
      "ci_hi": 0.012564452659602265,
      "point_ok": false,
      "ci_ok": false
    },
    "HDFS_v1/supervised_lgbm": {
      "mean_drop": 0.00011509776298386231,
      "ci_lo": -9.829211219498646e-05,
      "ci_hi": 0.00023595628187758066,
      "point_ok": false,
      "ci_ok": false
    }
  },
  "verdict": "CONFIRMED"
}
```

## H2: CONFIRMED

**Bar:** Kendall tau < 0.8 between RANDOM and CHRONO AUPRC rankings, on >=1 dataset

**Interpretation:** Which method looks best depends on how you split the data.

```json
{
  "detail": {
    "BGL": {
      "tau": 0.2,
      "p": 0.7194444444444444,
      "rank_random": [
        "supervised_lgbm",
        "deeplog",
        "semantic_knn",
        "ocsvm",
        "pca",
        "iforest"
      ],
      "rank_chrono": [
        "supervised_lgbm",
        "ocsvm",
        "iforest",
        "deeplog",
        "pca",
        "semantic_knn"
      ],
      "unstable": true
    },
    "HDFS_v1": {
      "tau": 0.9999999999999999,
      "p": 0.002777777777777778,
      "rank_random": [
        "supervised_lgbm",
        "semantic_knn",
        "ocsvm",
        "pca",
        "deeplog",
        "iforest"
      ],
      "rank_chrono": [
        "supervised_lgbm",
        "semantic_knn",
        "ocsvm",
        "pca",
        "deeplog",
        "iforest"
      ],
      "unstable": false
    }
  },
  "verdict": "CONFIRMED"
}
```

## H3: CONFIRMED

**Bar:** Spearman rho > 0.4, p < 0.05, across (dataset x seed x arm) cells

**Interpretation:** Cells with more unseen templates in the chronological test set show larger AUPRC drops, supporting template drift as a mechanism.

```json
{
  "rho": 0.8276503452591527,
  "p": 3.5321608778168928e-16,
  "n": 60,
  "verdict": "CONFIRMED"
}
```

## H4: NOT CONFIRMED

**Bar:** semantic_knn's RANDOM->CHRONO AUPRC drop smaller than pca's and deeplog's, both datasets, CI of difference excl. 0

**Interpretation:** The semantic arm is not more robust to chronological splitting than PCA/DeepLog on these datasets.

```json
{
  "detail": {
    "BGL": {
      "pca": {
        "mean_diff": -0.1251844446158662,
        "ci_lo": -0.1305832249606645,
        "ci_hi": -0.11978566427106789,
        "point_ok": false,
        "ci_ok": false
      },
      "deeplog": {
        "mean_diff": 0.0010960554009163293,
        "ci_lo": -0.0007757502141432671,
        "ci_hi": 0.00317390244134228,
        "point_ok": true,
        "ci_ok": false
      }
    },
    "HDFS_v1": {
      "pca": {
        "mean_diff": -0.33884414743225977,
        "ci_lo": -0.3420184784330355,
        "ci_hi": -0.33566981643148414,
        "point_ok": false,
        "ci_ok": false
      },
      "deeplog": {
        "mean_diff": 0.0015882688619991536,
        "ci_lo": -0.13702564616750043,
        "ci_hi": 0.14005350839657055,
        "point_ok": true,
        "ci_ok": false
      }
    }
  },
  "verdict": "NOT CONFIRMED"
}
```

## H5: NOT CONFIRMED

**Bar:** mean(F1_oracle - F1_val) >= 0.10 under CHRONO, CI excl. 0, on >=1 dataset

**Interpretation:** Threshold leakage is not large enough, on these datasets, to meet the pre-registered bar.

```json
{
  "detail": {
    "BGL": {
      "mean_gap": 0.007611493489748386,
      "ci_lo": 0.005297191473197143,
      "ci_hi": 0.010158385288033062,
      "point_ok": false,
      "ci_ok": false,
      "n": 30
    },
    "HDFS_v1": {
      "mean_gap": 0.06057674083277558,
      "ci_lo": 0.016875627220609014,
      "ci_hi": 0.10822787842853411,
      "point_ok": false,
      "ci_ok": false,
      "n": 30
    }
  },
  "verdict": "NOT CONFIRMED"
}
```
