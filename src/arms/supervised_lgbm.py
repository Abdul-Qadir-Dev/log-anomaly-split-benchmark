"""Supervised LightGBM ceiling reference (uses true labels at train time)."""
import lightgbm as lgb

from features import count_matrix


class SupervisedLGBMArm:
    name = "supervised_lgbm"

    def fit(self, ctx, train_sessions):
        X = count_matrix(train_sessions, ctx.n_vocab)
        y = train_sessions["label"].to_numpy()
        self.model_ = lgb.LGBMClassifier(n_estimators=200, random_state=0, verbosity=-1)
        self.model_.fit(X, y)
        return self

    def score(self, ctx, sessions):
        X = count_matrix(sessions, ctx.n_vocab)
        return self.model_.predict_proba(X)[:, 1]
