"""Isolation Forest on template count vectors."""
from sklearn.ensemble import IsolationForest

from features import count_matrix


class IForestArm:
    name = "iforest"

    def fit(self, ctx, train_sessions):
        X = count_matrix(train_sessions, ctx.n_vocab)
        self.model_ = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1)
        self.model_.fit(X)
        return self

    def score(self, ctx, sessions):
        X = count_matrix(sessions, ctx.n_vocab)
        return -self.model_.score_samples(X)
