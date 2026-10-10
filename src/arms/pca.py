"""PCA reconstruction-error anomaly score (Xu et al. 2009)."""
import numpy as np
from sklearn.decomposition import PCA

from features import count_matrix


class PCAArm:
    name = "pca"

    def fit(self, ctx, train_sessions):
        X = count_matrix(train_sessions, ctx.n_vocab).toarray().astype("float32")
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        max_components = max(1, min(Xc.shape) - 1)
        probe = PCA(n_components=min(max_components, 200), random_state=0)
        probe.fit(Xc)
        cum = np.cumsum(probe.explained_variance_ratio_)
        k = int(np.searchsorted(cum, 0.95) + 1)
        k = max(1, min(k, probe.n_components_))

        self.pca_ = PCA(n_components=k, random_state=0)
        self.pca_.fit(Xc)
        return self

    def score(self, ctx, sessions):
        X = count_matrix(sessions, ctx.n_vocab).toarray().astype("float32")
        Xc = X - self.mean_
        proj = self.pca_.inverse_transform(self.pca_.transform(Xc))
        return np.sum((Xc - proj) ** 2, axis=1)
