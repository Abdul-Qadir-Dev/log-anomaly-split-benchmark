"""One-Class SVM (RBF) on TF-IDF template vectors."""
from sklearn.svm import OneClassSVM

from features import count_matrix, tfidf_matrix

MAX_TRAIN = 50_000


class OCSVMArm:
    name = "ocsvm"

    def fit(self, ctx, train_sessions):
        if len(train_sessions) > MAX_TRAIN:
            train_sessions = train_sessions.sample(n=MAX_TRAIN, random_state=0)
        counts = count_matrix(train_sessions, ctx.n_vocab)
        X, self.tfidf_ = tfidf_matrix(counts)
        self.model_ = OneClassSVM(kernel="rbf")
        self.model_.fit(X)
        return self

    def score(self, ctx, sessions):
        counts = count_matrix(sessions, ctx.n_vocab)
        X, _ = tfidf_matrix(counts, self.tfidf_)
        return -self.model_.decision_function(X)
