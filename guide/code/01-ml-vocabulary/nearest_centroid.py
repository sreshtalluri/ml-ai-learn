# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/01-what-is-machine-learning
"""The smallest complete learner: a nearest-centroid classifier with fit / predict / score, then the
same vocabulary (parameters, hyperparameters, loss, metric, train vs test) on scikit-learn.
Data is SYNTHETIC (two Gaussian blobs).

Run from guide/:  uv run code/01-ml-vocabulary/nearest_centroid.py
"""
import numpy as np
from sklearn.neighbors import NearestCentroid as SkNearestCentroid


class NearestCentroid:
    def fit(self, X, y):
        """Training: learn one centroid (mean feature vector) per class. These are the parameters."""
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])  # [k, d]
        return self

    def predict(self, X):
        """Inference: assign each row to the class whose centroid is closest."""
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)  # [n, k] squared distances
        return self.classes_[d.argmin(axis=1)]

    def score(self, X, y):
        """Metric: accuracy, the share of rows predicted correctly."""
        return float(np.mean(self.predict(X) == y))


def make_blobs(n_per_class, separation, rng):
    X0 = rng.normal(size=(n_per_class, 2)) + separation / 2
    X1 = rng.normal(size=(n_per_class, 2)) - separation / 2
    return np.vstack([X0, X1]), np.array([0] * n_per_class + [1] * n_per_class)


def train_test_split(X, y, test_fraction, rng):
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X, y = make_blobs(100, separation=2.0, rng=rng)                  # 200 examples, 2 features
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, 0.3, rng)
    print(f"X {X.shape}, y {y.shape}; train {X_tr.shape[0]}, test {X_te.shape[0]}")

    model = NearestCentroid().fit(X_tr, y_tr)
    print(f"parameters (centroids):\n{model.centroids_.round(3)}")
    print(f"train accuracy {model.score(X_tr, y_tr):.3f}, test accuracy {model.score(X_te, y_te):.3f}")

    for sep in (0.5, 2.0, 4.0):                                      # the data, not the model, sets the ceiling
        Xs, ys = make_blobs(100, sep, np.random.default_rng(0))
        a, b, c, d = train_test_split(Xs, ys, 0.3, np.random.default_rng(1))
        print(f"separation {sep}: test accuracy {NearestCentroid().fit(a, c).score(b, d):.3f}")

    sk = SkNearestCentroid().fit(X_tr, y_tr)
    assert np.allclose(sk.centroids_, model.centroids_)
    assert (sk.predict(X_te) == model.predict(X_te)).all()
    print("matches sklearn.neighbors.NearestCentroid")
