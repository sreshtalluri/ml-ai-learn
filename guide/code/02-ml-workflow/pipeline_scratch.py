# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/13-ml-pipelines
"""A preprocessing pipeline from scratch (imputer, scaler, feature selector, model), k-fold CV, and
the leakage you get when a step is fit outside the folds. All data is SYNTHETIC; the leakage demo
uses PURE NOISE labels, so any accuracy above 0.5 is fake.

Run from guide/:  uv run code/02-ml-workflow/pipeline_scratch.py
"""
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MedianImputer:
    def fit(self, X, y=None):
        self.medians_ = np.nanmedian(X, axis=0)           # learned from TRAINING rows only
        return self

    def transform(self, X):
        return np.where(np.isnan(X), self.medians_, X)


class Standardizer:
    def fit(self, X, y=None):
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


class TopKByCorrelation:
    """Keep the k features most correlated with y. It looks at y, so it can leak."""
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        Xc, yc = X - X.mean(0), y - y.mean()
        corr = np.abs(Xc.T @ yc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc) + 1e-12)
        self.keep_ = np.argsort(-corr)[: self.k]
        return self

    def transform(self, X):
        return X[:, self.keep_]


class Pipeline:
    def __init__(self, *steps):
        self.transforms, self.model = steps[:-1], steps[-1]

    def fit(self, X, y):
        for t in self.transforms:
            X = t.fit(X, y).transform(X)                   # fit, then pass the transformed data on
        self.model.fit(X, y)
        return self

    def predict(self, X):
        for t in self.transforms:
            X = t.transform(X)                             # transform only: reuse training statistics
        return self.model.predict(X)


def kfold_accuracy(make_model, X, y, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    folds = np.array_split(idx, k)
    scores = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        model = make_model().fit(X[tr], y[tr])
        scores.append(np.mean(model.predict(X[te]) == y[te]))
    return float(np.mean(scores))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Real signal + missing values
    X = rng.normal(size=(500, 4)) * [1, 10, 100, 1] + [0, 50, 500, 0]
    y = (X[:, 0] + (X[:, 1] - 50) / 10 + rng.normal(0, 1, 500) > 0).astype(int)
    X[rng.random(X.shape) < 0.1] = np.nan
    print(f"missing values: {int(np.isnan(X).sum())} of {X.size}")
    acc = kfold_accuracy(lambda: Pipeline(MedianImputer(), Standardizer(), LogisticRegression()), X, y)
    print(f"scratch pipeline 5-fold accuracy: {acc:.3f}")

    # Leakage: 100 examples, 5,000 noise features, labels are coin flips
    Xn = rng.normal(size=(100, 5000))
    yn = rng.integers(0, 2, 100)
    keep = TopKByCorrelation(20).fit(Xn, yn).keep_         # WRONG: selection saw every label
    leaky = kfold_accuracy(LogisticRegression, Xn[:, keep], yn)
    honest = kfold_accuracy(lambda: Pipeline(TopKByCorrelation(20), LogisticRegression()), Xn, yn)
    print(f"noise labels: selection outside CV {leaky:.3f}, selection inside CV {honest:.3f}")

    sk = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
    print(f"sklearn Pipeline, selection inside CV: {cross_val_score(sk, Xn, yn, cv=5).mean():.3f}")
    sk_full = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
    print(f"sklearn Pipeline on the real-signal data: {cross_val_score(sk_full, X, y, cv=5).mean():.3f}")

    assert leaky > 0.75 and honest < 0.65
    print("checks pass: leakage inflates noise to look like signal")
