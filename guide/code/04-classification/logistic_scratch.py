# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/03-logistic-regression
"""Binary logistic regression and softmax regression from scratch (NumPy), checked against
scikit-learn. Data is SYNTHETIC.

Run from guide/:  uv run code/04-classification/logistic_scratch.py
"""
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression


def sigmoid(z):
    # Stable for large |z|: never calls exp on a large positive number
    e = np.exp(-np.abs(z))                                 # always in (0, 1]
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)                           # log(0) guard
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic(X, y, lr=0.5, steps=3000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)                              # [n] probabilities
        w -= lr * X.T @ (p - y) / n                         # gradient of mean BCE: (p - y) x
        b -= lr * np.mean(p - y)
    return w, b


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)                   # subtract row max: same result, no overflow
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_softmax(X, y, k, lr=0.5, steps=3000):
    n, d = X.shape
    W, b = np.zeros((d, k)), np.zeros(k)
    Y = np.eye(k)[y]                                        # one-hot, [n, k]
    for _ in range(steps):
        P = softmax(X @ W + b)                              # [n, k]
        W -= lr * X.T @ (P - Y) / n                         # same (p - y) x shape as the binary case
        b -= lr * (P - Y).mean(axis=0)
    return W, b


if __name__ == "__main__":
    print(f"sigmoid(-1000), sigmoid(0), sigmoid(1000) = {sigmoid(np.array([-1000.0, 0.0, 1000.0]))}")

    X, y = make_classification(n_samples=500, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    w, b = fit_logistic(X, y)
    p = sigmoid(X @ w + b)
    print(f"binary: BCE {bce(y, p):.4f}, accuracy {np.mean((p >= 0.5) == y):.3f}, w = {w.round(3)}")

    sk = LogisticRegression(C=np.inf, max_iter=5000).fit(X, y)   # C=inf: no penalty, to match ours
    print(f"sklearn (no penalty): w = {sk.coef_[0].round(3)}")

    Xm, ym = make_blobs(n_samples=600, centers=3, cluster_std=1.5, random_state=1)
    Xm = (Xm - Xm.mean(0)) / Xm.std(0)
    W, bm = fit_softmax(Xm, ym, k=3)
    P = softmax(Xm @ W + bm)
    print(f"softmax: rows sum to 1: {np.allclose(P.sum(1), 1)}, accuracy {np.mean(P.argmax(1) == ym):.3f}")
    skm = LogisticRegression(C=np.inf, max_iter=5000).fit(Xm, ym)
    print(f"sklearn multinomial accuracy {skm.score(Xm, ym):.3f}")

    assert np.allclose(p, sk.predict_proba(X)[:, 1], atol=0.02)
    assert np.mean(P.argmax(1) == skm.predict(Xm)) > 0.98
    print("checks pass: scratch probabilities match scikit-learn")
