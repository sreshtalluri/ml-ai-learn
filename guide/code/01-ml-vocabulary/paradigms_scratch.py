"""One dataset, four feedback signals, all in NumPy: supervised (every label), unsupervised (no labels),
semi-supervised (6 labels + self-training), self-supervised (labels made from the data itself).
Then the scikit-learn equivalents. Data is SYNTHETIC.

Run from guide/:  uv run code/01-ml-vocabulary/paradigms_scratch.py
"""
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier


def fit_logistic(X, y, steps=2000, lr=0.1):
    """Supervised: gradient descent on log loss. X is [n, 2], y is [n] of 0/1."""
    Xb = np.c_[X, np.ones(len(X))]                # add a bias column -> [n, 3]
    w = np.zeros(3)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))             # predicted P(y = 1)
        w -= lr * Xb.T @ (p - y) / len(y)         # gradient of mean log loss
    return w


def predict_proba(w, X):
    return 1 / (1 + np.exp(-np.c_[X, np.ones(len(X))] @ w))


def kmeans(X, k, rng, iters=50):
    """Unsupervised: alternate 'assign each point to nearest centre' and 'move centre to its points' mean'."""
    centres = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)   # [n, k] squared distances
        assign = d.argmin(axis=1)
        centres = np.array([X[assign == c].mean(axis=0) for c in range(k)])
    return assign, centres


def self_train(X, y_partial, threshold=0.9, rounds=10):
    """Semi-supervised: fit on labeled rows, pseudo-label confident unlabeled rows, refit. -1 = unlabeled."""
    y = y_partial.copy()
    for _ in range(rounds):
        known = y != -1
        w = fit_logistic(X[known], y[known])
        p = predict_proba(w, X)
        confident = (~known) & ((p > threshold) | (p < 1 - threshold))
        if not confident.any():
            break
        y[confident] = (p[confident] > 0.5).astype(int)
    return w


def next_word_pairs(text):
    """Self-supervised: every position becomes a (previous word, next word) example. No human labels."""
    words = text.split()
    return list(zip(words[:-1], words[1:]))


def fit_bigram(pairs):
    counts = defaultdict(Counter)
    for prev, nxt in pairs:
        counts[prev][nxt] += 1
    return {prev: c.most_common(1)[0][0] for prev, c in counts.items()}


if __name__ == "__main__":
    # SYNTHETIC: two Gaussian blobs, 100 points each, centres (-2, 0) and (2, 1), std 1.1.
    rng = np.random.default_rng(4)
    X = np.r_[rng.normal([-2, 0], 1.1, (100, 2)), rng.normal([2, 1], 1.1, (100, 2))]
    y = np.r_[np.zeros(100, int), np.ones(100, int)]

    w = fit_logistic(X, y)
    acc_sup = np.mean((predict_proba(w, X) > 0.5) == y)
    print(f"Supervised, 200 labels:            accuracy {acc_sup:.3f}")

    assign, _ = kmeans(X, 2, np.random.default_rng(0))
    agree = max(np.mean(assign == y), np.mean(assign != y))   # cluster ids are arbitrary
    print(f"Unsupervised, 0 labels:            clusters match hidden labels {agree:.3f}")

    idx = np.r_[rng.choice(100, 3, replace=False), 100 + rng.choice(100, 3, replace=False)]
    y_partial = np.full(200, -1)
    y_partial[idx] = y[idx]
    w_few = fit_logistic(X[idx], y[idx])
    w_semi = self_train(X, y_partial)
    print(f"6 labels, supervised on those 6:   accuracy {np.mean((predict_proba(w_few, X) > 0.5) == y):.3f}")
    print(f"6 labels, self-training on 200:    accuracy {np.mean((predict_proba(w_semi, X) > 0.5) == y):.3f}")

    text = ("the cat sat on the mat . the dog sat on the rug . the cat ate the fish . "
            "the dog ate the bone . the cat sat on the rug .")
    pairs = next_word_pairs(text)
    model = fit_bigram(pairs)
    print(f"\nSelf-supervised: {len(text.split())} words -> {len(pairs)} (input, target) pairs, e.g. {pairs[:3]}")
    print(f"Learned next word: sat -> {model['sat']!r}, on -> {model['on']!r}, cat -> {model['cat']!r}")

    print("\nscikit-learn, same data:")
    print(f"  LogisticRegression, 200 labels:  accuracy {LogisticRegression().fit(X, y).score(X, y):.3f}")
    km = KMeans(2, n_init=10, random_state=0).fit(X)
    print(f"  KMeans, 0 labels:                match {max(np.mean(km.labels_ == y), np.mean(km.labels_ != y)):.3f}")
    semi = SelfTrainingClassifier(LogisticRegression(), threshold=0.9).fit(X, y_partial)
    print(f"  SelfTrainingClassifier, 6 labels: accuracy {semi.score(X, y):.3f}")

    assert len(pairs) == len(text.split()) - 1 and model["sat"] == "on"
