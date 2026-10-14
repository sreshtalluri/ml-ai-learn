# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/14-naive-bayes and phases/01-math-foundations/07-bayes-theorem
"""Naive Bayes from scratch (NumPy): Bayes' rule, multinomial NB in log space, Gaussian NB, checked against scikit-learn.

Run from guide/:  uv run code/05-instance-and-probabilistic/nb_scratch.py
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB


# --- Chunk 1: Bayes' rule with one piece of evidence ---------------------------
def posterior(prior, p_e_given_h, p_e_given_not_h):
    evidence = p_e_given_h * prior + p_e_given_not_h * (1 - prior)   # P(e), total probability
    return p_e_given_h * prior / evidence


p1 = posterior(0.0001, 0.99, 0.01)   # rare disease, 99% sensitive test, 1% false positives
p2 = posterior(p1, 0.99, 0.01)       # second positive test: yesterday's posterior is today's prior
print(f"P(sick | 1 positive) = {p1:.4f}   P(sick | 2 positives) = {p2:.4f}")


# --- Chunk 2: fit multinomial NB = count words per class, smooth, take logs ----
def fit_mnb(X_counts, y, alpha=1.0):
    classes = np.unique(y)
    log_prior = np.log([np.mean(y == c) for c in classes])                          # [C]
    counts = np.array([X_counts[y == c].sum(axis=0) for c in classes]) + alpha      # [C, V]
    log_like = np.log(counts / counts.sum(axis=1, keepdims=True))                   # rows sum to 1
    return classes, log_prior, log_like


# --- Chunk 3: predict = sum of logs, then normalize with log-sum-exp ------------
def predict_proba_mnb(model, X_counts):
    classes, log_prior, log_like = model
    scores = X_counts @ log_like.T + log_prior               # [n, C] log P(c) + sum_w count * log P(w|c)
    scores -= scores.max(axis=1, keepdims=True)              # shift so exp() can't underflow to 0/0
    probs = np.exp(scores)
    return probs / probs.sum(axis=1, keepdims=True)


docs = ["win money now", "win a free prize", "free money offer",
        "meeting at noon", "lunch meeting today", "project update today"]
labels = np.array([1, 1, 1, 0, 0, 0])                        # 1 = spam
vec = CountVectorizer(token_pattern=r"\b\w+\b")
Xc = vec.fit_transform(docs).toarray()
model = fit_mnb(Xc, labels)
xq = vec.transform(["free money today"]).toarray()
ours = predict_proba_mnb(model, xq)[0, 1]
lib = MultinomialNB(alpha=1.0).fit(Xc, labels).predict_proba(xq)[0, 1]
print(f"\nP(spam | 'free money today'): scratch {ours:.4f}   scikit-learn {lib:.4f}")

# Why logs: 1000 likelihoods of ~0.01 multiply to 1e-2000, below float64's smallest number.
print(f"product of 1000 x 0.01 = {np.prod(np.full(1000, 0.01))}   sum of logs = {np.log(0.01) * 1000:.1f}")


# --- Chunk 4: Gaussian NB = one normal distribution per class and feature -------
def fit_gnb(X, y, eps=1e-9):
    classes = np.unique(y)
    mean = np.array([X[y == c].mean(axis=0) for c in classes])                     # [C, p]
    var = np.array([X[y == c].var(axis=0) for c in classes]) + eps * X.var(axis=0).max()
    log_prior = np.log([np.mean(y == c) for c in classes])
    return classes, mean, var, log_prior


def predict_gnb(model, X):
    classes, mean, var, log_prior = model
    # log N(x; mean, var) summed over features, for every class: [n, C]
    ll = -0.5 * (np.log(2 * np.pi * var)[None] + (X[:, None, :] - mean[None]) ** 2 / var[None]).sum(axis=2)
    return classes[np.argmax(ll + log_prior, axis=1)]


# SYNTHETIC: two classes, two continuous features
rng = np.random.default_rng(0)
Xg = np.r_[rng.normal([0, 0], [1.0, 2.0], (150, 2)), rng.normal([2, 1], [1.0, 1.0], (150, 2))]
yg = np.r_[np.zeros(150, int), np.ones(150, int)]
idx = rng.permutation(300)
tr, te = idx[:240], idx[240:]
ours_g = predict_gnb(fit_gnb(Xg[tr], yg[tr]), Xg[te])
lib_g = GaussianNB().fit(Xg[tr], yg[tr]).predict(Xg[te])
print(f"\nGaussian NB on 60 test rows: scratch acc {np.mean(ours_g == yg[te]):.3f}, "
      f"agrees with scikit-learn on {np.mean(ours_g == lib_g):.0%}")

assert abs(ours - lib) < 1e-9 and abs(ours - 0.7253) < 1e-4
