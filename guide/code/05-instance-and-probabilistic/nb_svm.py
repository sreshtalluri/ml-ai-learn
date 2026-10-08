"""Naive Bayes worked example and SVM margins/kernels on synthetic data.

Run from guide/:  uv run code/05-instance-and-probabilistic/nb_svm.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, plt, save  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Naive Bayes by hand: tiny spam corpus
# ---------------------------------------------------------------------------
docs = ["win money now", "win a free prize", "free money offer", "meeting at noon", "lunch meeting today", "project update today"]
labels = np.array([1, 1, 1, 0, 0, 0])   # 1 = spam
vec = CountVectorizer(token_pattern=r"\b\w+\b")
Xc = vec.fit_transform(docs).toarray()
vocab = vec.get_feature_names_out()
V = len(vocab)
query = ["free", "money", "today"]
for c, name in [(1, "spam"), (0, "ham")]:
    counts = Xc[labels == c].sum(axis=0)
    total = counts.sum()
    prior = (labels == c).mean()
    like = [(counts[list(vocab).index(w)] + 1) / (total + V) for w in query]
    print(f"{name}: prior={prior}, total words={total}, V={V}, P(w|c) with +1 smoothing = "
          + ", ".join(f"{w}:{lk:.4f}" for w, lk in zip(query, like))
          + f", score = {prior * np.prod(like):.3e}")
nb = MultinomialNB(alpha=1.0).fit(Xc, labels)
p = nb.predict_proba(vec.transform([" ".join(query)]).toarray())[0]
print(f"scikit-learn MultinomialNB P(spam | 'free money today') = {p[1]:.4f}")

# ---------------------------------------------------------------------------
# 2. SVM figure: margin with different C, and an RBF kernel on rings
# ---------------------------------------------------------------------------
X, y = make_blobs(n_samples=80, centers=[[-1.5, 0], [1.5, 0.5]], cluster_std=1.0, random_state=6)
Xr, yr = make_circles(n_samples=200, factor=0.4, noise=0.08, random_state=1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, C in [(axes[0], 100), (axes[1], 0.05)]:
    m = SVC(kernel="linear", C=C).fit(X, y)
    gx, gy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-3.5, 4, 200))
    d = m.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contour(gx, gy, d, levels=[-1, 0, 1], colors=[GRAY, "black", GRAY], linestyles=["--", "-", "--"])
    ax.scatter(*X[y == 0].T, s=16, color=BLUE)
    ax.scatter(*X[y == 1].T, s=16, color=ORANGE, marker="s")
    sv = m.support_vectors_
    ax.scatter(*sv.T, s=90, facecolors="none", edgecolors="black", linewidths=1)
    width = 2 / np.linalg.norm(m.coef_)
    ax.set(title=f"Linear SVM, C = {C}: margin {width:.2f}, {len(sv)} support vectors", xlabel="x₁")
    ax.title.set_fontsize(9)
    print(f"linear SVM C={C}: margin width {width:.3f}, support vectors {len(sv)}, train acc {m.score(X, y):.3f}")
axes[0].set_ylabel("x₂")
ax = axes[2]
lin = SVC(kernel="linear").fit(Xr, yr)
rbf = SVC(kernel="rbf", gamma=1.0, C=1.0).fit(Xr, yr)
gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
d = rbf.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
ax.contourf(gx, gy, d > 0, levels=[-0.5, 0.5, 1.5], colors=[BLUE, ORANGE], alpha=0.12)
ax.contour(gx, gy, d, levels=[0], colors="black")
ax.scatter(*Xr[yr == 0].T, s=10, color=BLUE)
ax.scatter(*Xr[yr == 1].T, s=10, color=ORANGE, marker="s")
ax.set(title=f"RBF kernel: acc {rbf.score(Xr, yr):.2f} (linear kernel: {lin.score(Xr, yr):.2f})", xlabel="x₁")
ax.title.set_fontsize(9)
print(f"rings: linear acc {lin.score(Xr, yr):.3f}, rbf acc {rbf.score(Xr, yr):.3f}")
save(fig, "support-vector-machines")

# Naive Bayes figure: word likelihoods per class
fig, ax = plt.subplots(figsize=(8, 3.6))
idx = np.arange(V)
spam_like = (Xc[labels == 1].sum(0) + 1) / (Xc[labels == 1].sum() + V)
ham_like = (Xc[labels == 0].sum(0) + 1) / (Xc[labels == 0].sum() + V)
ax.bar(idx - 0.2, spam_like, width=0.4, color=ORANGE, label="P(word | spam)")
ax.bar(idx + 0.2, ham_like, width=0.4, color=BLUE, label="P(word | ham)")
ax.set_xticks(idx, vocab, rotation=45, ha="right")
ax.set(title="Naive Bayes likelihoods with Laplace smoothing (6-email toy corpus)", ylabel="probability")
ax.title.set_fontsize(10)
ax.legend(frameon=False, fontsize=8)
save(fig, "naive-bayes")
