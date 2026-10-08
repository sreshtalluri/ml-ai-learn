"""Supervised vs unsupervised vs self-supervised on the same synthetic data, plus a semi-supervised baseline.

Run from guide/:  uv run code/01-ml-vocabulary/paradigms.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402

X, y = make_blobs(n_samples=200, centers=[[-2, 0], [2, 1]], cluster_std=1.1, random_state=4)  # SYNTHETIC

# Supervised: we have every label
sup = LogisticRegression().fit(X, y)
print(f"Supervised (200 labels): training accuracy {sup.score(X, y):.3f}")

# Unsupervised: no labels; K-means finds 2 groups (cluster ids are arbitrary)
km = KMeans(2, n_init=10, random_state=0).fit(X)
agree = max(np.mean(km.labels_ == y), np.mean(km.labels_ != y))
print(f"Unsupervised (0 labels): K-means groups match the hidden labels {agree:.3f} of the time")

# Semi-supervised: only 6 labels, pseudo-label the rest
rng = np.random.default_rng(0)
y_partial = np.full_like(y, -1)
idx = np.concatenate([rng.choice(np.where(y == c)[0], 3, replace=False) for c in (0, 1)])
y_partial[idx] = y[idx]
few = LogisticRegression().fit(X[idx], y[idx])
semi = SelfTrainingClassifier(LogisticRegression(), threshold=0.9).fit(X, y_partial)
print(f"Only 6 labels, supervised on those 6: accuracy {few.score(X, y):.3f}")
print(f"Only 6 labels, self-training on all 200: accuracy {semi.score(X, y):.3f}")

# Self-supervised: labels created from the data itself (next-token pairs)
text = "the model predicts the next token from the previous tokens".split()
pairs = [(" ".join(text[:i]), text[i]) for i in range(1, 5)]
print("\nSelf-supervised training pairs made from one sentence:")
for ctx, nxt in pairs:
    print(f"  input: {ctx!r:40s} target: {nxt!r}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
gx, gy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-4, 5, 200))
ax = axes[0]
ax.contourf(gx, gy, sup.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape), levels=[-0.5, 0.5, 1.5], colors=[BLUE, ORANGE], alpha=0.12)
ax.scatter(*X[y == 0].T, s=12, color=BLUE, label="label 0")
ax.scatter(*X[y == 1].T, s=12, color=ORANGE, marker="s", label="label 1")
ax.set(title="Supervised: learn X → y from labels", xlabel="x₁", ylabel="x₂")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
for c, col in zip((0, 1), (TEAL, "#7c3aed")):
    ax.scatter(*X[km.labels_ == c].T, s=12, color=col, label=f"cluster {c}")
ax.scatter(*km.cluster_centers_.T, marker="x", s=120, color="black", linewidths=2.5)
ax.set(title="Unsupervised: find structure, no labels", xlabel="x₁")
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
ax.scatter(*X.T, s=10, color=GRAY, alpha=0.5, label="unlabeled")
ax.scatter(*X[idx][y[idx] == 0].T, s=60, color=BLUE, edgecolor="black", label="labeled 0")
ax.scatter(*X[idx][y[idx] == 1].T, s=60, color=ORANGE, marker="s", edgecolor="black", label="labeled 1")
ax.contour(gx, gy, semi.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape), levels=[0.5], colors="black", linewidths=1)
ax.set(title="Semi-supervised: 6 labels + 194 unlabeled", xlabel="x₁")
ax.legend(frameon=False, fontsize=8, loc="lower right")
save(fig, "learning-paradigms")
