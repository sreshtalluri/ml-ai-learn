"""K-means: worked example (one full iteration by hand), from scratch, scikit-learn, elbow/silhouette, figure.

Run from guide/:  uv run code/07-unsupervised/kmeans.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PALETTE, TEAL, plt, save  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Worked example: six points, K = 2
# ---------------------------------------------------------------------------
names = list("ABCDEF")
X = np.array([[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5]], dtype=float)
mu = np.array([[1, 1], [4.5, 5]], dtype=float)   # initialize at A and F


def assign(X, mu):
    d2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)  # [n, K] squared distances
    return d2, d2.argmin(axis=1)


def inertia(X, mu, a):
    return float(((X - mu[a]) ** 2).sum())


for it in range(1, 4):
    d2, a = assign(X, mu)
    print(f"Iteration {it}: centroids {mu.tolist()}")
    for n, row, k in zip(names, d2, a):
        print(f"  {n} d²={row.round(2).tolist()} -> cluster {k + 1}")
    print(f"  inertia after assignment: {inertia(X, mu, a):.3f}")
    new_mu = np.array([X[a == k].mean(axis=0) for k in range(len(mu))])
    print(f"  update -> {new_mu.round(4).tolist()}  inertia {inertia(X, new_mu, a):.3f}")
    if np.allclose(new_mu, mu):
        print("  converged")
        break
    mu = new_mu


# ---------------------------------------------------------------------------
# 2. From scratch vs scikit-learn on SYNTHETIC blobs
# ---------------------------------------------------------------------------
def kmeans(X, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    mu = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        _, a = assign(X, mu)
        new = np.array([X[a == j].mean(axis=0) if np.any(a == j) else mu[j] for j in range(k)])
        if np.allclose(new, mu):
            break
        mu = new
    return mu, a


Xb, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=7)
mu_s, a_s = kmeans(Xb, 4, seed=1)
runs = [inertia(Xb, *kmeans(Xb, 4, seed=s)) for s in range(10)]
sk = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Xb)
print(f"\nfrom scratch, one random start (seed 1): inertia={inertia(Xb, mu_s, a_s):.1f}  <- stuck in a local optimum")
print(f"from scratch, best of 10 random starts:  inertia={min(runs):.1f}")
print(f"scikit-learn (k-means++, n_init=10):      inertia={sk.inertia_:.1f}")

ks = range(1, 9)
inertias = [KMeans(k, n_init=10, random_state=0).fit(Xb).inertia_ for k in ks]
sils = [np.nan] + [silhouette_score(Xb, KMeans(k, n_init=10, random_state=0).fit_predict(Xb)) for k in ks[1:]]
print("K, inertia, silhouette:", [(k, round(i, 1), None if np.isnan(s) else round(s, 3)) for k, i, s in zip(ks, inertias, sils)])

# ---------------------------------------------------------------------------
# 3. Figure: clustering result, elbow + silhouette, and a failure case
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ax = axes[0]
for k in range(4):
    ax.scatter(*Xb[sk.labels_ == k].T, s=10, color=PALETTE[k])
ax.scatter(*sk.cluster_centers_.T, marker="x", s=120, color="black", linewidths=2.5)
ax.set(title="K = 4 on synthetic blobs", xlabel="feature 1", ylabel="feature 2")

ax = axes[1]
ax.plot(list(ks), inertias, "-o", color=ORANGE, label="inertia (left)")
ax.set(title="Choosing K: elbow and silhouette", xlabel="K", ylabel="inertia")
ax2 = ax.twinx()
ax2.plot(list(ks), sils, "-s", color=TEAL, label="silhouette (right)")
ax2.set_ylabel("silhouette")
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax.legend(handles=ax.lines + ax2.lines, frameon=False, fontsize=8, loc="center right")

Xc, _ = make_circles(n_samples=300, factor=0.35, noise=0.05, random_state=3)
lab = KMeans(2, n_init=10, random_state=0).fit_predict(Xc)
ax = axes[2]
ax.scatter(*Xc[lab == 0].T, s=10, color=BLUE)
ax.scatter(*Xc[lab == 1].T, s=10, color=ORANGE)
ax.set(title="Failure: rings get cut in half", xlabel="feature 1")
fig.subplots_adjust(wspace=0.45)
save(fig, "k-means")
