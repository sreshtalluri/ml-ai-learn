"""DBSCAN and Gaussian mixtures (EM) from scratch, checked against scikit-learn, plus the
clustering-methods figure (K-means vs DBSCAN, dendrogram, GMM soft membership).

Run from guide/:  uv run code/07-unsupervised/clustering.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import GRAY, ORANGE, PALETTE, plt, save  # noqa: E402

rng = np.random.default_rng(0)
np.set_printoptions(suppress=True)


# ---------------------------------------------------------------------------
# 1. From scratch (NumPy only)
# ---------------------------------------------------------------------------
def dbscan(X, eps, min_samples):
    """Label each point with a cluster id, or -1 for noise. Also return the core-point mask."""
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))   # [n, n] pairwise distances
    nbrs = [np.flatnonzero(row <= eps) for row in D]                 # neighbors, including itself
    core = np.array([len(nb) >= min_samples for nb in nbrs])
    labels = np.full(len(X), -1)
    cluster = 0
    for i in np.flatnonzero(core):
        if labels[i] != -1:                                          # already reached from another core point
            continue
        labels[i], stack = cluster, [i]
        while stack:                                                 # flood-fill through core points
            j = stack.pop()
            for q in nbrs[j]:
                if labels[q] == -1:
                    labels[q] = cluster                              # core or border point joins
                    if core[q]:
                        stack.append(q)                              # only core points keep expanding
        cluster += 1
    return labels, core


def gauss_pdf(X, mean, cov):
    """Multivariate normal density N(x | mean, cov) for every row of X."""
    d = X - mean                                                     # [n, p]
    q = np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)           # squared Mahalanobis distance
    return np.exp(-0.5 * q) / np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(cov))


def gmm_em(X, k, iters=200, seed=0, tol=1e-8):
    """Fit a full-covariance Gaussian mixture with EM. Returns weights, means, covariances, responsibilities."""
    n, p = X.shape
    r = np.random.default_rng(seed)
    pi, mu = np.full(k, 1 / k), X[r.choice(n, k, replace=False)]    # equal weights, random points as means
    S = np.array([np.cov(X.T) for _ in range(k)])                    # start every component at the data covariance
    prev = -np.inf
    for _ in range(iters):
        dens = np.stack([pi[j] * gauss_pdf(X, mu[j], S[j]) for j in range(k)], axis=1)   # [n, k]
        ll = np.log(dens.sum(axis=1)).sum()                          # log-likelihood never decreases
        R = dens / dens.sum(axis=1, keepdims=True)                   # E-step: responsibilities, rows sum to 1
        Nk = R.sum(axis=0)                                           # M-step: effective points per component
        pi, mu = Nk / n, (R.T @ X) / Nk[:, None]
        S = np.array([(R[:, j, None] * (X - mu[j])).T @ (X - mu[j]) / Nk[j] + 1e-6 * np.eye(p)
                      for j in range(k)])                            # small ridge stops collapse onto one point
        if ll - prev < tol:
            break
        prev = ll
    return pi, mu, S, R, ll


# ---------------------------------------------------------------------------
# 2. DBSCAN worked example on a line
# ---------------------------------------------------------------------------
pts = np.array([[1.0], [1.5], [2.0], [2.4], [6.0], [6.3], [6.8], [12.0]])
lab_s, core_s = dbscan(pts, eps=0.6, min_samples=3)
db = DBSCAN(eps=0.6, min_samples=3).fit(pts)
print("DBSCAN eps=0.6 min_samples=3")
print(f"  from scratch labels: {lab_s.tolist()}  core: {np.flatnonzero(core_s).tolist()}")
print(f"  scikit-learn labels: {db.labels_.tolist()}  core: {sorted(db.core_sample_indices_.tolist())}")

# ---------------------------------------------------------------------------
# 3. GMM soft membership for one point between two clusters (SYNTHETIC)
# ---------------------------------------------------------------------------
Xg, _ = make_blobs(n_samples=400, centers=[[0, 0], [4, 0]], cluster_std=[1.0, 1.5], random_state=3)
probe = np.array([[2.2, 0.0]])
pi_s, mu_s, S_s, _, ll_s = gmm_em(Xg, 2)
o = np.argsort(mu_s[:, 0])                                           # order components left, right
dens = np.array([pi_s[j] * gauss_pdf(probe, mu_s[j], S_s[j])[0] for j in o])
gmm = GaussianMixture(2, tol=1e-9, max_iter=500, random_state=0).fit(Xg)   # default tol=1e-3 stops after 3 iterations here
order = np.argsort(gmm.means_[:, 0])
print("GMM on two synthetic blobs (tight at x=0, wide at x=4)")
print(f"  from scratch: weights {pi_s[o].round(3)}, means x {mu_s[o, 0].round(3)}, log-likelihood {ll_s:.2f}")
print(f"  scikit-learn: weights {gmm.weights_[order].round(3)}, means x {gmm.means_[order, 0].round(3)}, "
      f"log-likelihood {gmm.score(Xg) * len(Xg):.2f}")
print(f"  P(cluster | x=(2.2, 0)): from scratch {(dens / dens.sum()).round(3)}, "
      f"scikit-learn {gmm.predict_proba(probe)[0][order].round(3)} (left, right)")

# ---------------------------------------------------------------------------
# 4. Figure: K-means vs DBSCAN on moons, dendrogram, GMM soft memberships
# ---------------------------------------------------------------------------
Xm, _ = make_moons(n_samples=300, noise=0.06, random_state=4)
Xm = np.vstack([Xm, rng.uniform([-1.5, -1], [2.5, 1.5], (15, 2))])   # some noise points
km = KMeans(2, n_init=10, random_state=0).fit_predict(Xm)
dbl = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xm)
print(f"moons: DBSCAN found {len(set(dbl)) - (1 if -1 in dbl else 0)} clusters and {np.sum(dbl == -1)} noise points; "
      f"from scratch agrees on every point: {bool(np.array_equal(dbscan(Xm, 0.2, 5)[0], dbl))}")
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for c in range(2):
    axes[0].scatter(*Xm[km == c].T, s=8, color=PALETTE[c])
axes[0].set(title="K-means (K = 2): cuts across the moons")
for c in sorted(set(dbl)):
    axes[1].scatter(*Xm[dbl == c].T, s=8 if c >= 0 else 20, color=PALETTE[c] if c >= 0 else "black", marker="o" if c >= 0 else "x")
axes[1].set(title="DBSCAN (ε = 0.2, min 5): shapes + noise (×)")
Xh, _ = make_blobs(n_samples=12, centers=[[0, 0], [3, 3], [6, 0]], cluster_std=0.5, random_state=1)
dendrogram(linkage(Xh, "ward"), ax=axes[2], color_threshold=4, above_threshold_color=GRAY)
axes[2].axhline(4, color=ORANGE, ls="--", lw=1)
axes[2].set(title="Hierarchical (Ward): cut the tree to choose K", xlabel="point", ylabel="merge distance")
pr = gmm.predict_proba(Xg)[:, order[1]]
sc = axes[3].scatter(*Xg.T, c=pr, cmap="coolwarm", s=8)
fig.colorbar(sc, ax=axes[3], label="P(right cluster)")
axes[3].set(title="Gaussian mixture: soft membership")
for ax in axes:
    ax.title.set_fontsize(9)
save(fig, "density-hierarchical-and-mixture-clustering")
