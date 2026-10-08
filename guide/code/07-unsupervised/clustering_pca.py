"""DBSCAN, hierarchical clustering, Gaussian mixtures, and PCA (with a worked 2D example).

Run from guide/:  uv run code/07-unsupervised/clustering_pca.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_digits, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PALETTE, PURPLE, TEAL, plt, save  # noqa: E402

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. DBSCAN worked example on a line
# ---------------------------------------------------------------------------
pts = np.array([[1.0], [1.5], [2.0], [2.4], [6.0], [6.3], [6.8], [12.0]])
db = DBSCAN(eps=0.6, min_samples=3).fit(pts)
print("DBSCAN eps=0.6 min_samples=3 labels:", db.labels_.tolist(), " core:", sorted(db.core_sample_indices_.tolist()))

# GMM soft membership for one point between two clusters
Xg, _ = make_blobs(n_samples=400, centers=[[0, 0], [4, 0]], cluster_std=[1.0, 1.5], random_state=3)
gmm = GaussianMixture(2, random_state=0).fit(Xg)
order = np.argsort(gmm.means_[:, 0])
probe = np.array([[2.2, 0.0]])
print(f"GMM P(cluster | x=(2.2, 0)) = {gmm.predict_proba(probe)[0][order].round(3)} (left, right)")

# ---------------------------------------------------------------------------
# 2. Figure: K-means vs DBSCAN on moons, dendrogram, GMM soft memberships
# ---------------------------------------------------------------------------
Xm, _ = make_moons(n_samples=300, noise=0.06, random_state=4)
Xm = np.vstack([Xm, rng.uniform([-1.5, -1], [2.5, 1.5], (15, 2))])   # some noise points
km = KMeans(2, n_init=10, random_state=0).fit_predict(Xm)
dbl = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xm)
print(f"moons: DBSCAN found {len(set(dbl)) - (1 if -1 in dbl else 0)} clusters and {np.sum(dbl == -1)} noise points")
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

# ---------------------------------------------------------------------------
# 3. PCA worked example
# ---------------------------------------------------------------------------
X = np.array([[2.0, 1.0], [3.0, 3.0], [4.0, 3.0], [5.0, 5.0], [6.0, 3.0]])
mu = X.mean(axis=0)
Xc = X - mu
C = Xc.T @ Xc / (len(X) - 1)                         # sample covariance, as scikit-learn uses
vals, vecs = np.linalg.eigh(C)
idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]
pc1 = vecs[:, 0] * np.sign(vecs[0, 0])
print(f"\nPCA: mean {mu}, covariance\n{C}")
print(f"eigenvalues {vals.round(4)}, explained variance ratio {(vals / vals.sum()).round(4)}")
print(f"PC1 direction {pc1.round(4)}, projections (scores) {(Xc @ pc1).round(4)}")
skl = PCA(2).fit(X)
print(f"scikit-learn explained_variance_ratio_ {skl.explained_variance_ratio_.round(4)}")

digits = load_digits()
Zs = StandardScaler().fit_transform(digits.data)
pca = PCA().fit(Zs)
cum = np.cumsum(pca.explained_variance_ratio_)
print(f"digits (64 dims): components for 90% variance = {np.searchsorted(cum, 0.90) + 1}")
emb_pca = PCA(2).fit_transform(Zs)
emb_tsne = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(Zs)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
ax = axes[0]
ax.scatter(*X.T, color=BLUE, s=40, zorder=3)
t = np.linspace(-3, 3, 2)
ax.plot(mu[0] + t * pc1[0], mu[1] + t * pc1[1], color=ORANGE, lw=2, label="PC1")
pc2 = np.array([-pc1[1], pc1[0]])
ax.plot(mu[0] + t * pc2[0] * 0.6, mu[1] + t * pc2[1] * 0.6, color=TEAL, lw=2, label="PC2")
proj = mu + np.outer(Xc @ pc1, pc1)
for a, b in zip(X, proj):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=GRAY, ls="--", lw=1)
ax.scatter(*proj.T, color=ORANGE, s=20, zorder=3)
ax.set(title=f"PC1 keeps {vals[0] / vals.sum():.0%} of the variance", aspect="equal", xlabel="x₁", ylabel="x₂")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
ax.plot(np.arange(1, 65), cum, color=PURPLE, lw=2)
ax.axhline(0.9, color=GRAY, ls="--", lw=1)
ax.set(title="Digits: cumulative explained variance", xlabel="number of components", ylabel="fraction of variance")
for ax, emb, title in [(axes[2], emb_pca, "PCA to 2D (linear)"), (axes[3], emb_tsne, "t-SNE to 2D (nonlinear)")]:
    ax.scatter(*emb.T, c=digits.target, cmap="tab10", s=4)
    ax.set(title=title, xticks=[], yticks=[])
for ax in axes:
    ax.title.set_fontsize(9)
save(fig, "pca")
