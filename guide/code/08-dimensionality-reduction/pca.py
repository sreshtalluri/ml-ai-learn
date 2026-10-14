"""PCA from scratch (eigendecomposition and SVD), reconstruction error, scikit-learn check,
digits explained variance, PCA vs t-SNE figure.

Run from guide/:  uv run code/08-dimensionality-reduction/pca.py
"""
# Adapted from AI Engineering from Scratch (MIT), phases/01-math-foundations/10-dimensionality-reduction
# and phases/01-math-foundations/11-singular-value-decomposition
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

np.set_printoptions(suppress=True)


# ---------------------------------------------------------------------------
# 1. From scratch (NumPy only)
# ---------------------------------------------------------------------------
def pca_eig(X):
    """PCA via the covariance matrix: center, covariance, eigendecomposition, sort largest first."""
    mu = X.mean(axis=0)                                  # [p]
    Xc = X - mu                                          # [n, p] centered data
    C = Xc.T @ Xc / (len(X) - 1)                         # [p, p] sample covariance, as scikit-learn uses
    vals, vecs = np.linalg.eigh(C)                       # eigh: C is symmetric; returns ascending order
    order = np.argsort(vals)[::-1]
    return mu, C, vals[order], vecs[:, order]            # columns of vecs are the components v_k


def pca_svd(X):
    """PCA via SVD of the centered data: Xc = U S V^T, components are rows of V^T, variances s^2 / (n - 1)."""
    mu = X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)   # never forms X^T X, so better conditioned
    return mu, s ** 2 / (len(X) - 1), Vt.T


def project(X, mu, V, k):
    """Scores: coordinates of each centered point along the first k components."""
    return (X - mu) @ V[:, :k]                           # [n, p] @ [p, k] -> [n, k]


def reconstruct(Z, mu, V):
    """Map k-dimensional scores back to the original p features."""
    return Z @ V[:, :Z.shape[1]].T + mu                  # [n, k] @ [k, p] -> [n, p]


def sign_fix(V):
    """Eigenvectors are only defined up to sign: make the largest-magnitude entry of each column positive."""
    return V * np.sign(V[np.abs(V).argmax(axis=0), range(V.shape[1])])


# ---------------------------------------------------------------------------
# 2. Worked example: five 2D points
# ---------------------------------------------------------------------------
X = np.array([[2.0, 1.0], [3.0, 3.0], [4.0, 3.0], [5.0, 5.0], [6.0, 3.0]])
mu, C, vals, vecs = pca_eig(X)
vecs = sign_fix(vecs)
pc1 = vecs[:, 0]
print(f"PCA: mean {mu}, covariance\n{C}")
print(f"eigenvalues {vals.round(4)}, explained variance ratio {(vals / vals.sum()).round(4)}")
print(f"PC1 direction {pc1.round(4)}, projections (scores) {project(X, mu, vecs, 1).ravel().round(4)}")
_, vals_svd, vecs_svd = pca_svd(X)
print(f"SVD route: variances {vals_svd.round(4)}, PC1 {sign_fix(vecs_svd)[:, 0].round(4)}")
X1 = reconstruct(project(X, mu, vecs, 1), mu, vecs)
err = ((X - X1) ** 2).sum() / (len(X) - 1)
print(f"rank-1 reconstruction: squared error / (n - 1) = {err:.4f}  (dropped eigenvalue λ2 = {vals[1]:.4f})")
skl = PCA(2).fit(X)
print(f"scikit-learn explained_variance_ratio_ {skl.explained_variance_ratio_.round(4)}, "
      f"PC1 {sign_fix(skl.components_.T)[:, 0].round(4)}")

# ---------------------------------------------------------------------------
# 3. Digits: 64 pixel features
# ---------------------------------------------------------------------------
digits = load_digits()
Zs = StandardScaler().fit_transform(digits.data)
mu_d, var_d, V_d = pca_svd(Zs)
cum = np.cumsum(var_d) / var_d.sum()
print(f"\ndigits (64 dims): components for 90% variance = {np.searchsorted(cum, 0.90) + 1}")
for k in (5, 15, 31):
    R = reconstruct(project(Zs, mu_d, V_d, k), mu_d, V_d)
    print(f"  k={k:2d}: variance kept {cum[k - 1]:.3f}, reconstruction MSE {((Zs - R) ** 2).mean():.3f}")
pca = PCA().fit(Zs)
print(f"scikit-learn agrees on digits variances: {bool(np.allclose(pca.explained_variance_, var_d))}")
emb_pca = project(Zs, mu_d, V_d, 2)
emb_tsne = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(Zs)

# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
ax = axes[0]
ax.scatter(*X.T, color=BLUE, s=40, zorder=3)
t = np.linspace(-3, 3, 2)
ax.plot(mu[0] + t * pc1[0], mu[1] + t * pc1[1], color=ORANGE, lw=2, label="PC1")
pc2 = np.array([-pc1[1], pc1[0]])
ax.plot(mu[0] + t * pc2[0] * 0.6, mu[1] + t * pc2[1] * 0.6, color=TEAL, lw=2, label="PC2")
for a, b in zip(X, X1):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=GRAY, ls="--", lw=1)
ax.scatter(*X1.T, color=ORANGE, s=20, zorder=3)
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
