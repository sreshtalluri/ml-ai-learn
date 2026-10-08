---
title: PCA, t-SNE, and UMAP
summary: Compute principal components and explained variance for a small dataset, project onto them, and read nonlinear embedding plots without fooling yourself.
skill: classical-ml
minutes: 35
prerequisites: [vectors-and-matrices, probability-and-statistics]
related: [k-means, word-embeddings, k-nearest-neighbors]
---

# PCA, t-SNE, and UMAP

> **Mental model.** PCA rotates the coordinate system so the first axis points along the direction where the data is most spread out, the second along the next-most spread out direction at a right angle, and so on. Keep the first few axes and drop the rest.

**You will learn to**
- Center data and compute a covariance matrix.
- Find principal components (eigenvectors) and explained variance (eigenvalues), at an intuitive level and by hand for 2D.
- Project data onto components and reconstruct it.
- Choose the number of components from cumulative explained variance.
- Use t-SNE and UMAP for exploration while avoiding their classic misreadings.

**Why it matters.** High-dimensional data is slow, noisy, and impossible to look at. PCA compresses it, removes redundant correlated features, denoises, and makes plots possible. Nonlinear methods make beautiful 2D maps of embeddings, and are easy to over-interpret.

## 1. Intuition

Imagine a cloud of points shaped like a tilted cigar. Describing each point with the original $x$ and $y$ wastes information, because $x$ and $y$ are correlated. If you rotate your axes so one runs along the cigar, almost all the variation is along that one axis. The coordinate along the cigar is the first **principal component** (PC1); the short coordinate across it is PC2. Drop PC2 and you lose little.

Each component is a weighted combination of the original features, not one of them. That makes PCA great for compression and terrible for "which original feature matters": components mix features.

High variance is not the same as usefulness. The direction that varies most might be irrelevant noise for your prediction task.

## 2. Visualization

<!-- lab:pca -->
![Four panels. A small 2D dataset with PC1 (orange) along the main spread and PC2 (teal) at a right angle, points projected onto PC1. Cumulative explained variance for the 64-pixel digits dataset, crossing 90% at 31 components. Digits embedded in 2D by PCA (overlapping colors) and by t-SNE (well-separated colored islands).](../../figures/pca.png)

*Left: the worked example below; PC1 keeps 84% of the variance. Right: t-SNE separates the digit classes far more visibly than PCA, but distances between its islands and their sizes are not meaningful.*

*Interactive version: [open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/pca/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Guess the angle of PC1 by eye before pressing **Snap to PC1**.
2. Rotate the axis 90° away from PC1. What share of variance is kept now, and which eigenvalue is it?
3. Set the correlation to 0. Why does every angle keep about the same variance, and what does that mean for compression?

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $X$ | data, one row per example | $[n, p]$ |
| $\mu$ | column means | $[p]$ |
| $\tilde{X} = X - \mu$ | centered data | $[n, p]$ |
| $C$ | covariance matrix $\frac{1}{n-1}\tilde{X}^\top \tilde{X}$ | $[p, p]$ |
| $v_k$ | $k$-th principal component (unit eigenvector of $C$) | $[p]$ |
| $\lambda_k$ | variance along $v_k$ (eigenvalue) | scalar |
| $Z = \tilde{X} V_k$ | scores: coordinates in the first $k$ components | $[n, k]$ |

### The idea in one line

```math
C\,v_k = \lambda_k v_k, \qquad \lambda_1 \ge \lambda_2 \ge \dots, \qquad \text{explained variance ratio}_k = \frac{\lambda_k}{\sum_j \lambda_j}
```

An eigenvector of $C$ is a direction that $C$ only stretches (by $\lambda$), never turns. For a covariance matrix, those directions are the axes of the data's ellipse, and $\lambda$ is the variance along each.

### Worked example

Five points: $(2,1), (3,3), (4,3), (5,5), (6,3)$.

**Step 1: center.** Mean $\mu = (4, 3)$. Centered: $(-2,-2), (-1,0), (0,0), (1,2), (2,0)$.

**Step 2: covariance** (divide by $n - 1 = 4$):
$\text{var}(x_1) = \frac{4 + 1 + 0 + 1 + 4}{4} = 2.5$, $\text{var}(x_2) = \frac{4 + 0 + 0 + 4 + 0}{4} = 2$, $\text{cov} = \frac{4 + 0 + 0 + 2 + 0}{4} = 1.5$.

```math
C = \begin{bmatrix}2.5 & 1.5\\1.5 & 2\end{bmatrix}
```

**Step 3: eigenvalues.** For a 2×2 matrix, $\lambda = \frac{\text{tr} \pm \sqrt{\text{tr}^2 - 4\det}}{2}$ with $\text{tr} = 4.5$ and $\det = 2.5 \times 2 - 1.5^2 = 2.75$:
$\lambda = \frac{4.5 \pm \sqrt{20.25 - 11}}{2} = \frac{4.5 \pm 3.041}{2}$, so $\lambda_1 = 3.771$ and $\lambda_2 = 0.729$.

**Step 4: explained variance.** $3.771 / 4.5 = 0.838$: PC1 keeps 83.8% of the variance.

**Step 5: PC1 direction.** Solve $(C - \lambda_1 I)v = 0$: $(2.5 - 3.771)v_1 + 1.5 v_2 = 0$, so $v_2 = 0.847\,v_1$. Normalizing: $v_1 = (0.763, 0.646)$.

**Step 6: project.** The first point's score is $(-2)(0.763) + (-2)(0.646) = -2.819$. All scores: $[-2.819, -0.763, 0, 2.056, 1.526]$. Each 2D point is now one number. scikit-learn reports the same explained variance ratios, $[0.838, 0.162]$.

### t-SNE and UMAP (intuition only)

Both build a graph of which points are near which in the original space, then arrange points in 2D so that neighbors stay neighbors. They preserve **local** structure and distort global structure on purpose.

## 4. Implementation

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca = make_pipeline(StandardScaler(), PCA(n_components=0.90))   # keep 90% of variance
Z = pca.fit_transform(X_train)
print(pca[-1].n_components_, pca[-1].explained_variance_ratio_.cumsum()[-1])
X_back = pca.inverse_transform(Z)          # reconstruction (compression / denoising)

emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X_scaled)
# UMAP: pip install umap-learn; umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X_scaled)
```

From scratch: `vals, vecs = np.linalg.eigh(np.cov(X.T))`, sort by `vals` descending, and project with `(X - X.mean(0)) @ vecs[:, :k]`. In practice libraries use the SVD of the centered data, which is more stable.

Runnable script (the worked example, digits explained variance, PCA vs t-SNE): [`code/07-unsupervised/clustering_pca.py`](../../code/07-unsupervised/clustering_pca.py).

## 5. Engineering

**Uses.** Compression before a downstream model (fewer features, less overfitting, faster); removing multicollinearity; denoising by reconstructing from top components; visualization; speeding up nearest-neighbor search.

**Preprocessing.** Center always; standardize when features have different units, or PCA will just find the feature with the largest scale.

**Choosing k.** Cumulative explained variance (90% or 95% is common) or downstream validation performance. On the 64-pixel digits, 31 components hold 90% of the variance.

**Cost.** Full SVD is $O(np\min(n,p))$; randomized or incremental PCA handle large data.

**Leakage.** Fit PCA on training data only, inside the pipeline.

> [!IMPORTANT]
> **Reading t-SNE and UMAP plots.** Clear islands do not prove real categories. Cluster sizes and the distances between clusters are not meaningful; results change with perplexity, `n_neighbors`, and random seed; and both methods can create apparent clusters in data without them. Use them to generate hypotheses, then test those hypotheses in the original space.

### Common mistakes

- Skipping standardization when features have different units.
- Interpreting components as original features.
- Feeding t-SNE coordinates into a downstream model (it has no `transform` for new data, and its geometry is distorted).
- Dropping low-variance components that actually carry the predictive signal.

## 6. Knowledge check

<!-- quiz:pca -->
**[Take the PCA quiz](../../quizzes/pca.md)**
<!-- /quiz -->

**Practice exercise.** A covariance matrix is $\begin{bmatrix}4 & 0\\0 & 1\end{bmatrix}$. What are the principal components and their explained variance ratios?

<details>
<summary>Solution</summary>

The matrix is already diagonal, so the components are the original axes: PC1 = $(1, 0)$ with variance 4, PC2 = $(0, 1)$ with variance 1. Ratios: $4/5 = 0.8$ and $0.2$.
</details>

**Implementation challenge.** Implement PCA with `np.linalg.svd` on centered data, verify it matches scikit-learn (up to sign), then reconstruct digits from 5, 15, and 31 components and plot the reconstructions.

## Summary

- PCA finds orthogonal directions of maximum variance: eigenvectors of the covariance matrix, ordered by eigenvalue.
- Explained variance ratio $\lambda_k/\sum\lambda$ tells you how much each component keeps; choose k by cumulative variance or validation.
- Center (and usually scale) first; fit on training data only.
- Components mix features; high variance is not the same as predictive value.
- t-SNE and UMAP preserve neighborhoods for exploration; their clusters, sizes, and gaps are hypotheses, not facts.

**Next:** [From text to vectors](../09-classical-nlp/01-text-to-vectors.md)

**Related:** [K-means](../07-unsupervised/01-k-means.md) · [Word embeddings](../09-classical-nlp/02-word-embeddings.md) · [Model card: PCA](../../models/pca.md)
