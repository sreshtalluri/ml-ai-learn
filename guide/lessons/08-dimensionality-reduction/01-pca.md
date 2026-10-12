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
3. Set the correlation to 0. Where does PC1 point now, why does it keep only about 61% of the variance, and what does that mean for compression?

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

## Interview angle

<details>
<summary><strong>Explain PCA. What are the principal components mathematically?</strong></summary>

PCA finds orthogonal directions of maximal variance. Center the data, form the covariance matrix $C = \frac{1}{n-1}\tilde{X}^\top\tilde{X}$, and take its eigenvectors: $Cv_k = \lambda_k v_k$. The first eigenvector is the unit direction maximizing the projected variance $v^\top C v$; a Lagrange multiplier on $\lVert v \rVert = 1$ yields exactly the eigen-equation, and $\lambda_k$ is the variance along $v_k$. Each later component maximizes variance while staying orthogonal to the earlier ones, and projecting onto the top $k$ gives the best rank-$k$ reconstruction in squared error. In practice, compute it with an SVD of the centered data, $\tilde{X} = USV^\top$: the rows of $V^\top$ are the components and $\lambda_k = s_k^2/(n-1)$. SVD avoids forming $C$ and is more numerically stable. The explained variance ratio $\lambda_k/\sum_j \lambda_j$ says how much each component keeps.

</details>

<details>
<summary><strong>Should you standardize features before PCA?</strong></summary>

Standardize when features are in different units or scales; otherwise PCA mostly rediscovers whichever feature has the largest numeric variance. With income in dollars (variance around $10^8$) and age in years (variance around $10^2$), PC1 is essentially income, not because income is informative but because of its units. Standardizing is equivalent to running PCA on the correlation matrix, giving every feature equal initial weight. Don't standardize when features share meaningful units and their scale differences are signal: pixel intensities, or measurements on one common scale, where low-variance features are often noise and should count less. Standardizing those would amplify noise. Always center, because the components are defined relative to the mean. And fit both the scaler and PCA on training data only, inside a pipeline, then apply them unchanged to validation, test, and production data.

</details>

<details>
<summary><strong>You applied PCA keeping 95% of the variance before a classifier, and accuracy dropped significantly. Why?</strong></summary>

PCA is unsupervised: it keeps directions of high variance, not directions that predict the label. A low-variance direction can carry most of the class signal; picture two classes separated along a narrow axis while both spread widely along another. Dropping the bottom 5% of variance can discard exactly that axis. Other causes: features weren't standardized, so the kept variance is dominated by large-unit features; or PCA and scaling were fit differently between training and serving. What to do: treat the number of components as a hyperparameter chosen by validation performance, not by a variance threshold; try supervised alternatives such as LDA, or a regularized model on all features (L2 handles collinearity without discarding directions); and skip PCA for tree models. Checking each component's correlation with the target shows where the signal actually lives.

</details>

<details>
<summary><strong>Two features have covariance matrix [[3, 1], [1, 3]]. Find the principal components and the fraction of variance PC1 explains.</strong></summary>

Trace $= 6$ and determinant $= 9 - 1 = 8$. The eigenvalues solve $\lambda^2 - 6\lambda + 8 = 0$, so $\lambda = \frac{6 \pm \sqrt{36 - 32}}{2} = \frac{6 \pm 2}{2}$, giving $\lambda_1 = 4$ and $\lambda_2 = 2$. For $\lambda_1 = 4$: $(3 - 4)v_1 + v_2 = 0$, so $v_2 = v_1$, and normalized PC1 $= (1, 1)/\sqrt{2} \approx (0.707, 0.707)$. PC2 is orthogonal: $(1, -1)/\sqrt{2}$. PC1 explains $4/6 = 66.7\%$ of the variance. Interpretation: when two features have equal variance and positive covariance, PC1 is their scaled sum and PC2 their difference, whatever the correlation strength; the correlation only changes how the variance splits between them. Quick sanity check: the eigenvalues always sum to the trace, the total variance, here $4 + 2 = 6$.

</details>
