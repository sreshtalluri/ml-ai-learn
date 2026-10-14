---
title: PCA, t-SNE, and UMAP
summary: Compute principal components and explained variance for a small dataset, project onto them, and read nonlinear embedding plots without fooling yourself.
skill: classical-ml
minutes: 35
prerequisites: [vectors-and-matrices, probability-and-statistics]
related: [k-means, word-embeddings, k-nearest-neighbors]
adapted_from: [phases/01-math-foundations/10-dimensionality-reduction, phases/01-math-foundations/11-singular-value-decomposition]
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

### Build it from scratch

Six short NumPy functions cover PCA two ways, plus projection and reconstruction. The full script is [`code/08-dimensionality-reduction/pca.py`](../../code/08-dimensionality-reduction/pca.py).

**PCA by eigendecomposition.** This is the math section line for line: center, form the sample covariance (divide by $n - 1$), and take its eigenvectors. `eigh` is for symmetric matrices and returns eigenvalues in ascending order, so reverse them.

```python
def pca_eig(X):
    """PCA via the covariance matrix: center, covariance, eigendecomposition, sort largest first."""
    mu = X.mean(axis=0)                                  # [p]
    Xc = X - mu                                          # [n, p] centered data
    C = Xc.T @ Xc / (len(X) - 1)                         # [p, p] sample covariance, as scikit-learn uses
    vals, vecs = np.linalg.eigh(C)                       # eigh: C is symmetric; returns ascending order
    order = np.argsort(vals)[::-1]
    return mu, C, vals[order], vecs[:, order]            # columns of vecs are the components v_k
```

*Predict:* if you forgot to center (used `X.T @ X` directly) on the worked example, roughly which direction would the first eigenvector point?

**PCA by SVD.** Factor the centered data as $\tilde{X} = U S V^\top$. Then $\tilde{X}^\top \tilde{X} = V S^2 V^\top$, so the columns of $V$ are the same components and $\lambda_k = s_k^2/(n-1)$. This route never forms $\tilde{X}^\top \tilde{X}$, which squares the condition number and loses precision, so it is the safer default when features are nearly collinear.

```python
def pca_svd(X):
    """PCA via SVD of the centered data: Xc = U S V^T, components are rows of V^T, variances s^2 / (n - 1)."""
    mu = X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)   # never forms X^T X, so better conditioned
    return mu, s ** 2 / (len(X) - 1), Vt.T
```

*Predict:* for the five worked-example points, the singular values are $s_1, s_2$. What is $s_1^2$, given $\lambda_1 = 3.771$ and $n = 5$?

**Project and reconstruct.** Projecting keeps the first $k$ coordinates in the rotated system: $Z = \tilde{X} V_k$, shape $[n, k]$. Reconstructing rotates back and adds the mean: $\hat{X} = Z V_k^\top + \mu$, shape $[n, p]$. The squared error you lose equals the sum of the dropped eigenvalues (times $n - 1$).

```python
def project(X, mu, V, k):
    """Scores: coordinates of each centered point along the first k components."""
    return (X - mu) @ V[:, :k]                           # [n, p] @ [p, k] -> [n, k]

def reconstruct(Z, mu, V):
    """Map k-dimensional scores back to the original p features."""
    return Z @ V[:, :Z.shape[1]].T + mu                  # [n, k] @ [k, p] -> [n, p]
```

*Predict:* reconstruct the worked example from PC1 alone. Where does the point $(4, 3)$ land, and why?

**Sign convention.** If $v$ is an eigenvector, so is $-v$. Different routes (and library versions) can return either, so flip each component to make its largest entry positive before comparing.

```python
def sign_fix(V):
    """Eigenvectors are only defined up to sign: make the largest-magnitude entry of each column positive."""
    return V * np.sign(V[np.abs(V).argmax(axis=0), range(V.shape[1])])
```

*Predict:* after flipping PC1, what happens to the scores $[-2.819, -0.763, 0, 2.056, 1.526]$, and does the explained variance change?

### Use the library

scikit-learn's `PCA` centers the data and picks a solver with `svd_solver="auto"`: in version 1.5 and later, the covariance eigendecomposition (the first route above) when there are at least 10 times more rows than features and at most 1,000 features, a full SVD for small inputs, and a randomized SVD when you keep only a few components of a large matrix. Put it in a pipeline with the scaler so both are fit on training data only. t-SNE has no from-scratch version here; use the library.

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

### Run it

From `guide/`:

```bash
uv run code/08-dimensionality-reduction/pca.py
```

Key lines of output:

```text
eigenvalues [3.7707 0.7293], explained variance ratio [0.8379 0.1621]
PC1 direction [0.763  0.6464], projections (scores) [-2.8188 -0.763   0.      2.0558  1.526 ]
SVD route: variances [3.7707 0.7293], PC1 [0.763  0.6464]
rank-1 reconstruction: squared error / (n - 1) = 0.7293  (dropped eigenvalue λ2 = 0.7293)
scikit-learn explained_variance_ratio_ [0.8379 0.1621], PC1 [0.763  0.6464]

digits (64 dims): components for 90% variance = 31
  k= 5: variance kept 0.414, reconstruction MSE 0.559
  k=15: variance kept 0.709, reconstruction MSE 0.277
  k=31: variance kept 0.900, reconstruction MSE 0.095
scikit-learn agrees on digits variances: True
```

Both routes reproduce the worked example, and the rank-1 reconstruction error is exactly the dropped eigenvalue $\lambda_2$. On the standardized digits, reconstruction error falls as you keep more components, and 31 components hold 90% of the variance.

## 5. Engineering

**Uses.** Compression before a downstream model (fewer features, less overfitting, faster); removing multicollinearity; denoising by reconstructing from top components; anomaly detection (a point with high reconstruction error does not fit the learned subspace); visualization; speeding up nearest-neighbor search.

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

**Next:** [Feature engineering](../24-applied-ml/01-feature-engineering.md)

**Related:** [K-means](../07-unsupervised/01-k-means.md) · [Word embeddings](../09-classical-nlp/02-word-embeddings.md) · [Model card: PCA](../../models/pca.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Dimensionality Reduction](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/10-dimensionality-reduction), [Singular Value Decomposition](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/11-singular-value-decomposition).*
<!-- /deeper -->

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
