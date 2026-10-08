---
name: PCA
tags: [unsupervised, dimensionality-reduction, interpretable, low-latency]
lessons: [pca]
labs: []
---

# PCA (principal component analysis)

## Problem type

Unsupervised linear dimensionality reduction.

## Input

Numeric features, centered (and usually standardized). Batch $[n, p]$.

## Output

Scores in $k < p$ dimensions; components (directions) and explained variance.

## Mental model

Rotate the coordinate system so the first axis points where the data varies most, the next where it varies most at a right angle, and keep only the first few axes.

## Core objective

Find orthonormal directions maximizing projected variance; equivalently, eigenvectors of the covariance matrix $C$ ordered by eigenvalue, or minimize reconstruction error $\lVert \tilde{X} - \tilde{X}V_kV_k^\top\rVert^2$.

## Training process

SVD of the centered data matrix (randomized or incremental for large data).

## Preprocessing

Center; standardize when units differ; handle missing values.

## Assumptions

Linear structure; variance reflects signal.

## Key hyperparameters

Number of components $k$ (or a target explained-variance fraction); whitening.

## Good use cases

Compression; removing multicollinearity; denoising; visualization; speeding up downstream models.

## Poor use cases

Nonlinear manifolds; when the predictive signal lives in low-variance directions; when interpretability of original features is required.

## Strengths

Fast, deterministic, invertible (approximately), well understood.

## Weaknesses

Linear only; components mix features; sensitive to scaling and outliers.

## Computational cost

$O(np\min(n, p))$ for full SVD; transform is $O(pk)$ per example.

## Evaluation metrics

Explained variance ratio; reconstruction error; downstream validation performance.

## Failure modes

Skipping standardization; fitting on all data before splitting; discarding useful low-variance signal.

## Minimal implementation

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pca = make_pipeline(StandardScaler(), PCA(n_components=0.95)).fit(X_train)
Z = pca.transform(X_test)
```

## Compared with neighbors

- **t-SNE / UMAP:** nonlinear, for visualization, not for downstream features.
- **Autoencoder:** nonlinear generalization; a linear autoencoder recovers the PCA subspace.

## Learn more

[PCA, t-SNE, and UMAP](../lessons/08-dimensionality-reduction/01-pca.md)
