---
name: K-means
tags: [unsupervised, clustering, interpretable, low-latency]
lessons: [k-means]
labs: [kmeans]
---

# K-means

## Problem type

Unsupervised clustering.

## Input

Numeric feature vectors, scaled. Batch $[n, p]$.

## Output

A cluster index per example and K centroids.

## Mental model

Drop K pins; each point joins its nearest pin; each pin moves to the middle of its group; repeat until nothing moves.

## Core objective

```math
\min_{\mu, c}\; \sum_i \lVert x_i - \mu_{c(i)}\rVert^2
```

## Training process

Lloyd's algorithm: alternate assignment to the nearest centroid and update to the mean; k-means++ initialization; several restarts (`n_init`).

## Preprocessing

Standardize features; reduce dimensionality for very high-dimensional data; handle outliers.

## Assumptions

Roughly spherical, similarly sized, similarly dense clusters; Euclidean distance is meaningful.

## Key hyperparameters

K; initialization method; `n_init`; maximum iterations.

## Good use cases

Customer or document segmentation as a first pass; vector quantization; building IVF indexes for vector search.

## Poor use cases

Non-convex shapes, very unequal cluster sizes or densities, heavy outliers, categorical data.

## Strengths

Simple, fast, scalable (mini-batch variant), centroids are easy to interpret.

## Weaknesses

Must choose K; local optima; assumes round clusters; sensitive to scaling and outliers.

## Computational cost

$O(nKp)$ per iteration.

## Evaluation metrics

Inertia (for the elbow), silhouette score, stability across seeds, downstream usefulness.

## Failure modes

Unscaled features; picking K by inertia alone; treating clusters as verified categories.

## Minimal implementation

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0).fit(X_scaled)
labels, centers = km.labels_, km.cluster_centers_
```

## Compared with neighbors

- **DBSCAN:** arbitrary shapes and noise, no K.
- **Gaussian mixture:** soft, elliptical clusters.
- **Hierarchical:** nested structure via a dendrogram.

## Learn more

[K-means clustering](../lessons/07-unsupervised/01-k-means.md)
