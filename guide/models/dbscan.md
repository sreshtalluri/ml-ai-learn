---
name: DBSCAN
tags: [unsupervised, clustering]
lessons: [density-hierarchical-and-mixture-clustering]
labs: []
---

# DBSCAN

## Problem type

Unsupervised density-based clustering and noise detection.

## Input

Scaled numeric features (or a precomputed distance matrix).

## Output

A cluster label per point, with −1 for noise.

## Mental model

A point in a crowded neighborhood is a core point; clusters grow by chaining core points that are close to each other; isolated points are noise.

## Core objective

No global objective. Core point: at least `min_samples` points within distance ε (including itself). Clusters are connected components of core points plus their border points.

## Training process

Range queries for each point (accelerated with spatial indexes), then expansion of clusters from core points.

## Preprocessing

Standardize features; choose a meaningful distance; consider dimensionality reduction.

## Assumptions

Clusters are regions of similar, higher density separated by lower density.

## Key hyperparameters

ε (`eps`) and `min_samples`; distance metric.

## Good use cases

Arbitrary cluster shapes; data with outliers; spatial data; anomaly flagging.

## Poor use cases

Clusters with very different densities; very high-dimensional data; when every point must belong to a cluster.

## Strengths

No K needed; finds non-convex shapes; labels noise explicitly.

## Weaknesses

Sensitive to ε and scaling; one ε can't fit varying densities (HDBSCAN helps); border points can be ambiguous.

## Computational cost

About $O(n \log n)$ with spatial indexes in low dimensions; up to $O(n^2)$ otherwise.

## Evaluation metrics

Fraction of noise, silhouette (with care), stability, domain review.

## Failure modes

Default ε on unscaled data; everything merging into one cluster or becoming noise.

## Minimal implementation

```python
from sklearn.cluster import DBSCAN
labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(X_scaled)
```

## Compared with neighbors

- **K-means:** needs K, round clusters, no noise label.
- **HDBSCAN:** handles varying densities, fewer parameters to tune.
- **Gaussian mixture:** probabilistic, elliptical clusters.

## Learn more

[DBSCAN, hierarchical clustering, and Gaussian mixtures](../lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md)
