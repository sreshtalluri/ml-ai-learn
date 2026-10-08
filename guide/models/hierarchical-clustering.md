---
name: Hierarchical clustering
tags: [unsupervised, clustering, interpretable, small-data]
lessons: [density-hierarchical-and-mixture-clustering]
labs: []
---

# Hierarchical clustering

## Problem type

Unsupervised clustering that produces a tree of nested groups.

## Input

Scaled numeric features or a distance matrix.

## Output

A dendrogram (merge tree); a flat clustering by cutting it at a height or number of clusters.

## Mental model

Start with every point alone and repeatedly merge the two closest groups, recording each merge, until everything is one group.

## Core objective

Greedy agglomeration under a linkage criterion: single (closest pair), complete (farthest pair), average, or Ward (smallest increase in within-cluster variance).

## Training process

Compute pairwise distances, then merge iteratively, updating inter-cluster distances.

## Preprocessing

Standardize features; choose a distance and linkage that match the expected cluster shapes.

## Assumptions

Depends on linkage: Ward favors compact, round clusters; single linkage can follow chains.

## Key hyperparameters

Linkage; distance metric; cut height or number of clusters.

## Good use cases

Small to medium datasets; exploring structure at several granularities; taxonomies; gene-expression analysis.

## Poor use cases

Large datasets (memory and time); when merges should be revisable (they are final).

## Strengths

No need to choose K up front; the dendrogram is an interpretable summary.

## Weaknesses

$O(n^2)$ memory; greedy merges can't be undone; sensitive to linkage choice and noise.

## Computational cost

$O(n^2)$ memory and $O(n^2)$ to $O(n^3)$ time depending on the algorithm.

## Evaluation metrics

Cophenetic correlation, silhouette for chosen cuts, domain review.

## Failure modes

Chaining with single linkage; reading structure into dendrograms of random data.

## Minimal implementation

```python
from scipy.cluster.hierarchy import fcluster, linkage
Z = linkage(X_scaled, method="ward")
labels = fcluster(Z, t=3, criterion="maxclust")
```

## Compared with neighbors

- **K-means:** flat, fast, needs K.
- **DBSCAN:** density-based with noise.

## Learn more

[DBSCAN, hierarchical clustering, and Gaussian mixtures](../lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md)
