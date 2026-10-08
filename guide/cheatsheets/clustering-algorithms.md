---
title: Clustering algorithms
summary: K-means, DBSCAN, hierarchical, and Gaussian mixtures compared, with how to choose and validate.
---

# Clustering algorithms

| Method | Finds | Needs K? | Handles noise? | Cost | Fails on |
|---|---|---|---|---|---|
| K-means | round, similar-size clusters | yes | no | $O(nKp)$ per iteration | curved shapes, unequal sizes, outliers |
| DBSCAN | dense regions of any shape | no (ε, min_samples) | yes (label −1) | ~$O(n\log n)$ to $O(n^2)$ | varying densities, high dimensions |
| HDBSCAN | density clusters across scales | no | yes | similar to DBSCAN | very high dimensions |
| Hierarchical (Ward) | nested, compact clusters | cut the tree | no | $O(n^2)$ memory | large $n$ |
| Gaussian mixture | elliptical clusters, soft membership | yes (BIC helps) | via low likelihood | $O(nKp^2)$ per iteration | non-Gaussian shapes |

**Always:** scale features; try several K or parameter settings; check stability across seeds and subsamples.

**Choosing K:** elbow in inertia, silhouette score (−1 to 1), BIC for mixtures, and above all whether the clusters are useful.

**Anomaly detection:** DBSCAN noise, low mixture likelihood, distance to the nearest centroid, or Isolation Forest; thresholds need labeled incidents.

> [!IMPORTANT]
> A discovered cluster is a hypothesis, not a verified real-world category.

Lessons: [K-means](../lessons/07-unsupervised/01-k-means.md) · [DBSCAN, hierarchical, and mixtures](../lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md) · Lab: [K-means](https://sreshtalluri.github.io/ml-ai-learn/labs/kmeans/)
