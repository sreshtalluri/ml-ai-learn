---
title: DBSCAN, hierarchical clustering, and Gaussian mixtures
summary: Choose a clustering method by the shape and density of your clusters, run DBSCAN by hand, read a dendrogram, and use soft memberships and anomaly scores.
skill: classical-ml
minutes: 30
prerequisites: [k-means]
related: [k-means, pca, probability-and-statistics]
---

# DBSCAN, hierarchical clustering, and Gaussian mixtures

> **Mental model.** K-means asks "which center is nearest?" DBSCAN asks "is this point in a crowded neighborhood?" Hierarchical clustering asks "which groups should merge next?" A Gaussian mixture asks "how likely is this point under each of several bell curves?" Different questions find different structure.

**You will learn to**
- Classify points as core, border, or noise in DBSCAN and tune ε and min_samples.
- Build and cut a dendrogram from hierarchical clustering.
- Explain Gaussian mixture models, soft memberships, and the EM idea.
- Use clustering ideas for anomaly detection.
- Pick a method from the data's cluster shapes, sizes, and noise.

**Why it matters.** K-means fails on curved shapes, unequal sizes, and outliers, which are common in real data. These three methods cover most of what K-means can't, and density ideas power practical anomaly detection.

## 1. Intuition

**DBSCAN (density-based).** A point is **core** if at least `min_samples` points (including itself) lie within distance ε. Clusters grow by chaining core points that are within ε of each other, then adding **border** points that are near a core point. Everything else is **noise**. You never choose K, and clusters can be any shape.

**Hierarchical (agglomerative).** Start with every point as its own cluster and repeatedly merge the two closest clusters. The full merge history is a tree, the **dendrogram**. Cut it at a height to get a clustering; different heights give coarser or finer groups.

**Gaussian mixture model (GMM).** Assume the data comes from K overlapping bell-shaped (Gaussian) clouds, each with its own center, spread, and orientation. Each point gets a *probability* of belonging to each cloud: **soft membership**. A point between two clusters might be 60/40.

## 2. Visualization

![Four panels. K-means splits two crescent shapes across the middle. DBSCAN recovers both crescents and marks scattered points as noise. A dendrogram of 12 points with a dashed cut line giving three clusters. A Gaussian mixture colors points by their probability of belonging to the right cluster, with a smooth transition between clusters.](../../figures/density-hierarchical-and-mixture-clustering.png)

*Synthetic data. On two moons plus 15 random points, DBSCAN finds both moons and labels 10 points as noise; K-means can only draw a straight boundary.*

*Interactive: the K-means lab's "Rings" and "Stretched" presets show the shapes that motivate these methods. [Open it](https://sreshtalluri.github.io/ml-ai-learn/labs/kmeans/).*

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| ε | DBSCAN neighborhood radius |
| `min_samples` | points (including itself) needed within ε for a core point |
| $\pi_k$ | mixing weight of component $k$ (sums to 1) |
| $\mu_k, \Sigma_k$ | mean and covariance of Gaussian component $k$ |
| $\gamma_{ik}$ | responsibility: probability point $i$ belongs to component $k$ |

### Gaussian mixture

```math
p(x) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(x \mid \mu_k, \Sigma_k),
\qquad
\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
```

**Expectation-maximization (EM)** alternates: compute responsibilities $\gamma_{ik}$ (E-step), then re-estimate $\pi_k, \mu_k, \Sigma_k$ as responsibility-weighted averages (M-step). K-means is the limiting case with hard assignments and identical round clusters.

### Linkage for hierarchical clustering

The distance between clusters $A$ and $B$ can be the closest pair (single linkage, finds chains), the farthest pair (complete), the average pair (average), or the increase in within-cluster variance from merging (Ward, which produces compact clusters similar to K-means).

### Worked example: DBSCAN by hand

Points on a line: $1.0, 1.5, 2.0, 2.4, 6.0, 6.3, 6.8, 12.0$, with ε = 0.6 and `min_samples` = 3.

| Point | Neighbors within 0.6 (including itself) | Count | Type |
|---|---|---|---|
| 1.0 | 1.0, 1.5 | 2 | border (near core 1.5) |
| 1.5 | 1.0, 1.5, 2.0 | 3 | **core** |
| 2.0 | 1.5, 2.0, 2.4 | 3 | **core** |
| 2.4 | 2.0, 2.4 | 2 | border (near core 2.0) |
| 6.0 | 6.0, 6.3 | 2 | border (near core 6.3) |
| 6.3 | 6.0, 6.3, 6.8 | 3 | **core** |
| 6.8 | 6.3, 6.8 | 2 | border |
| 12.0 | 12.0 | 1 | **noise** |

Result: cluster 0 = {1.0, 1.5, 2.0, 2.4}, cluster 1 = {6.0, 6.3, 6.8}, and 12.0 is noise. scikit-learn returns exactly these labels.

### Worked example: soft membership

A two-component GMM fit to synthetic data (a tight cluster at 0 and a wider one at 4) gives the point $(2.2, 0)$ responsibilities $[0.402, 0.598]$. Although it is closer to the left center, the right cluster's larger spread makes the point more plausible under it. K-means would assign it to the left cluster with full confidence.

## 4. Implementation

```python
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X_raw)

# DBSCAN: pick eps from the "knee" of sorted k-th neighbor distances
dists, _ = NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)
labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(X)      # -1 marks noise

Z = linkage(X, method="ward")
labels_h = fcluster(Z, t=3, criterion="maxclust")            # cut into 3 clusters

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0).fit(X)
probs = gmm.predict_proba(X)                                  # soft memberships
anomaly_score = -gmm.score_samples(X)                         # low likelihood = unusual
```

Runnable script (DBSCAN by hand, GMM probabilities, and figures): [`code/07-unsupervised/clustering_pca.py`](../../code/07-unsupervised/clustering_pca.py).

## 5. Engineering

| Method | Strength | Weakness |
|---|---|---|
| K-means | fast, simple, scales to millions | round, similar-size clusters; must pick K; outlier-sensitive |
| DBSCAN | arbitrary shapes, labels noise, no K | sensitive to ε and scaling; struggles with varying densities (HDBSCAN helps) |
| Hierarchical | nested structure, dendrogram for exploration | $O(n^2)$ memory, slow beyond tens of thousands of points |
| Gaussian mixture | soft memberships, elliptical clusters, likelihoods | assumes Gaussian shapes; initialization matters; choose K (BIC helps) |

**Anomaly detection.** Points DBSCAN labels as noise, points with low GMM likelihood, or points far from their K-means centroid are candidate anomalies. Isolation Forest is another common, scalable choice. Anomaly scores need a threshold, and that threshold should be set with whatever labeled incidents you have.

**Choosing K for GMMs.** Use BIC or AIC (they penalize extra components) together with domain review.

> [!WARNING]
> **Failure modes.** Unscaled features changing every distance; one ε that can't fit clusters of different densities; reading dendrogram structure in random data; GMM components collapsing onto single points (use regularization or a different covariance type).

### Common mistakes

- Running DBSCAN with default ε on unscaled data.
- Treating DBSCAN noise as errors to delete rather than signals to inspect.
- Using hierarchical clustering on 1 million points.

## 6. Knowledge check

<!-- quiz:density-hierarchical-and-mixture-clustering -->
**[Take the clustering methods quiz](../../quizzes/density-hierarchical-and-mixture-clustering.md)**
<!-- /quiz -->

**Practice exercise.** With the same points and ε = 0.6, what changes if `min_samples` = 2?

<details>
<summary>Solution</summary>

Every point with at least one neighbor within 0.6 becomes core: 1.0, 1.5, 2.0, 2.4, 6.0, 6.3, 6.8. The clusters stay {1.0–2.4} and {6.0–6.8}, but now all of their points are core. 12.0 is still noise (only itself).
</details>

**Implementation challenge.** Generate three blobs with very different densities. Show that one DBSCAN ε can't separate all three well, then try HDBSCAN (`sklearn.cluster.HDBSCAN`) and compare.

## Summary

- DBSCAN grows clusters from dense core points, finds arbitrary shapes, and labels noise; tune ε and `min_samples`, and scale first.
- Hierarchical clustering builds a merge tree; cut the dendrogram to choose the number of clusters.
- Gaussian mixtures give soft memberships and likelihoods via EM; K-means is a hard, round special case.
- Low density or low likelihood points are natural anomaly candidates.

**Next:** [PCA, t-SNE, and UMAP](../08-dimensionality-reduction/01-pca.md)

**Related:** [K-means](01-k-means.md) · [Model card: DBSCAN](../../models/dbscan.md) · [Model card: Gaussian mixture](../../models/gaussian-mixture-model.md)
