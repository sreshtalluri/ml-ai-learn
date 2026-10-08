---
name: Gaussian mixture model
tags: [unsupervised, clustering, generative, small-data]
lessons: [density-hierarchical-and-mixture-clustering, probability-and-statistics]
labs: []
---

# Gaussian mixture model

## Problem type

Unsupervised soft clustering and density estimation.

## Input

Scaled numeric features.

## Output

Per-point membership probabilities for each component; a likelihood for any point.

## Mental model

The data comes from K overlapping bell-shaped clouds, each with its own center, spread, and orientation; every point gets a probability of belonging to each cloud.

## Core objective

Maximize the log-likelihood $\sum_i \ln \sum_k \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$.

## Training process

Expectation-maximization: compute responsibilities (E-step), re-estimate weights, means, and covariances (M-step), repeat to convergence; multiple initializations.

## Preprocessing

Standardize features; reduce dimensionality if $p$ is large relative to $n$.

## Assumptions

Each cluster is approximately Gaussian (elliptical).

## Key hyperparameters

Number of components (choose by BIC/AIC and domain review); covariance type (full, tied, diag, spherical); regularization of covariances.

## Good use cases

Soft cluster assignments; elliptical clusters; anomaly scoring via low likelihood; generative sampling.

## Poor use cases

Strongly non-Gaussian shapes; very high dimensions with few points.

## Strengths

Soft memberships; models covariance; provides likelihoods.

## Weaknesses

Local optima; sensitive to initialization; must pick K; components can collapse without regularization.

## Computational cost

$O(nKp^2)$ per iteration with full covariances.

## Evaluation metrics

BIC/AIC, held-out log-likelihood, silhouette, stability.

## Failure modes

Collapsing components; wrong covariance type; overconfident memberships far from data.

## Minimal implementation

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0).fit(X_scaled)
probs = gmm.predict_proba(X_scaled); scores = gmm.score_samples(X_scaled)
```

## Compared with neighbors

- **K-means:** the hard-assignment, spherical special case.
- **DBSCAN:** arbitrary shapes, no probabilities.

## Learn more

[DBSCAN, hierarchical clustering, and Gaussian mixtures](../lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md)
