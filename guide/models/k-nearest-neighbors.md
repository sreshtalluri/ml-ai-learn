---
name: K-nearest neighbors
tags: [supervised, classification, regression, interpretable, small-data]
lessons: [k-nearest-neighbors]
labs: [knn]
---

# K-nearest neighbors

## Problem type

Supervised classification or regression (instance-based, "lazy" learning).

## Input

Numeric feature vectors on comparable scales, or embeddings.

## Output

Majority class (or class proportions) of the K nearest neighbors; mean target for regression.

## Mental model

To label something new, find the K most similar examples you've already seen and let them vote. The training data is the model.

## Core objective

No training objective. Prediction: $\hat{y} = \text{mode}\{y_i : x_i \in N_K(q)\}$ using a distance such as Euclidean $\lVert q - x_i\rVert_2$.

## Training process

Store the training set (optionally build a KD-tree, ball tree, or approximate nearest-neighbor index).

## Preprocessing

Standardize features (essential); choose a distance suited to the data (cosine for embeddings); impute missing values; reduce dimensionality if $p$ is large.

## Assumptions

Nearby points in feature space have similar labels; features are meaningfully scaled.

## Key hyperparameters

K; distance metric; uniform versus distance weighting.

## Good use cases

Small to medium datasets; quick baselines; recommendation ("users like you"); similarity search over embeddings.

## Poor use cases

High-dimensional raw features; very large datasets with strict latency limits (without ANN indexes); unscaled mixed-unit features.

## Strengths

Simple, no training time, flexible nonlinear boundaries, easy to explain by showing neighbors.

## Weaknesses

Slow, memory-heavy prediction; sensitive to scaling and irrelevant features; suffers from the curse of dimensionality.

## Computational cost

Training $O(1)$; brute-force prediction $O(np)$ per query; ANN indexes make it sublinear.

## Evaluation metrics

Accuracy, F1, or RMSE with cross-validation over K.

## Failure modes

Unscaled features (distance dominated by one feature); class imbalance at boundaries; stale stored examples under drift.

## Minimal implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)).fit(X_train, y_train)
```

## Compared with neighbors

- **K-means:** unsupervised; K means clusters, not neighbors.
- **Logistic regression:** learns a global linear rule; far faster inference.
- **Embedding retrieval:** the same nearest-neighbor search at scale, used in RAG.

## Learn more

[K-nearest neighbors](../lessons/05-instance-and-probabilistic/01-k-nearest-neighbors.md)
