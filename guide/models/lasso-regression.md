---
name: Lasso regression
tags: [supervised, regression, interpretable, low-latency, small-data]
lessons: [regularization-and-regression-metrics, linear-regression]
labs: [linear-regression]
---

# Lasso regression

## Problem type

Supervised regression with built-in feature selection.

## Input

Numeric feature vector, standardized. Batch $[n, p]$.

## Output

A real number per example; a sparse weight vector.

## Mental model

Linear regression that charges a fee proportional to each weight's absolute size, so unhelpful weights are pushed all the way to zero and dropped.

## Core objective

```math
\min_{w,b}\; \frac{1}{n}\sum_i (y_i - w\cdot x_i - b)^2 + \lambda \lVert w\rVert_1
```

## Training process

Coordinate descent with soft-thresholding: each weight is updated to $\text{sign}(\rho)\max(|\rho| - \lambda/2, 0)/s$, where $\rho$ is its correlation with the current residual and $s$ its variance (for the objective above).

## Preprocessing

Standardize features; one-hot categoricals; impute missing values.

## Assumptions

Linear relationship; a relatively small number of truly relevant features.

## Key hyperparameters

$\lambda$ (`alpha`), chosen by cross-validation (`LassoCV`).

## Good use cases

High-dimensional data where you expect few relevant features; producing compact, interpretable models.

## Poor use cases

Groups of highly correlated features (it picks one arbitrarily); strongly nonlinear problems.

## Strengths

Sparse, interpretable models; automatic feature selection; cheap inference.

## Weaknesses

Unstable selection among correlated features; can underfit when many features matter a little.

## Computational cost

Coordinate descent $O(np)$ per pass; inference $O(\text{nonzero weights})$.

## Evaluation metrics

RMSE, MAE, R²; number and stability of selected features across folds.

## Failure modes

Treating dropped features as proven irrelevant; unscaled features; selection that changes with every resample.

## Minimal implementation

```python
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
model = make_pipeline(StandardScaler(), LassoCV(cv=5)).fit(X_train, y_train)
selected = model[-1].coef_ != 0
```

## Compared with neighbors

- **Ridge:** shrinks smoothly, keeps all features.
- **Elastic net:** adds L2 to stabilize selection among correlated features.

## Learn more

[Regularization and regression metrics](../lessons/03-regression/02-regularization-and-regression-metrics.md)
