---
name: Gradient boosting
tags: [supervised, classification, regression]
lessons: [random-forests-and-boosting, decision-trees, gradient-descent]
labs: []
---

# Gradient boosting

## Problem type

Supervised classification, regression, and ranking on tabular data.

## Input

Numeric and encoded categorical features; no scaling needed.

## Output

Sum of many small trees' outputs (passed through a sigmoid or softmax for classification).

## Mental model

Start with a simple guess, then repeatedly add a small tree that fixes part of the remaining error, each contribution scaled down by a learning rate.

## Core objective

Minimize a differentiable loss by gradient descent in function space: $F_m(x) = F_{m-1}(x) + \eta\, h_m(x)$, where $h_m$ is fitted to the negative gradient (residuals for squared error).

## Training process

Sequential: compute pseudo-residuals, fit a shallow tree, add it with shrinkage, repeat; stop early on validation loss.

## Preprocessing

Encode categoricals (or use CatBoost/LightGBM native handling); minimal otherwise.

## Assumptions

Few; the signal can be built additively from many shallow trees.

## Key hyperparameters

`n_estimators`, `learning_rate`, `max_depth`, `subsample`, column subsampling, minimum leaf size, L1/L2 leaf regularization.

## Good use cases

Most structured/tabular prediction problems; ranking; competitions; production scoring.

## Poor use cases

Raw images, audio, and long text; extrapolation beyond the training range; tiny datasets without tuning.

## Strengths

Often the most accurate model on tabular data; handles nonlinearity, interactions, and mixed types; fast inference.

## Weaknesses

Many interacting hyperparameters; can overfit without early stopping; sequential training; less interpretable (use SHAP).

## Computational cost

Training $O(M \cdot n \cdot p)$ with histogram methods; inference $O(M \cdot \text{depth})$ for $M$ trees.

## Evaluation metrics

Log loss, AUC, RMSE on a validation set used for early stopping; a separate test set.

## Failure modes

Too many rounds; target leakage via target encoding; drift; poor calibration without post-hoc calibration.

## Minimal implementation

```python
from sklearn.ensemble import HistGradientBoostingClassifier
gb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, early_stopping=True, random_state=0)
gb.fit(X_train, y_train)
```

## Compared with neighbors

- **Random forest:** parallel, variance reduction, less tuning.
- **XGBoost / LightGBM / CatBoost:** optimized gradient-boosting libraries with regularization and speed tricks.
- **Neural networks:** better for unstructured data; usually worse on medium tabular data.

## Learn more

[Random forests and gradient boosting](../lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md)
