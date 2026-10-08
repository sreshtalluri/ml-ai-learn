---
name: XGBoost
tags: [supervised, classification, regression]
lessons: [random-forests-and-boosting]
labs: []
---

# XGBoost

## Problem type

Supervised classification, regression, and ranking on tabular data.

## Input

Numeric features (missing values handled natively); categoricals encoded or via native support.

## Output

Additive tree-ensemble score, transformed by the loss link (sigmoid for log loss).

## Mental model

Gradient boosting engineered for accuracy and speed: each new tree is chosen using both the gradient and the curvature of the loss, with explicit regularization on tree complexity.

## Core objective

```math
\mathcal{L} = \sum_i \ell(y_i, \hat{y}_i) + \sum_m \Big(\gamma T_m + \tfrac{1}{2}\lambda \lVert w_m\rVert^2\Big)
```

$T_m$ is the number of leaves and $w_m$ the leaf weights of tree $m$; splits are scored with a second-order Taylor approximation of the loss.

## Training process

Sequential boosting with histogram-based split finding, sparsity-aware handling of missing values, row and column subsampling, and early stopping on a validation set.

## Preprocessing

Minimal; encode categoricals; keep a clean validation split for early stopping.

## Assumptions

Same as gradient boosting.

## Key hyperparameters

`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `reg_alpha`, `gamma`.

## Good use cases

Production tabular models; ranking; anywhere a strong, fast tree ensemble is needed.

## Poor use cases

Unstructured data; small datasets where simpler models suffice; extrapolation.

## Strengths

Strong accuracy, built-in regularization, native missing-value handling, GPU support, mature tooling.

## Weaknesses

Many hyperparameters; overfits without early stopping; needs SHAP or similar for explanations.

## Computational cost

Roughly $O(M \cdot n \cdot p)$ training with histograms; fast inference.

## Evaluation metrics

Validation log loss or RMSE for early stopping; AUC, PR AUC, calibration on test.

## Failure modes

Early-stopping on the test set; leakage; distribution shift; miscalibration.

## Minimal implementation

```python
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=5, subsample=0.8,
                      colsample_bytree=0.8, early_stopping_rounds=100, eval_metric="logloss")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

## Compared with neighbors

- **LightGBM:** leaf-wise growth, often faster on large data.
- **CatBoost:** native categorical handling, strong defaults.
- **Random forest:** less tuning, usually slightly less accurate.

## Learn more

[Random forests and gradient boosting](../lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md)
