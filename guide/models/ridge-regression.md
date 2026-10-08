---
name: Ridge regression
tags: [supervised, regression, interpretable, low-latency, small-data]
lessons: [regularization-and-regression-metrics, linear-regression]
labs: [linear-regression]
---

# Ridge regression

## Problem type

Supervised regression.

## Input

Numeric feature vector $x \in \mathbb{R}^p$, standardized. Batch $[n, p]$.

## Output

A real number per example.

## Mental model

Linear regression with a fee on large weights: it shrinks every coefficient smoothly toward zero, spreading weight across correlated features instead of letting them fight.

## Core objective

```math
\min_{w,b}\; \frac{1}{n}\sum_i (y_i - w\cdot x_i - b)^2 + \lambda \lVert w\rVert_2^2
```

## Training process

Closed form $w = (X^\top X + \lambda I)^{-1}X^\top y$ on centered data, or gradient descent. The added $\lambda I$ makes the matrix invertible even with collinear features.

## Preprocessing

Standardize features (the penalty compares weight sizes); one-hot categoricals; impute missing values; fit on training data only.

## Assumptions

Same as linear regression (linearity, independent errors); tolerates multicollinearity.

## Key hyperparameters

$\lambda$ (`alpha` in scikit-learn), chosen by cross-validation on a log grid.

## Good use cases

Many correlated features; more features than examples; stable, interpretable baselines.

## Poor use cases

Strong nonlinearity; when you need sparse feature selection (use lasso).

## Strengths

Stable coefficients, closed-form solution, fast, rarely overfits with a tuned $\lambda$.

## Weaknesses

Never sets weights exactly to zero; still linear.

## Computational cost

Training $O(np^2 + p^3)$; inference $O(p)$.

## Evaluation metrics

RMSE, MAE, R² on held-out data.

## Failure modes

Unscaled features distorting the penalty; λ chosen on the test set; nonlinear signal left unmodeled.

## Minimal implementation

```python
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])).fit(X_train, y_train)
```

## Compared with neighbors

- **Linear regression:** ridge with λ = 0; less stable under collinearity.
- **Lasso:** L1 penalty, produces exact zeros.
- **Elastic net:** mixes both.

## Learn more

[Regularization and regression metrics](../lessons/03-regression/02-regularization-and-regression-metrics.md)
