---
name: Linear regression
tags: [supervised, regression, interpretable, low-latency, small-data]
lessons: [linear-regression, regularization-and-regression-metrics]
labs: [linear-regression]
---

# Linear regression

## Problem type

Supervised regression: predict a continuous number.

## Input

A feature vector $x$ of $p$ numeric values (categorical features one-hot encoded). Batch shape $[n, p]$.

## Output

One real number $\hat{y}$ per example (shape $[n]$). Unbounded: it can be negative or arbitrarily large.

## Mental model

Fit the flat line (or hyperplane) that best explains the target as a weighted sum of features. Each weight is "how much $\hat{y}$ changes per unit of that feature, holding the others fixed."

## Core objective

Minimize mean squared error:

```math
\min_{w,b}\; \frac{1}{n}\sum_{i=1}^{n}\left(y_i - (w \cdot x_i + b)\right)^2
```

## Training process

Closed form via the normal equation $w = (X^\top X)^{-1} X^\top y$ (in practice a QR or SVD solver), or iteratively with gradient descent using $\partial \text{MSE}/\partial w = \frac{2}{n}X^\top(\hat{y} - y)$.

## Preprocessing

One-hot encode categoricals. Standardize features if using gradient descent or regularization. Impute missing values. Consider `log(y)` for skewed, positive targets. Fit all preprocessing on the training split only.

## Assumptions

Linear relationship between features and target; independent errors; constant error variance (homoscedasticity); low multicollinearity for interpretable coefficients. Normal errors only matter for classical confidence intervals.

## Key hyperparameters

None for plain OLS (`fit_intercept` aside). Regularized variants add $\lambda$ (`alpha` in scikit-learn) and, for Elastic Net, the L1/L2 mix (`l1_ratio`). Gradient descent adds learning rate and number of steps.

## Good use cases

Baselines for any continuous target; pricing and demand where coefficients must be explained; small datasets; latency-critical serving; settings with mostly additive effects.

## Poor use cases

Strongly nonlinear relationships or interactions (unless engineered); targets with hard bounds (probabilities, counts near zero); heavy outliers with MSE loss; raw images, audio, or text.

## Strengths

Fast to train and serve; coefficients are directly interpretable; convex objective with a unique optimum; well-understood statistics; hard to overfit when $n \gg p$.

## Weaknesses

Cannot capture nonlinearity on its own; sensitive to outliers; unstable coefficients under multicollinearity; extrapolates linearly without limit.

## Computational cost

Training: $O(np^2 + p^3)$ for the closed form, $O(np)$ per gradient step. Inference: $O(p)$ per example. Memory: $p + 1$ numbers.

## Evaluation metrics

RMSE or MAE (in target units), R² (variance explained vs. predicting the mean), MAPE when relative error matters and targets are far from zero. Always inspect residual plots.

## Failure modes

Outliers dragging the fit; extrapolation beyond the training range; coefficient sign flips from correlated features; drift in the underlying relationship over time; leakage through features computed with future data.

## Minimal implementation

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
```

From scratch: [`code/03-regression/linear_regression.py`](../code/03-regression/linear_regression.py).

## Compared with neighbors

- **Ridge / Lasso:** same model plus a penalty on weights; prefer them with many or correlated features.
- **Logistic regression:** same linear score, passed through a sigmoid to predict a class probability.
- **Gradient-boosted trees:** capture nonlinearity and interactions automatically, usually more accurate on tabular data, less interpretable.

## Learn more

[Linear regression lesson](../lessons/03-regression/01-linear-regression.md)
