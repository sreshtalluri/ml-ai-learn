---
title: Regression
summary: Predict continuous values with linear models, regularize them, and choose the right error metric.
skill: classical-ml
---

# Module 3: Regression

Regression predicts a number: a price, a demand forecast, a latency. This module builds the simplest useful model, linear regression, all the way from intuition to a gradient-descent update you compute by hand. It then shows how penalties (ridge, lasso, elastic net) keep weights under control and how to choose between MAE, RMSE, R², and MAPE.

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Linear regression](01-linear-regression.md) | compute predictions, residuals, MSE, gradients, and one update step by hand |
| 2 | [Regularization and regression metrics](02-regularization-and-regression-metrics.md) | explain how L1 and L2 change weights, and pick a metric that matches error cost |

**Interactive lab:** [Linear regression](https://sreshtalluri.github.io/ml-ai-learn/labs/linear-regression/): drag points, see residuals, toggle L1/L2.

**Code:** [`code/03-regression/`](../../code/03-regression/)
