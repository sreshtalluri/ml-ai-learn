---
title: Regression metrics
summary: MAE, MSE, RMSE, R², and MAPE, with when to use each.
---

# Regression metrics

| Metric | Formula | Units | Use when | Watch out |
|---|---|---|---|---|
| MAE | $\frac{1}{n}\sum\lvert y - \hat{y}\rvert$ | target units | every unit of error costs the same; outliers shouldn't dominate | not differentiable at 0 (fine for reporting) |
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | squared units | training loss; large errors are especially bad | hard to interpret |
| RMSE | $\sqrt{\text{MSE}}$ | target units | report MSE in readable units | dominated by a few big misses |
| R² | $1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$ | none | compare against predicting the mean | can be negative on test data; not comparable across datasets |
| MAPE | $\frac{100\%}{n}\sum\left\lvert\frac{y - \hat{y}}{y}\right\rvert$ | percent | relative error across scales | explodes near $y = 0$; asymmetric |
| Huber loss | quadratic near 0, linear beyond δ | | robust training loss | choose δ |

**Worked example.** Errors $[-10, 10, -10, 10, -5]$: MAE 9.0, RMSE 9.2. Change one error to 100: MAE 28.0, RMSE 45.6. RMSE reacts five times more.

**Always also:** plot residuals versus predictions (curvature means a missing nonlinearity; a fan shape means non-constant variance), and compare against a baseline (mean, last value, seasonal naive).

Lessons: [Linear regression](../lessons/03-regression/01-linear-regression.md) · [Regularization and regression metrics](../lessons/03-regression/02-regularization-and-regression-metrics.md)
