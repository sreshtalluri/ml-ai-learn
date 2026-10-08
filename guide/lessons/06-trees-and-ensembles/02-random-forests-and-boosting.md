---
title: Random forests and gradient boosting
summary: Combine many trees by bagging (random forests) or boosting (XGBoost, LightGBM, CatBoost), run boosting rounds by hand, and tune the knobs that matter.
skill: classical-ml
minutes: 40
prerequisites: [decision-trees, gradient-descent]
related: [decision-trees, overfitting-and-bias-variance, classification-metrics]
---

# Random forests and gradient boosting

> **Mental model.** Bagging asks many independent experts and averages their answers, so individual quirks cancel out. Boosting builds a team in sequence: each new member studies only the mistakes the team is still making, and contributes a small, careful correction.

**You will learn to**
- Explain bootstrap sampling and random feature selection in random forests.
- Compare bagging and boosting in terms of bias and variance.
- Run gradient-boosting rounds on a tiny regression dataset by hand.
- Tune learning rate, number of estimators, depth, and subsampling, and know how they interact.
- Explain feature importance and SHAP at a high level.

**Why it matters.** For structured, tabular data, gradient-boosted trees (XGBoost, LightGBM, CatBoost) are often the strongest model you can train, and random forests are a robust, low-effort baseline. Knowing their knobs is a core skill for production ML.

## 1. Intuition

A single deep tree has low bias but high variance: retrain on slightly different data and it changes a lot.

**Bagging (bootstrap aggregating):** train many deep trees, each on a random sample of the rows drawn *with replacement* (a bootstrap sample). Average their predictions (or vote). Individual trees' errors are partly independent, so averaging cancels much of the variance. **Random forests** add a twist: at each split a tree may consider only a random subset of features, which makes trees more different from each other and the average even more stable.

**Boosting:** start with a simple prediction (the mean). Fit a small tree to the current errors (residuals). Add a fraction of it (the **learning rate**) to the ensemble. Repeat hundreds of times. Each tree is weak, but the sum becomes strong. Boosting mainly reduces bias, and it can overfit if you keep adding trees, so you stop early using a validation set.

## 2. Visualization

![Left: gradient boosting accuracy versus number of trees for learning rates 0.5 and 0.05, showing the larger learning rate peaking early and the smaller one improving steadily. Right: four data points with the boosted prediction after zero to three rounds, stepping closer to the points each round.](../../figures/random-forests-and-boosting.png)

*Synthetic data. A large learning rate gets good quickly and then overfits (training accuracy keeps rising while test accuracy stalls); a small learning rate needs more trees but is steadier. Right: the worked example below, round by round.*

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $B$ | number of trees in a forest |
| $T_b(x)$ | prediction of tree $b$ |
| $F_m(x)$ | boosted ensemble after $m$ rounds |
| $h_m(x)$ | the small tree fitted in round $m$ |
| $\eta$ | learning rate (shrinkage) |
| $r_i$ | residual, or more generally the negative gradient of the loss |

### Bagging

```math
\hat{y}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)
```

If each tree has variance $\sigma^2$ and pairwise correlation $\rho$, the average has variance $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. More trees shrink the second term; random feature selection lowers $\rho$, shrinking the first.

### Gradient boosting

```math
F_0(x) = \bar{y}, \qquad r_i = y_i - F_{m-1}(x_i), \qquad h_m = \text{tree fit to } \{(x_i, r_i)\}, \qquad F_m(x) = F_{m-1}(x) + \eta\, h_m(x)
```

For squared error the residual is exactly the negative gradient of the loss with respect to the prediction, so boosting is gradient descent in function space. Other losses (log loss for classification) replace $r_i$ with their negative gradient. XGBoost and LightGBM also use second derivatives and add regularization on leaf values.

### Worked example: three boosting rounds

Data $x = [1, 2, 3, 4]$, $y = [2, 4, 7, 9]$; stumps (depth-1 trees); $\eta = 0.5$.

**Round 0.** $F_0 = \bar{y} = 5.5$ for every point. MSE $= \frac{3.5^2 + 1.5^2 + 1.5^2 + 3.5^2}{4} = \frac{29}{4} = 7.25$.

**Round 1.** Residuals $r = y - F_0 = [-3.5, -1.5, 1.5, 3.5]$. The best stump splits at $x \le 2.5$: left mean $-2.5$, right mean $2.5$.
$F_1 = 5.5 + 0.5 \times [-2.5, -2.5, 2.5, 2.5] = [4.25, 4.25, 6.75, 6.75]$. MSE $= \frac{2.25^2 + 0.25^2 + 0.25^2 + 2.25^2}{4} = 2.5625$.

**Round 2.** Residuals $[-2.25, -0.25, 0.25, 2.25]$. Best stump splits at $x \le 1.5$: left $-2.25$, right mean $(-0.25 + 0.25 + 2.25)/3 = 0.75$.
$F_2 = [4.25 - 1.125,\; 4.25 + 0.375,\; 6.75 + 0.375,\; 6.75 + 0.375] = [3.125, 4.625, 7.125, 7.125]$. MSE $= 1.2969$.

**Round 3.** Residuals $[-1.125, -0.625, -0.125, 1.875]$. Split at $x \le 3.5$: left mean $-0.625$, right $1.875$.
$F_3 = [2.8125, 4.3125, 6.8125, 8.0625]$. MSE $= 0.4180$.

Each round fixes part of what remains, and the learning rate keeps every correction partial.

## 4. Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier           # pip install xgboost

rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=0)
rf.fit(X_train, y_train)

xgb = XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=100, eval_metric="logloss",
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)   # stops when validation loss stalls
print(xgb.best_iteration)
```

Runnable script (boosting by hand, depth comparison, learning-rate curves): [`code/06-trees-and-ensembles/trees.py`](../../code/06-trees-and-ensembles/trees.py). It uses scikit-learn's `GradientBoostingClassifier`, so no extra install is needed.

## 5. Engineering

**The knobs.**

| Knob | Effect if increased | Typical risk |
|---|---|---|
| `n_estimators` | more corrective trees | overfitting, slower training (use early stopping) |
| `learning_rate` | larger contribution per tree | unstable or aggressive fit |
| `max_depth` | more feature interactions | overfitting |
| `subsample` | fraction of rows per tree | too low can underfit |
| `colsample_bytree` | fraction of features per tree | too low can miss signal |
| regularization (`lambda`, `min_child_weight`) | simpler leaf values | too much underfits |

Lower learning rate plus more trees (with early stopping) is the usual recipe.

**Random forests** need little tuning: more trees never hurts accuracy (only time), and `max_features` is the main knob. Out-of-bag error gives a free validation estimate.

**Which library?** LightGBM is fast on large data; CatBoost handles categorical features natively and is robust with defaults; XGBoost is mature and widely deployed.

**Interpretation.** Impurity-based importances are biased toward high-cardinality features; permutation importance on validation data is more honest. SHAP values attribute each prediction to features additively and are the standard for explaining boosted trees.

**An honest note on the figure's data.** On the noisy moons dataset in the script, a depth-6 tree, a random forest, and gradient boosting all reach about 0.92 test accuracy. Ensembles don't always win by a large margin on a small, easy 2D problem. Their advantages show up as stability across resamples and on larger, messier, higher-dimensional tables.

> [!WARNING]
> **Failure modes.** Boosting for too many rounds without early stopping; tuning on the test set; leakage through target-encoded categoricals; extrapolation (tree ensembles predict constants outside the training range); trusting impurity importance.

### Common mistakes

- Treating `n_estimators` as "more is better" for boosting (it is for forests).
- Scaling features for tree models (unneeded).
- Comparing a tuned XGBoost to an untuned baseline and calling it a fair comparison.

## 6. Knowledge check

<!-- quiz:random-forests-and-boosting -->
**[Take the ensembles quiz](../../quizzes/random-forests-and-boosting.md)**
<!-- /quiz -->

**Practice exercise.** Data $x = [1, 2]$, $y = [10, 20]$, $\eta = 0.1$, stumps. Compute $F_0$, the first residuals, the stump's predictions, and $F_1$.

<details>
<summary>Solution</summary>

$F_0 = 15$. Residuals $[-5, 5]$. A stump separates the two points: $h = [-5, 5]$. $F_1 = 15 + 0.1 \times [-5, 5] = [14.5, 15.5]$. With $\eta = 0.1$ it takes many rounds to approach $[10, 20]$, which is the point of shrinkage.
</details>

**Implementation challenge.** Implement gradient boosting for regression from scratch using `DecisionTreeRegressor(max_depth=2)` as the weak learner, then compare its validation MSE curve with scikit-learn's `GradientBoostingRegressor` using the same settings.

## Summary

- Bagging (random forests) averages many decorrelated deep trees and mainly reduces variance.
- Boosting adds small trees fitted to residuals (negative gradients), scaled by a learning rate, and mainly reduces bias.
- Low learning rate plus many trees with early stopping is the standard boosting recipe.
- Gradient-boosted trees are a top baseline for tabular data; explain them with SHAP or permutation importance.

**Next:** [K-means clustering](../07-unsupervised/01-k-means.md)

**Related:** [Decision trees](01-decision-trees.md) · [Gradient descent](../11-gradient-descent-backprop/01-gradient-descent.md) · [Model card: gradient boosting](../../models/gradient-boosting.md) · [Model card: XGBoost](../../models/xgboost.md)
