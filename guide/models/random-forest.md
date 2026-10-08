---
name: Random forest
tags: [supervised, classification, regression, small-data]
lessons: [random-forests-and-boosting, decision-trees]
labs: []
---

# Random forest

## Problem type

Supervised classification or regression on tabular data.

## Input

Numeric and encoded categorical features; no scaling needed.

## Output

Average of tree predictions (regression) or vote / averaged class probabilities (classification).

## Mental model

Ask many independent, slightly different experts (deep trees trained on bootstrap samples with random feature subsets) and average their answers so individual quirks cancel out.

## Core objective

Each tree greedily minimizes impurity on its bootstrap sample; the forest averages: $\hat{y}(x) = \frac{1}{B}\sum_b T_b(x)$.

## Training process

For each of $B$ trees: draw a bootstrap sample, grow a deep tree considering a random subset of `max_features` at each split. Trees train in parallel.

## Preprocessing

Encode categoricals; impute missing values (or use implementations that handle them); drop ID columns.

## Assumptions

Few; works best when trees' errors are not strongly correlated.

## Key hyperparameters

`n_estimators` (more is never worse for accuracy), `max_features`, `max_depth`, `min_samples_leaf`, `bootstrap`.

## Good use cases

Robust tabular baselines with little tuning; noisy data; feature-importance exploration (with care).

## Poor use cases

Extrapolation; very high-dimensional sparse text; strict latency or memory limits with hundreds of deep trees.

## Strengths

Hard to overfit by adding trees; little tuning; out-of-bag error estimate for free; handles nonlinearity and interactions.

## Weaknesses

Large models; slower inference than a single tree; usually a bit less accurate than tuned gradient boosting; biased impurity importances.

## Computational cost

Training about $O(B\,p'\,n\log n)$ with $p'$ = `max_features`; inference $O(B \cdot \text{depth})$.

## Evaluation metrics

Out-of-bag score; cross-validated accuracy, AUC, RMSE; permutation importance.

## Failure modes

Leakage via ID-like features; poor extrapolation; overconfident probabilities on unseen regions.

## Minimal implementation

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", oob_score=True, n_jobs=-1, random_state=0)
rf.fit(X_train, y_train); print(rf.oob_score_)
```

## Compared with neighbors

- **Decision tree:** a single tree; interpretable but high variance.
- **Gradient boosting:** sequential, reduces bias, usually more accurate when tuned.

## Learn more

[Random forests and gradient boosting](../lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md)
