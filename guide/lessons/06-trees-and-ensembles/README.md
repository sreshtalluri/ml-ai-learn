---
title: Trees and ensembles
summary: Decision trees, random forests, and gradient boosting, the strongest default family for tabular data.
skill: classical-ml
---

# Module 6: Decision trees, random forests, and gradient boosting

Trees split data with simple rules and capture nonlinear interactions without feature scaling. Alone they overfit; combined they are among the most accurate models for structured data. Bagging (random forests) reduces variance; boosting (XGBoost, LightGBM, CatBoost) reduces bias by fitting each new tree to the remaining errors.

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Decision trees](01-decision-trees.md) | compute Gini impurity and information gain and choose a split by hand |
| 2 | [Random forests and gradient boosting](02-random-forests-and-boosting.md) | compare bagging and boosting and run one boosting round by hand |

> [!TIP]
> For structured/tabular data, gradient-boosted trees are often a top baseline. For raw images, audio, and language, deep learning usually has the stronger inductive bias.
