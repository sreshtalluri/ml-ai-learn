---
name: Decision tree
tags: [supervised, classification, regression, interpretable, low-latency, small-data]
lessons: [decision-trees]
labs: []
---

# Decision tree

## Problem type

Supervised classification or regression.

## Input

Numeric and (encoded) categorical features; no scaling needed.

## Output

The class distribution or mean target of the leaf an example falls into.

## Mental model

A learned flowchart of yes/no questions, each chosen to make the groups below it as pure as possible.

## Core objective

Greedy splitting: at each node choose the feature and threshold maximizing impurity decrease, $\Delta = I(\text{parent}) - \sum_{\text{child}} \frac{n_c}{n}I(c)$, with Gini $1 - \sum p_k^2$, entropy, or variance for regression.

## Training process

Recursive partitioning until a stopping rule (depth, minimum samples) is met; optional cost-complexity pruning.

## Preprocessing

Minimal: encode categoricals; handle missing values (some implementations do natively); drop ID-like columns.

## Assumptions

The target can be approximated by axis-aligned rectangular regions.

## Key hyperparameters

`max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`, `ccp_alpha`, criterion.

## Good use cases

Interpretable rules; quick baselines; building block for forests and boosting.

## Poor use cases

Smooth or diagonal boundaries; extrapolation; whenever stability matters (single trees change a lot with small data changes).

## Strengths

Readable when shallow; no scaling; handles interactions and mixed feature types; fast inference.

## Weaknesses

High variance; overfits when deep; staircase approximations; biased impurity importances.

## Computational cost

Training about $O(p\,n\log n)$; inference $O(\text{depth})$.

## Evaluation metrics

Accuracy, F1, RMSE; tree depth and leaf count for complexity.

## Failure modes

Memorization with unlimited depth; splits on high-cardinality IDs; no extrapolation.

## Minimal implementation

```python
from sklearn.tree import DecisionTreeClassifier, export_text
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10).fit(X_train, y_train)
print(export_text(tree, feature_names=list(feature_names)))
```

## Compared with neighbors

- **Random forest:** averages many randomized trees; lower variance.
- **Gradient boosting:** sums many small trees fitted to residuals; lower bias.
- **Logistic regression:** smooth linear boundary, also interpretable.

## Learn more

[Decision trees](../lessons/06-trees-and-ensembles/01-decision-trees.md)
