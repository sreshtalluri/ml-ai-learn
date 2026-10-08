---
name: Support vector machine
tags: [supervised, classification, regression, nlp, small-data]
lessons: [support-vector-machines]
labs: []
---

# Support vector machine

## Problem type

Supervised classification (and regression via SVR).

## Input

Scaled numeric features or sparse TF-IDF vectors.

## Output

A signed decision value (margin); class by its sign. Probabilities need calibration.

## Mental model

Choose the separating boundary with the widest empty street around it; only the points on the street's edges (support vectors) matter. Kernels let the street curve.

## Core objective

```math
\min_{w,b}\; \tfrac12\lVert w\rVert^2 + C\sum_i \max\big(0,\; 1 - y_i(w\cdot x_i + b)\big)
```

## Training process

Quadratic programming in the dual (kernel SVMs) or coordinate descent / SGD in the primal (linear SVMs).

## Preprocessing

Standardize features; for text, TF-IDF.

## Assumptions

Classes are (approximately) separable in the original or kernel-induced space.

## Key hyperparameters

C; kernel (linear, RBF, polynomial); RBF width γ; class weights.

## Good use cases

High-dimensional sparse text (linear); small-to-medium datasets with complex boundaries (RBF).

## Poor use cases

Millions of rows with kernels; when probabilities or interpretability are required out of the box.

## Strengths

Effective in high dimensions; margin maximization regularizes; flexible kernels.

## Weaknesses

Kernel SVMs scale poorly; sensitive to C and γ; uncalibrated outputs; kernel models are hard to interpret.

## Computational cost

Kernel training roughly $O(n^2)$ to $O(n^3)$; linear SVM training roughly linear in $n$; inference proportional to the number of support vectors (kernel) or $p$ (linear).

## Evaluation metrics

Accuracy, F1, ROC AUC on decision values.

## Failure modes

Unscaled features; untuned hyperparameters; reading decision values as probabilities.

## Minimal implementation

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")).fit(X_train, y_train)
```

## Compared with neighbors

- **Logistic regression:** log loss and probabilities instead of hinge loss and margins.
- **KNN:** also local, but no training and no margin.
- **Gradient boosting:** usually stronger on tabular data at scale.

## Learn more

[Support vector machines](../lessons/05-instance-and-probabilistic/03-support-vector-machines.md)
