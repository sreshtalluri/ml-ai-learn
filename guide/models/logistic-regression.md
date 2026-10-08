---
name: Logistic regression
tags: [supervised, classification, interpretable, low-latency, small-data, nlp]
lessons: [logistic-regression, classification-metrics]
labs: [threshold]
---

# Logistic regression

## Problem type

Supervised binary (or, with softmax, multiclass) classification.

## Input

Numeric features (scaled), one-hot categoricals, or sparse TF-IDF vectors. Batch $[n, p]$.

## Output

A probability $p(y = 1 \mid x)$ per example (or a distribution over classes).

## Mental model

Compute a linear score, then squeeze it into a probability between 0 and 1 with the sigmoid. Each weight shifts the log-odds.

## Core objective

```math
\min_{w,b}\; -\frac{1}{n}\sum_i \big[y_i\ln\sigma(w\cdot x_i + b) + (1-y_i)\ln(1 - \sigma(w\cdot x_i + b))\big] + \lambda\lVert w\rVert^2
```

## Training process

Convex optimization (L-BFGS, Newton, or gradient descent) with gradient $(\hat{p} - y)x$.

## Preprocessing

Scale numeric features; one-hot categoricals; add interactions or splines for curved boundaries.

## Assumptions

Log-odds are linear in the features; examples are independent.

## Key hyperparameters

Regularization strength `C` $= 1/\lambda$; penalty type (L1, L2, elastic net); class weights.

## Good use cases

Baselines for any classification task; sparse text; credit and risk scoring that needs explanations; settings that need calibrated probabilities.

## Poor use cases

Strongly nonlinear boundaries without feature engineering; raw images or audio.

## Strengths

Fast, interpretable, usually well calibrated, convex (unique optimum), strong with high-dimensional sparse data.

## Weaknesses

Linear boundary; sensitive to correlated features for interpretation; needs feature engineering for interactions.

## Computational cost

Training roughly $O(np)$ per iteration; inference $O(p)$.

## Evaluation metrics

Log loss, ROC AUC, PR AUC, precision and recall at the operating threshold, calibration (Brier score, reliability diagram).

## Failure modes

Default 0.5 threshold on imbalanced data; perfect separation without regularization; nonlinear patterns missed.

## Minimal implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
```

## Compared with neighbors

- **Linear regression:** same linear score, no sigmoid, squared error.
- **Linear SVM:** hinge loss, margins, no native probabilities.
- **Gradient boosting:** nonlinear and usually more accurate on tabular data; less interpretable.

## Learn more

[Logistic regression](../lessons/04-classification/01-logistic-regression.md) · [Classification metrics](../lessons/04-classification/02-classification-metrics.md)
