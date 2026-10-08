---
title: Regularization
summary: Every technique that fights overfitting, what it does, and when to reach for it.
---

# Regularization

| Technique | What it does | Where |
|---|---|---|
| L2 (ridge, weight decay) | adds $\lambda\sum w^2$; shrinks weights smoothly | linear models, neural nets (AdamW) |
| L1 (lasso) | adds $\lambda\sum\lvert w\rvert$; zeroes weights | sparse linear models |
| Elastic net | L1 + L2 | correlated features with sparsity |
| Early stopping | keep the best validation checkpoint | boosting, neural nets |
| Dropout | randomly zero activations (scale by $1/(1-p)$) | neural nets |
| Data augmentation | label-preserving variations | vision, audio, text |
| Tree constraints | max depth, min samples per leaf, pruning | trees and ensembles |
| Shrinkage | small learning rate per boosting round | gradient boosting |
| Bagging | average models on bootstrap samples | random forests |
| Normalization | batch/layer norm stabilizes activations | deep nets |
| Smaller model / fewer features | reduce capacity | anywhere |
| More data | the most reliable fix for variance | anywhere |

**Diagnose first:** a large train-validation gap means overfitting (regularize); both errors high means underfitting (add capacity, features, or train longer).

**Always:** scale features before L1/L2; tune strength by cross-validation on a log scale; never tune on the test set.

Lessons: [Overfitting and bias-variance](../lessons/02-ml-workflow/02-overfitting-and-bias-variance.md) · [Regularization for linear models](../lessons/03-regression/02-regularization-and-regression-metrics.md) · [Training and regularization](../lessons/12-training-regularization/01-training-and-regularization.md)
