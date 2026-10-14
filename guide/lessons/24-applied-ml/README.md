---
title: Applied ML in practice
summary: "The day-to-day craft of tabular ML: engineer features without leakage, handle rare positives, tune efficiently, forecast time series, and detect anomalies."
skill: classical-ml
---

# Module 24: Applied ML in practice

Most real ML work happens around the model: shaping features, choosing metrics that fit a skewed problem, spending a tuning budget wisely, and validating in a way that matches how the model will be used. This module covers those skills, each with from-scratch code you can run.

| Problem | Core skill | Classic failure |
|---|---|---|
| Raw columns | scale, encode, impute, select | target encoding or scaling fit on test data |
| Rare positives | AUPRC, thresholds, class weights | trusting accuracy at 99.9% negatives |
| Many knobs | random search, halving, Bayesian optimization | reporting the best search score as performance |
| Ordered data | lags, walk-forward validation, baselines | shuffled K-fold, centered rolling windows |
| No labels | robust stats, isolation forest, alert budgets | z-scores masked by the outlier itself |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Feature engineering](01-feature-engineering.md) | scale, log-transform, one-hot and target-encode, impute with indicators, filter features, and avoid target-encoding leakage |
| 2 | [Imbalanced data](02-imbalanced-data.md) | evaluate rare-positive classifiers with precision, recall, MCC, and AUPRC, and use class weights, SMOTE, and threshold tuning |
| 3 | [Hyperparameter tuning](03-hyperparameter-tuning.md) | run random search, successive halving, and Bayesian optimization on log scales, and report nested-CV scores |
| 4 | [Time series](04-time-series.md) | build lag features, validate walk-forward, and beat naive and seasonal-naive baselines |
| 5 | [Anomaly detection](05-anomaly-detection.md) | score anomalies with robust z-scores, k-NN, LOF, and isolation forests, and set thresholds from an alert budget |

