---
title: Classification metrics
summary: Confusion matrix, precision, recall, F1, ROC and PR curves, calibration, and how to pick a threshold.
---

# Classification metrics

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | TP | FN |
| **Actually negative** | FP | TN |

| Metric | Formula | Question it answers |
|---|---|---|
| Accuracy | $(TP + TN)/N$ | How often is the model right? (misleading when classes are imbalanced) |
| Precision | $TP/(TP + FP)$ | Of the alarms, how many were real? |
| Recall (sensitivity, TPR) | $TP/(TP + FN)$ | Of the real positives, how many did we catch? |
| Specificity (TNR) | $TN/(TN + FP)$ | Of the negatives, how many did we leave alone? |
| FPR | $FP/(FP + TN)$ | How often do negatives raise alarms? |
| F1 | $2PR/(P + R)$ | Balance of precision and recall (harmonic mean) |
| ROC AUC | area under TPR vs FPR | Probability a random positive outranks a random negative |
| PR AUC (average precision) | area under precision vs recall | Ranking quality for rare positives |
| Log loss | $-\frac{1}{n}\sum[y\ln p + (1-y)\ln(1-p)]$ | Quality of the probabilities themselves |
| Brier score | $\frac{1}{n}\sum(p - y)^2$ | Calibration plus sharpness |

**Choosing a threshold:** minimize $c_{FP}\cdot FP + c_{FN}\cdot FN$ on validation data. Lower thresholds raise recall and false positives.

**When precision matters more:** automated actions, spam filtering, scarce reviewer time. **When recall matters more:** screening, fraud with human review, safety detection.

**Calibration:** among predictions near 0.3, about 30% should be positive. Check a reliability diagram; recalibrate with Platt scaling or isotonic regression.

> [!WARNING]
> Never report accuracy alone on imbalanced data, and never tune the threshold on the test set.

Lessons: [Classification metrics and thresholds](../lessons/04-classification/02-classification-metrics.md) · [Logistic regression](../lessons/04-classification/01-logistic-regression.md) · Lab: [threshold](https://sreshtalluri.github.io/ml-ai-learn/labs/threshold/)
