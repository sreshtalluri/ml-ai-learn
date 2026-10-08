---
title: Common failure modes
summary: The mistakes that most often make ML and LLM systems fail, and how to catch each one.
---

# Common failure modes

| Failure | Symptom | Catch it with |
|---|---|---|
| Data leakage | suspiciously high scores; one dominant feature | "available at prediction time?" review; time/group splits; pipelines |
| Wrong split | great offline, poor in production | splits that mirror deployment (time, group) |
| Accuracy on imbalanced data | high accuracy, useless model | precision, recall, PR-AUC, cost-based thresholds |
| Overfitting | train ≫ validation performance | validation curves, regularization, early stopping |
| Underfitting | both errors high | more capacity, better features |
| Unscaled features | distance/penalty-based models ignore features | standardization in the pipeline |
| Tuning on the test set | optimistic reported results | separate validation set or CV; test once |
| Training-serving skew | quality drop from day one | shared feature code; log and compare distributions |
| Drift | gradual quality decline | input, prediction, and quality monitoring |
| Miscalibration | confident wrong probabilities | reliability diagrams; recalibration |
| Vanishing/exploding gradients | early layers don't learn / NaN loss | ReLU, init, normalization, residuals, clipping |
| Shape/broadcast bugs | wrong losses without errors | shape comments; assertions; dummy tensors |
| K-means on non-round data | confident wrong clusters | try DBSCAN/GMM; check assumptions |
| Retrieval misses (RAG) | fluent wrong answers | recall@k before prompt tuning |
| Hallucination | unsupported claims | grounding, citations, verification, abstention |
| Prompt injection | model follows document instructions | least privilege, allowlists, approvals, tests |
| Retry storms / duplicate side effects | outages amplify; double charges | backoff, circuit breakers, idempotency |
| Unversioned prompts/models/indexes | can't reproduce or roll back | version everything; log versions per request |

**Before trusting any result:** unseen and representative test set? leakage? metric matches error costs? slice performance? calibration? behavior under drift and outages? reproducible?

Lessons: [The ML workflow](../lessons/02-ml-workflow/01-ml-workflow.md) · [Reliability](../lessons/18-production-ai/02-reliability-cost-and-observability.md) · [Security](../lessons/19-safety-security/01-ai-security.md)
