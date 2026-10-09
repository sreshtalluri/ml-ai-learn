---
title: Model selection
summary: What to try first for each kind of problem, what to try next, and the questions to ask before trusting any result.
---

# Model selection cheat sheet

## Start simple, then earn complexity

Always fit a baseline first (predict the mean, the majority class, or a one-feature rule). A complex model only earns its place if it beats the baseline on a validation set that looks like production.

| Situation | Start with | Then consider | Lessons |
|---|---|---|---|
| Continuous tabular target | Linear / ridge regression | Random forest, gradient boosting | [Linear regression](../lessons/03-regression/01-linear-regression.md), [Ensembles](../lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md) |
| Binary or multiclass tabular | Logistic regression | XGBoost, LightGBM, CatBoost | [Logistic regression](../lessons/04-classification/01-logistic-regression.md) |
| Very small labeled dataset | Logistic regression, Naive Bayes, KNN | SVM, regularized linear models | [Naive Bayes](../lessons/05-instance-and-probabilistic/02-naive-bayes.md) |
| Sparse text with labels | TF-IDF + linear model | Fine-tuned transformer encoder | [Text to vectors](../lessons/09-classical-nlp/01-text-to-vectors.md) |
| No labels, want groups | K-means | DBSCAN, hierarchical, Gaussian mixture | [K-means](../lessons/07-unsupervised/01-k-means.md) |
| Rare events, few labels | Isolation-style anomaly scoring, thresholds | Semi-supervised, density models | [Clustering](../lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md) |
| High-dimensional visualization | PCA | UMAP / t-SNE (exploration only) | [PCA](../lessons/08-dimensionality-reduction/01-pca.md) |
| Image classification | Pretrained CNN or vision transformer | Fine-tuning, data augmentation | [CNNs](../lessons/13-deep-architectures/01-convolutional-networks.md) |
| Sequences / time series | Lagged features + gradient boosting | RNN/LSTM, temporal transformers | [RNNs](../lessons/13-deep-architectures/02-recurrent-networks.md) |
| Text generation | Pretrained decoder LLM + prompting | RAG, LoRA / PEFT, fine-tuning | [Adapting LLMs](../lessons/15-llms/03-adapting-llms.md) |
| Q&A over private or fresh knowledge | Retrieval + grounded generation | Hybrid search, reranking | [RAG](../lessons/16-rag/01-rag-pipeline.md) |
| Strict low latency | Simple model or small encoder | Distillation, quantization, caching | [Reliability and cost](../lessons/20-production-ai/02-reliability-cost-and-observability.md) |

> [!TIP]
> For structured (tabular) data, gradient-boosted trees are usually the strongest baseline. For raw images, audio, and language, deep learning has the better inductive bias.

## Choosing by constraint

| If you need... | Prefer | Avoid |
|---|---|---|
| Explanations stakeholders can read | Linear / logistic regression, shallow trees | Deep ensembles without explanation tooling |
| Calibrated probabilities | Logistic regression, calibrated boosting | Raw SVM scores, uncalibrated trees |
| Robustness to unscaled features | Trees and tree ensembles | KNN, SVM, neural nets, K-means without scaling |
| Training on tiny data | Strong priors: Naive Bayes, regularized linear | Large neural networks from scratch |
| Fast inference on CPU | Linear models, small trees, distilled models | KNN over large datasets, large transformers |
| Frequently changing facts | RAG (change the context) | Fine-tuning (changes weights, slow to update) |
| A new output format or style | Prompting, then fine-tuning / LoRA | RAG alone |

## How to study any model

1. State the task and output type.
2. Draw its input-to-output computation.
3. Write its objective or loss.
4. List assumptions and required preprocessing.
5. Identify the key hyperparameters.
6. Choose suitable evaluation metrics.
7. Describe failure modes.
8. Implement a baseline.
9. Do error analysis and compare against a simpler model.

## Questions to ask before trusting a result

- Was the test set truly unseen and representative of production?
- Could any feature leak the target or future information?
- Is the metric aligned with the business cost of errors and with class imbalance?
- How does the model perform across meaningful slices (segments, regions, time)?
- Is it calibrated? Does confidence match observed accuracy?
- What happens under missing values, drift, adversarial inputs, and outages?
- Can the pipeline be reproduced from the data version, code version, configuration, and random seed?

See every model side by side in the [model cards](../models/linear-regression.md).
