---
title: Learning paths
summary: Three routes through the course, by goal and time available, with the labs to do along the way.
---

# Learning paths

Pick one route and follow it in order. Each step is a lesson; the lab next to it is where to spend your hands-on time. Every lab lesson ends its visualization section with **Try it** prompts: make the prediction before touching a control.

## 1. Full course (12 weeks, 5 to 7 hours a week)

The complete sequence, matching the [12-week plan](lessons/20-projects/01-twelve-week-plan.md).

| Week | Lessons | Labs |
|---|---|---|
| 1 | [Vectors](lessons/00-foundations/01-vectors-and-matrices.md), [calculus](lessons/00-foundations/02-calculus-for-ml.md), [probability](lessons/00-foundations/03-probability-and-statistics.md), [Python](lessons/00-foundations/04-python-toolkit.md), [vocabulary](lessons/01-ml-vocabulary/01-ml-vocabulary.md), [paradigms](lessons/01-ml-vocabulary/02-learning-paradigms.md), [workflow](lessons/02-ml-workflow/01-ml-workflow.md), [overfitting](lessons/02-ml-workflow/02-overfitting-and-bias-variance.md) | paradigm guide, fit explorer |
| 2 | [Linear regression](lessons/03-regression/01-linear-regression.md), [regularization](lessons/03-regression/02-regularization-and-regression-metrics.md), [logistic regression](lessons/04-classification/01-logistic-regression.md), [metrics](lessons/04-classification/02-classification-metrics.md) | linear regression, threshold |
| 3 | [KNN](lessons/05-instance-and-probabilistic/01-k-nearest-neighbors.md), [Naive Bayes](lessons/05-instance-and-probabilistic/02-naive-bayes.md), [SVMs](lessons/05-instance-and-probabilistic/03-support-vector-machines.md), [decision trees](lessons/06-trees-and-ensembles/01-decision-trees.md) | KNN, decision tree |
| 4 | [Random forests and boosting](lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md) | boosting |
| 5 | [K-means](lessons/07-unsupervised/01-k-means.md), [other clustering](lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md), [PCA](lessons/08-dimensionality-reduction/01-pca.md) | K-means, PCA |
| 6 | [Text to vectors](lessons/09-classical-nlp/01-text-to-vectors.md), [embeddings](lessons/09-classical-nlp/02-word-embeddings.md) | TF-IDF |
| 7 | [Forward pass](lessons/10-neural-networks/01-neural-network-forward-pass.md), [gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md), [backpropagation](lessons/11-gradient-descent-backprop/02-backpropagation.md) | forward pass, gradient descent, backprop |
| 8 | [Training](lessons/12-training-regularization/01-training-and-regularization.md), [CNNs](lessons/13-deep-architectures/01-convolutional-networks.md), [RNNs](lessons/13-deep-architectures/02-recurrent-networks.md), [autoencoders and diffusion](lessons/13-deep-architectures/03-autoencoders-diffusion-and-transfer.md) | convolution |
| 9 | [Self-attention](lessons/14-transformers/01-self-attention.md), [transformer architecture](lessons/14-transformers/02-transformer-architecture.md) | self-attention |
| 10 | [Tokenization](lessons/15-llms/01-tokenization-and-pretraining.md), [decoding](lessons/15-llms/02-decoding.md), [adapting LLMs](lessons/15-llms/03-adapting-llms.md) | tokenizer, decoding |
| 11 | [RAG](lessons/16-rag/01-rag-pipeline.md), [evaluation](lessons/17-llm-evaluation/01-llm-evaluation.md) | RAG |
| 12 | [Architecture](lessons/18-production-ai/01-production-architecture.md), [reliability](lessons/18-production-ai/02-reliability-cost-and-observability.md), [security](lessons/19-safety-security/01-ai-security.md), [projects](lessons/20-projects/02-project-ladder.md) | architecture, prompt injection |

## 2. AI-engineering fast track (about 4 weeks)

For experienced software engineers who want to build LLM products and understand what's underneath. It skips most classical ML and keeps the ideas every AI engineer needs.

| Order | Lesson | Why it's on this path | Lab |
|---|---|---|---|
| 1 | [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md) | loss vs metric, training vs inference | |
| 2 | [Workflow, splits, and leakage](lessons/02-ml-workflow/01-ml-workflow.md) | evaluation discipline applies to LLM systems too | |
| 3 | [Overfitting and bias-variance](lessons/02-ml-workflow/02-overfitting-and-bias-variance.md) | why held-out evaluation matters | fit explorer |
| 4 | [Classification metrics](lessons/04-classification/02-classification-metrics.md) | precision, recall, thresholds, calibration | threshold |
| 5 | [Text to vectors](lessons/09-classical-nlp/01-text-to-vectors.md) | sparse retrieval (BM25's ancestor) | TF-IDF |
| 6 | [Word embeddings](lessons/09-classical-nlp/02-word-embeddings.md) | dense retrieval | |
| 7 | [Forward pass](lessons/10-neural-networks/01-neural-network-forward-pass.md) | what a layer computes | forward pass |
| 8 | [Gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md) | how anything is trained | gradient descent |
| 9 | [Self-attention](lessons/14-transformers/01-self-attention.md) | the core of every LLM | self-attention |
| 10 | [Transformer architecture](lessons/14-transformers/02-transformer-architecture.md) | where parameters, compute, and the KV cache live | |
| 11 | [Tokenization and pretraining](lessons/15-llms/01-tokenization-and-pretraining.md) | tokens drive cost and context limits | tokenizer |
| 12 | [Decoding](lessons/15-llms/02-decoding.md) | temperature and sampling | decoding |
| 13 | [Adapting LLMs](lessons/15-llms/03-adapting-llms.md) | prompting vs RAG vs LoRA | |
| 14 | [The RAG pipeline](lessons/16-rag/01-rag-pipeline.md) | the most common LLM architecture | RAG |
| 15 | [Evaluating LLM systems](lessons/17-llm-evaluation/01-llm-evaluation.md) | proving a change helped | |
| 16 | [Production architecture](lessons/18-production-ai/01-production-architecture.md) | the system around the model | architecture |
| 17 | [Reliability, cost, and observability](lessons/18-production-ai/02-reliability-cost-and-observability.md) | keeping it up and affordable | |
| 18 | [Securing AI systems](lessons/19-safety-security/01-ai-security.md) | prompt injection and tool safety | prompt injection |

Then build project 3 (RAG evaluation workbench) from the [project ladder](lessons/20-projects/02-project-ladder.md).

## 3. Interview prep (1 to 2 weeks)

For review once you've done a path above. Work from the cheat sheets, and take each lesson's quiz cold before rereading it.

1. Read the [model-selection](cheatsheets/model-selection.md), [classification metrics](cheatsheets/classification-metrics.md), [formulas](cheatsheets/formulas.md), and [failure modes](cheatsheets/failure-modes.md) cheat sheets.
2. Take every quiz without notes; reread only the lessons where you missed questions. The website's review queue tracks these for you.
3. Redo the hand calculations: [linear regression's gradient step](lessons/03-regression/01-linear-regression.md), [backpropagation's nine steps](lessons/11-gradient-descent-backprop/02-backpropagation.md), [attention's worked example](lessons/14-transformers/01-self-attention.md), [TF-IDF](lessons/09-classical-nlp/01-text-to-vectors.md), and [Gini](lessons/06-trees-and-ensembles/01-decision-trees.md). Check them in the Math Lab.
4. Practice the "explain it, calculate it, implement it, diagnose it" test out loud for: bias-variance, precision vs recall, bagging vs boosting, Q/K/V, RAG vs fine-tuning, and prompt-injection controls.
