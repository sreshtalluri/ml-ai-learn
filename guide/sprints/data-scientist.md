---
title: Data Scientist interview sprint
role: Data Scientist
order: 4
summary: Seven days on what data science loops test - statistics, A/B testing, metrics, classical ML, and product cases, plus enough LLM literacy for today's teams.
---

# Data Scientist: 7-day interview sprint

**The job.** Answer product questions with data: define metrics, run experiments, build models where they help, and explain the result to people who will act on it.

**What the loop usually tests**

| Round | What they probe | Where it's covered |
|---|---|---|
| Statistics | Distributions, hypothesis tests, confidence intervals, Bayes | Day 1 |
| Experimentation | A/B design, power, pitfalls, reading results | Day 1 |
| Modeling | Metrics, validation, classical models, interpretability | Days 2 to 4 |
| Product case | Metric definition, diagnosing a drop, deciding with data | Day 5 |
| LLM literacy | Evaluating and measuring LLM features | Day 6 |

## Day 1: Statistics and experiments

- Read: [Probability and statistics](../lessons/00-foundations/03-probability-and-statistics.md), [Experimentation and A/B testing](../lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)
- Lab: A/B test power and peeking
- Cheat sheet: [formulas](../cheatsheets/formulas.md)
- Be ready to: compute a sample size, explain a p-value without misstating it, and say why peeking inflates false positives.

## Day 2: Validation and metrics

- Read: [The language of ML](../lessons/01-ml-vocabulary/01-ml-vocabulary.md), [Workflow, splits, and leakage](../lessons/02-ml-workflow/01-ml-workflow.md), [Overfitting and bias-variance](../lessons/02-ml-workflow/02-overfitting-and-bias-variance.md), [Classification metrics](../lessons/04-classification/02-classification-metrics.md), [Regularization and regression metrics](../lessons/03-regression/02-regularization-and-regression-metrics.md)
- Labs: underfitting vs overfitting, classification threshold
- Cheat sheets: [classification metrics](../cheatsheets/classification-metrics.md), [regression metrics](../cheatsheets/regression-metrics.md)
- Be ready to: pick precision vs recall from business costs, spot leakage in a feature list, and choose MAE vs RMSE.

## Day 3: The models you'll actually use

- Read: [Linear regression](../lessons/03-regression/01-linear-regression.md), [Logistic regression](../lessons/04-classification/01-logistic-regression.md), [Decision trees](../lessons/06-trees-and-ensembles/01-decision-trees.md), [Random forests and boosting](../lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md)
- Labs: linear regression, decision-tree builder, gradient boosting
- Cheat sheet: [model selection](../cheatsheets/model-selection.md)
- Be ready to: interpret coefficients (and say when you can't), explain feature importance pitfalls, and justify a GBDT.

## Day 4: Segments, structure, and text

- Read: [K-means](../lessons/07-unsupervised/01-k-means.md), [Density, hierarchical, and mixture clustering](../lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md), [PCA](../lessons/08-dimensionality-reduction/01-pca.md), [Text to vectors](../lessons/09-classical-nlp/01-text-to-vectors.md)
- Labs: K-means, PCA, TF-IDF
- Cheat sheet: [clustering algorithms](../cheatsheets/clustering-algorithms.md)
- Be ready to: choose K and defend it, say why a discovered cluster isn't a real category, and build a quick text classifier baseline.

## Day 5: Product and system cases

- Read: [ML system design](../lessons/22-ml-system-design/01-ml-system-design.md), [Naive Bayes](../lessons/05-instance-and-probabilistic/02-naive-bayes.md)
- Lab: recommendation funnel
- Cheat sheet: [ML system design](../cheatsheets/ml-system-design.md)
- Be ready to: define a north-star metric with guardrails, and frame a churn or fraud problem as a model with a decision attached.

## Day 6: Measuring LLM features

- Read: [Adapting LLMs](../lessons/15-llms/03-adapting-llms.md), [The RAG pipeline](../lessons/16-rag/01-rag-pipeline.md), [Evaluating LLM systems](../lessons/17-llm-evaluation/01-llm-evaluation.md)
- Lab: RAG pipeline
- Cheat sheet: [LLM evaluation](../cheatsheets/llm-evaluation.md)
- Be ready to: design an offline eval for a summarization feature, validate an LLM judge against human labels, and plan the online test.

## Day 7: Mock interview

1. Rapid-fire: 20 mixed cards from this sprint, 60 seconds each ([open the drill](https://sreshtalluri.github.io/ml-ai-learn/drill/?sprint=data-scientist)).
2. Product case, 30 minutes: *Weekly active users dropped 6% last week. Walk me through how you'd find out why.*
3. Experiment case, 30 minutes: *We want to test a new checkout flow. Design the experiment, then interpret a result with p = 0.04 and a sample ratio mismatch.*
4. Modeling case, 30 minutes: *Predict which subscribers will cancel next month, and tell me what the business should do with the scores.*
