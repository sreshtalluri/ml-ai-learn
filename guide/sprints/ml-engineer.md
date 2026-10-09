---
title: ML Engineer interview sprint
role: ML Engineer
order: 1
summary: Seven days on what ML engineer loops test - modeling fundamentals, tree ensembles, deep learning, ML system design, and production.
---

# ML Engineer: 7-day interview sprint

**The job.** Turn a business problem into a model that works in production, and keep it working: data, features, training, evaluation, serving, and monitoring.

**What the loop usually tests**

| Round | What they probe | Where it's covered |
|---|---|---|
| ML fundamentals | Bias-variance, leakage, metrics, regularization, picking a model | Days 1 and 2 |
| Deep learning | Backprop, optimizers, training stability, embeddings, transformers | Days 3 and 4 |
| ML system design | End-to-end design: metrics, data, features, model, serving, monitoring | Day 5 |
| Production / MLOps | Latency, cost, drift, reliability, rollouts | Day 6 |
| ML coding | Implement a model or metric from scratch | Each lesson's implementation challenge |

## Day 1: How models fail

- Read: [Workflow, splits, and leakage](../lessons/02-ml-workflow/01-ml-workflow.md), [Overfitting and bias-variance](../lessons/02-ml-workflow/02-overfitting-and-bias-variance.md), [Classification metrics](../lessons/04-classification/02-classification-metrics.md)
- Labs: underfitting vs overfitting, classification threshold
- Cheat sheets: [classification metrics](../cheatsheets/classification-metrics.md), [failure modes](../cheatsheets/failure-modes.md)
- Be ready to: explain leakage with a real example, choose a metric for an imbalanced problem, and pick a threshold from costs.

## Day 2: Models that win on tabular data

- Read: [Linear regression](../lessons/03-regression/01-linear-regression.md), [Regularization](../lessons/03-regression/02-regularization-and-regression-metrics.md), [Logistic regression](../lessons/04-classification/01-logistic-regression.md), [Decision trees](../lessons/06-trees-and-ensembles/01-decision-trees.md), [Random forests and boosting](../lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md)
- Labs: linear regression, decision-tree builder, gradient boosting
- Cheat sheets: [model selection](../cheatsheets/model-selection.md), [regularization](../cheatsheets/regularization.md)
- Be ready to: compare bagging and boosting, say why GBDTs beat neural nets on most tabular data, and derive the MSE gradient.

## Day 3: Training neural networks

- Read: [Forward pass](../lessons/10-neural-networks/01-neural-network-forward-pass.md), [Gradient descent](../lessons/11-gradient-descent-backprop/01-gradient-descent.md), [Backpropagation](../lessons/11-gradient-descent-backprop/02-backpropagation.md), [Training and regularization](../lessons/12-training-regularization/01-training-and-regularization.md)
- Labs: forward pass, gradient descent, backpropagation
- Cheat sheets: [optimizers](../cheatsheets/optimizers.md), [loss functions](../cheatsheets/loss-functions.md)
- Be ready to: debug a loss that goes to NaN, explain Adam versus SGD, and say what batch norm and dropout do.

## Day 4: Embeddings and transformers

- Read: [Word embeddings](../lessons/09-classical-nlp/02-word-embeddings.md), [Self-attention](../lessons/14-transformers/01-self-attention.md), [Transformer architecture](../lessons/14-transformers/02-transformer-architecture.md), [Vector search](../lessons/16-rag/02-vector-search.md)
- Labs: self-attention, approximate nearest neighbors
- Cheat sheets: [transformer architecture](../cheatsheets/transformer-architecture.md), [vector search](../cheatsheets/vector-search.md)
- Be ready to: walk through Q, K, V with shapes, explain why attention is scaled by $\sqrt{d_k}$, and trade recall against latency in an ANN index.

## Day 5: ML system design

- Read: [ML system design](../lessons/22-ml-system-design/01-ml-system-design.md), [Experimentation and A/B testing](../lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)
- Labs: recommendation funnel, A/B test power and peeking
- Cheat sheet: [ML system design](../cheatsheets/ml-system-design.md)
- Be ready to: run the design framework end to end in 40 minutes on a feed, fraud, or search problem.

## Day 6: Production

- Read: [Production architecture](../lessons/20-production-ai/01-production-architecture.md), [Reliability, cost, and observability](../lessons/20-production-ai/02-reliability-cost-and-observability.md), [Serving LLMs](../lessons/18-llm-inference/02-serving-llms.md)
- Labs: production architecture, static vs continuous batching
- Cheat sheet: [production reliability](../cheatsheets/production-reliability.md)
- Be ready to: detect and respond to drift, design a safe rollout (shadow, canary, A/B), and size the serving cost.

## Day 7: Mock interview

1. Rapid-fire: 20 mixed cards from this sprint, 60 seconds each ([open the drill](https://sreshtalluri.github.io/ml-ai-learn/drill/?sprint=ml-engineer)). Reread any lesson behind a shaky card.
2. Design, 40 minutes each, out loud: *Design the "people you may know" recommender for a social network.* Then: *Design real-time payment fraud detection with a 50 ms budget.*
3. Coding, 20 minutes each: logistic regression trained with gradient descent in NumPy; ROC AUC from scores and labels without a library.
4. Project story: pick one project from the [project ladder](../lessons/23-projects/02-project-ladder.md) and practise its five-minute version: problem, baseline, what you tried, the result in numbers, and what you'd do next.
