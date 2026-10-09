---
title: Applied Scientist interview sprint
role: Applied Scientist
order: 3
summary: Seven days on what applied science loops test - derivations, modeling depth, deep learning, transformers, post-training, and experiment design.
---

# Applied Scientist: 7-day interview sprint

**The job.** Invent or adapt modeling approaches for a product problem and prove they work: the math, the experiments, and enough engineering to ship.

**What the loop usually tests**

| Round | What they probe | Where it's covered |
|---|---|---|
| ML breadth | Every classical model's assumptions, objective, and failure modes | Days 2 and 3 |
| ML depth / derivations | Gradients, likelihoods, bias-variance, why an algorithm converges | Days 1 to 4 (read the math sections in full) |
| Deep learning and LLMs | Architectures, optimization, attention, pretraining, post-training | Days 4 to 6 |
| Research and experiment design | Baselines, ablations, metrics, statistical significance | Day 6 |
| Paper or project deep dive | Defend your choices and results | Day 7 |

This sprint uses **full lessons**, not quick read: the math sections are the interview.

## Day 1: The math you'll be asked to use

- Read: [Probability and statistics](../lessons/00-foundations/03-probability-and-statistics.md), [Calculus for ML](../lessons/00-foundations/02-calculus-for-ml.md), [Vectors and matrices](../lessons/00-foundations/01-vectors-and-matrices.md), [Overfitting and bias-variance](../lessons/02-ml-workflow/02-overfitting-and-bias-variance.md)
- Lab: underfitting vs overfitting
- Cheat sheet: [formulas](../cheatsheets/formulas.md)
- Be ready to: derive the bias-variance decomposition, apply Bayes' rule to a base-rate problem, and explain MLE vs MAP.

## Day 2: Linear models, derived

- Read: [Linear regression](../lessons/03-regression/01-linear-regression.md), [Regularization](../lessons/03-regression/02-regularization-and-regression-metrics.md), [Logistic regression](../lessons/04-classification/01-logistic-regression.md), [Support vector machines](../lessons/05-instance-and-probabilistic/03-support-vector-machines.md), [Naive Bayes](../lessons/05-instance-and-probabilistic/02-naive-bayes.md)
- Labs: linear regression, classification threshold
- Be ready to: derive the normal equation and the logistic-loss gradient, explain L1 sparsity geometrically, and connect L2 to a Gaussian prior.

## Day 3: Trees, clustering, and dimensionality

- Read: [Decision trees](../lessons/06-trees-and-ensembles/01-decision-trees.md), [Random forests and boosting](../lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md), [K-means](../lessons/07-unsupervised/01-k-means.md), [Density, hierarchical, and mixture clustering](../lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md), [PCA](../lessons/08-dimensionality-reduction/01-pca.md)
- Labs: decision-tree builder, gradient boosting, K-means, PCA
- Be ready to: explain boosting as gradient descent in function space, K-means as coordinate descent (and EM for GMMs), and PCA via eigenvectors of the covariance matrix.

## Day 4: Deep learning

- Read: [Gradient descent](../lessons/11-gradient-descent-backprop/01-gradient-descent.md), [Backpropagation](../lessons/11-gradient-descent-backprop/02-backpropagation.md), [Training and regularization](../lessons/12-training-regularization/01-training-and-regularization.md), [Convolutional networks](../lessons/13-deep-architectures/01-convolutional-networks.md), [Recurrent networks](../lessons/13-deep-architectures/02-recurrent-networks.md)
- Labs: gradient descent, backpropagation, convolution
- Be ready to: backpropagate through a small network by hand, explain vanishing gradients and why residual connections and LSTMs help, and compare normalization schemes.

## Day 5: Transformers and generative models

- Read: [Self-attention](../lessons/14-transformers/01-self-attention.md), [Transformer architecture](../lessons/14-transformers/02-transformer-architecture.md), [Tokenization and pretraining](../lessons/15-llms/01-tokenization-and-pretraining.md), [Autoencoders, diffusion, and transfer learning](../lessons/13-deep-architectures/03-autoencoders-diffusion-and-transfer.md)
- Labs: self-attention, tokenizer
- Be ready to: count a transformer's parameters and FLOPs, explain the attention cost in sequence length, and contrast autoregressive and diffusion generation.

## Day 6: Post-training, evaluation, and experiments

- Read: [Fine-tuning in practice](../lessons/15-llms/04-fine-tuning-in-practice.md), [Evaluating LLM systems](../lessons/17-llm-evaluation/01-llm-evaluation.md), [Experimentation and A/B testing](../lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)
- Labs: LoRA and fine-tuning memory, A/B test power and peeking
- Be ready to: compare RLHF with PPO against DPO, design an ablation, and compute a sample size for an online test.

## Day 7: Mock interview

1. Rapid-fire: 20 mixed cards from this sprint, 90 seconds each ([open the drill](https://sreshtalluri.github.io/ml-ai-learn/drill/?sprint=applied-scientist)).
2. Whiteboard, 15 minutes each: the logistic-regression gradient; the nine backpropagation steps; the attention worked example with shapes.
3. Research design, 40 minutes: *Search relevance dropped for long-tail queries after a model update. Propose a modeling fix and an experiment that would convince a skeptical reviewer.*
4. Deep dive: pick a project from the [project ladder](../lessons/23-projects/02-project-ladder.md) and defend every choice against "why not X?"
