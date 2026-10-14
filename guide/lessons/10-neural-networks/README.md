---
title: Neural-network foundations
summary: Neurons, layers, weights, biases, activations, and tensor shapes, the building blocks of every deep model.
skill: deep-learning
---

# Module 10: Neural-network foundations

A neural network composes linear transformations with nonlinear activations. This module follows the numbers and shapes through a forward pass, then the loss that scores it, then PyTorch, the framework that turns both into a training loop.

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [The forward pass](01-neural-network-forward-pass.md) | compute a small MLP's output by hand and check every shape |
| 2 | [Loss functions](02-loss-functions.md) | choose and compute MSE, Huber, cross-entropy, focal, and contrastive losses, and explain why cross-entropy trains classifiers better than MSE |
| 3 | [Intro to PyTorch](03-intro-to-pytorch.md) | build a tiny autograd engine, then write a PyTorch training and evaluation loop on CPU |

**Interactive lab:** [Forward pass](https://sreshtalluri.github.io/ml-ai-learn/labs/nn-forward/)

**Code:** [`code/10-neural-networks/`](../../code/10-neural-networks/)
