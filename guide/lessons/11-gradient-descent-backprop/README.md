---
title: Gradient descent and backpropagation
summary: "How models learn: follow the gradient downhill, and compute that gradient efficiently with the chain rule."
skill: deep-learning
---

# Module 11: Gradient descent and backpropagation

Training is a loop: forward pass, loss, backward pass, optimizer step. Gradient descent says which way to move the parameters; backpropagation computes the gradient for every parameter in one backward sweep.

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Gradient descent](01-gradient-descent.md) | compute updates by hand, find the stable learning rate, and compare SGD, momentum, and Adam |
| 2 | [Backpropagation](02-backpropagation.md) | backpropagate through a one-neuron classifier by hand and explain vanishing and exploding gradients |

**Interactive labs:** [Gradient descent](https://sreshtalluri.github.io/ml-ai-learn/labs/gradient-descent/) · [Backpropagation](https://sreshtalluri.github.io/ml-ai-learn/labs/backprop/)
