---
title: Activation functions
summary: ReLU, GELU, sigmoid, tanh, and softmax, their derivatives, and where each is used.
---

# Activation functions

| Activation | Formula | Range | Derivative | Typical use |
|---|---|---|---|---|
| ReLU | $\max(0, z)$ | $[0, \infty)$ | 1 if $z > 0$, else 0 | hidden layers (default) |
| Leaky ReLU | $\max(\alpha z, z)$ | $(-\infty, \infty)$ | 1 or $\alpha$ | avoids dead units |
| GELU | $z\,\Phi(z)$ | about $(-0.17, \infty)$ | smooth | transformers |
| Sigmoid | $1/(1 + e^{-z})$ | $(0, 1)$ | $\sigma(1 - \sigma) \le 0.25$ | binary output probability, gates |
| Tanh | $\tanh z$ | $(-1, 1)$ | $1 - \tanh^2 z$ | RNNs, zero-centered outputs |
| Softmax | $e^{z_i}/\sum_j e^{z_j}$ | probabilities summing to 1 | $p_i(\delta_{ij} - p_j)$ | multiclass output, attention weights |

**Why nonlinearity:** without it, stacked linear layers collapse into one linear map ($W_2 W_1 x$).

**Gradient behavior:** sigmoid and tanh saturate (tiny derivatives for large $|z|$), contributing to vanishing gradients. ReLU's gradient is exactly 1 for positive inputs, but units stuck negative are "dead."

**Output layer by task:** regression, none; binary, sigmoid; multiclass, softmax; multi-label, one sigmoid per label.

Lessons: [The forward pass](../lessons/10-neural-networks/01-neural-network-forward-pass.md) · [Backpropagation](../lessons/11-gradient-descent-backprop/02-backpropagation.md) · Lab: [forward pass](https://sreshtalluri.github.io/ml-ai-learn/labs/nn-forward/)
