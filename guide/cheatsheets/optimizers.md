---
title: Optimizers
summary: Gradient descent variants, momentum, Adam and AdamW, learning-rate schedules, and the stability limit.
---

# Optimizers

| Optimizer | Update | Notes |
|---|---|---|
| Gradient descent | $\theta \leftarrow \theta - \eta g$ | full-batch gradient |
| SGD / mini-batch | same, $g$ from a batch | standard; noise helps generalization |
| Momentum | $v \leftarrow \beta v + g$; $\theta \leftarrow \theta - \eta v$ | accelerates consistent directions, damps zig-zags |
| Nesterov | momentum with look-ahead gradient | slightly faster convergence |
| RMSProp | divide by running RMS of gradients | per-parameter step sizes |
| Adam | bias-corrected moments $\hat m/\sqrt{\hat s}$ | default for deep nets; little tuning |
| AdamW | Adam + decoupled weight decay | default for transformers |

**Stability limit:** along a direction with curvature $\lambda$, plain gradient descent needs $\eta < 2/\lambda$. The steepest direction sets the limit for all.

**Schedules:** warmup (linear ramp from ~0), then cosine or step decay. Reduce on plateau for classical setups.

**Defaults that usually work:** AdamW, lr 1e-3 (small models) to 1e-4 or lower (fine-tuning large models), weight decay 0.01, warmup 1 to 5% of steps, gradient clipping at 1.0.

**Diagnose with the loss curve:** flat (lr too low or broken graph), exploding (lr too high), noisy (batch small or lr high), validation rising (overfitting, not an optimizer problem).

Lessons: [Gradient descent](../lessons/11-gradient-descent-backprop/01-gradient-descent.md) · [Training and regularization](../lessons/12-training-regularization/01-training-and-regularization.md) · Lab: [gradient descent](https://sreshtalluri.github.io/ml-ai-learn/labs/gradient-descent/)
