---
title: Loss functions
summary: Which loss to train with for each task, and what each one encourages.
---

# Loss functions

A **loss** is what the optimizer minimizes; a **metric** is what people judge. Pick the metric first, then a loss that pushes in the same direction.

| Task | Loss | Formula (one example) | Encourages |
|---|---|---|---|
| Regression | MSE | $(y - \hat{y})^2$ | close predictions; punishes big misses |
| Robust regression | MAE / Huber | $\lvert y - \hat{y}\rvert$ / hybrid | less outlier influence |
| Binary classification | binary cross-entropy | $-[y\ln p + (1-y)\ln(1-p)]$ | high probability on the true label |
| Multiclass | categorical cross-entropy | $-\ln p_{\text{true}}$ | high probability on the true class |
| Language modeling | token cross-entropy | $-\ln P(t_k \mid t_{<k})$ | predicting the next token |
| Margin classification | hinge | $\max(0, 1 - y f(x))$ | a margin between classes (SVM) |
| Metric learning | contrastive / triplet / InfoNCE | pull positives together, push negatives apart | similar items close in embedding space |
| Reconstruction | MSE / cross-entropy on inputs | $\lVert x - \hat{x}\rVert^2$ | autoencoders |
| Diffusion | noise-prediction MSE | $\lVert \epsilon - \epsilon_\theta\rVert^2$ | denoising |

**Implementation tips.** Pass logits, not probabilities, to `BCEWithLogitsLoss` and `CrossEntropyLoss` (numerically stable). Add regularization terms ($\lambda\lVert w\rVert^2$) to the loss, or use weight decay in the optimizer (AdamW).

**Quick numbers.** Cross-entropy for $p_{\text{true}} = 0.9, 0.5, 0.1, 0.01$: $0.105, 0.693, 2.303, 4.605$.

Lessons: [Logistic regression](../lessons/04-classification/01-logistic-regression.md) · [Forward pass](../lessons/10-neural-networks/01-neural-network-forward-pass.md) · [Tokenization and pretraining](../lessons/15-llms/01-tokenization-and-pretraining.md)
