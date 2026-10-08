---
name: Multilayer perceptron
tags: [supervised, classification, regression]
lessons: [neural-network-forward-pass, backpropagation, training-and-regularization]
labs: [nn-forward, backprop]
---

# Multilayer perceptron (MLP)

## Problem type

Supervised classification or regression; also a building block (prediction heads, transformer feed-forward layers).

## Input

Fixed-size numeric vectors (scaled), embeddings, or features from another network.

## Output

Logits (classification) or real values (regression).

## Mental model

A stack of layers that each mix their inputs with weights, add a bias, and bend the result with a nonlinearity, building more useful representations layer by layer.

## Core objective

Minimize a task loss (cross-entropy or MSE) plus regularization over the network $f(x) = W_L\,\phi(\dots\phi(W_1x + b_1)\dots) + b_L$.

## Training process

Mini-batch gradient descent (Adam/AdamW or SGD with momentum) with backpropagation; early stopping on validation loss.

## Preprocessing

Standardize numeric features; embed or one-hot categoricals.

## Assumptions

Enough data to fit the parameters; features are informative without strong spatial or sequential structure.

## Key hyperparameters

Depth and width; activation; learning rate and schedule; batch size; weight decay; dropout; epochs.

## Good use cases

Heads on top of pretrained encoders; medium-size tabular data with embeddings for categoricals; function approximation.

## Poor use cases

Small tabular datasets (boosting usually wins); images, sequences, or text without specialized architectures.

## Strengths

Universal approximator; flexible; GPU-friendly; composes with any architecture.

## Weaknesses

Data-hungry; many hyperparameters; less interpretable; sensitive to scaling and initialization.

## Computational cost

About $2 \times$ parameters multiply-adds per example per forward pass; backward is about twice that.

## Evaluation metrics

Validation loss curves; task metrics (accuracy, F1, RMSE).

## Failure modes

Overfitting; dead ReLUs; vanishing or exploding gradients; forgetting `model.eval()` at inference.

## Minimal implementation

```python
import torch.nn as nn
mlp = nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
```

## Compared with neighbors

- **Logistic regression:** an MLP with no hidden layer.
- **Gradient boosting:** usually stronger on small and medium tabular data.
- **CNN / transformer:** add inductive biases for images and sequences.

## Learn more

[The forward pass](../lessons/10-neural-networks/01-neural-network-forward-pass.md) · [Backpropagation](../lessons/11-gradient-descent-backprop/02-backpropagation.md)
