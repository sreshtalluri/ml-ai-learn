---
name: Autoencoder
tags: [unsupervised, self-supervised, dimensionality-reduction, generative]
lessons: [autoencoders-diffusion-and-transfer, pca]
labs: []
---

# Autoencoder

## Problem type

Unsupervised representation learning, compression, denoising, anomaly detection.

## Input

Any data the encoder can process (tabular vectors, images, sequences).

## Output

A reconstruction $\hat{x}$ and a latent code $z$.

## Mental model

Squeeze data through a narrow bottleneck and learn to rebuild it, so the bottleneck has to capture what matters.

## Core objective

Minimize reconstruction loss $\lVert x - g_\theta(f_\phi(x))\rVert^2$ (or cross-entropy); VAEs add a KL term to regularize the latent space.

## Training process

Gradient descent on reconstruction loss; denoising variants corrupt inputs and reconstruct clean targets.

## Preprocessing

Scale inputs to the decoder's output range; for anomaly detection, train on normal data only.

## Assumptions

Data lies near a lower-dimensional structure.

## Key hyperparameters

Latent size, architecture depth and width, noise level (denoising), KL weight (VAE).

## Good use cases

Anomaly detection by reconstruction error; denoising; learned compression; latent spaces for diffusion models.

## Poor use cases

When a simple PCA suffices; when anomalies are easy to reconstruct.

## Strengths

Nonlinear generalization of PCA; no labels needed; flexible.

## Weaknesses

Can learn the identity map without a strong bottleneck; reconstruction error is not a calibrated anomaly probability.

## Computational cost

Roughly twice the cost of the encoder alone.

## Evaluation metrics

Reconstruction error; downstream task performance; anomaly-detection precision and recall with labeled incidents.

## Failure modes

Reconstructing anomalies too well; training data contaminated with anomalies; drift in "normal."

## Minimal implementation

```python
import torch.nn as nn
ae = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 16),     # encoder
                   nn.ReLU(), nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 784))  # decoder
```

## Compared with neighbors

- **PCA:** linear, closed form; a linear autoencoder finds the same subspace.
- **Diffusion model:** generates by iterative denoising, often in an autoencoder's latent space.

## Learn more

[Autoencoders, diffusion, and transfer learning](../lessons/13-deep-architectures/03-autoencoders-diffusion-and-transfer.md)
