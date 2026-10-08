---
name: Diffusion model
tags: [self-supervised, generative, vision]
lessons: [autoencoders-diffusion-and-transfer]
labs: []
---

# Diffusion model

## Problem type

Generative modeling: images, audio, video, and other continuous data, often conditioned on text.

## Input

Training: clean samples. Generation: random noise plus optional conditioning (a text prompt).

## Output

A generated sample.

## Mental model

Learn to undo noise one small step at a time; then start from pure noise and denoise your way to a realistic sample.

## Core objective

```math
\min_\theta\; \mathbb{E}_{x_0, t, \epsilon}\big\lVert \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\; t)\big\rVert^2
```

## Training process

Sample a timestep and noise, corrupt a clean sample with the closed-form forward process, and train the network (often a U-Net or transformer) to predict the noise.

## Preprocessing

Normalize data to a fixed range; latent diffusion first encodes data with an autoencoder.

## Assumptions

Enough training data covering the target distribution.

## Key hyperparameters

Noise schedule, number of steps, sampler, guidance scale (for conditioning), model size.

## Good use cases

High-quality image and audio generation; inpainting; editing; data augmentation.

## Poor use cases

Low-latency generation without distillation; tasks needing exact, verifiable outputs.

## Strengths

High sample quality and diversity; stable training compared with GANs; flexible conditioning.

## Weaknesses

Slow sampling (many network evaluations); large compute; can memorize training data.

## Computational cost

Generation costs (number of steps) × (one network forward pass).

## Evaluation metrics

Human evaluation; FID and CLIP-based scores (with caveats); task-specific checks.

## Failure modes

Memorization and copyright issues; biased outputs; prompt-adherence failures; unsafe content without filters.

## Minimal implementation

```python
from diffusers import DiffusionPipeline   # pip install diffusers
pipe = DiffusionPipeline.from_pretrained(MODEL_ID)
image = pipe("a watercolor of a lighthouse", num_inference_steps=30).images[0]
```

## Compared with neighbors

- **Autoencoder / VAE:** single-pass reconstruction or sampling, lower quality.
- **Autoregressive models:** generate token by token; diffusion refines the whole sample in parallel each step.

## Learn more

[Autoencoders, diffusion, and transfer learning](../lessons/13-deep-architectures/03-autoencoders-diffusion-and-transfer.md)
