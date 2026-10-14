---
title: Deep-learning architectures
summary: CNNs for grids, RNNs and LSTMs for sequences, autoencoders, GANs, latent diffusion, and flow matching for generation, and transfer learning.
skill: deep-learning
---

# Module 13: Deep-learning architectures

Architectures encode assumptions (inductive biases) about the data. Convolutions assume local patterns that repeat across an image. Recurrence assumes information flows in order. Autoencoders assume data lies near a lower-dimensional space. Choosing an architecture is choosing an assumption.

| Architecture | Inductive bias | Common uses |
|---|---|---|
| MLP | generic feature interactions | tabular data, prediction heads |
| CNN | local patterns, translation structure | images, audio spectrograms |
| RNN / LSTM / GRU | sequential hidden state | time series, streaming |
| Transformer | content-based global attention | language, vision, multimodal |
| Autoencoder | compress then reconstruct | representation learning, denoising |
| GAN | a learned critic judges realism | narrow-domain images, adversarial losses |
| Diffusion / flow matching | iterative denoising along a path from noise | image, video, and audio generation |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Convolutional networks](01-convolutional-networks.md) | compute a convolution and pooling by hand and track feature-map sizes |
| 2 | [Recurrent networks, LSTMs, and GRUs](02-recurrent-networks.md) | unroll an RNN, explain gates, and say why transformers replaced them |
| 3 | [Autoencoders, diffusion, and transfer learning](03-autoencoders-diffusion-and-transfer.md) | explain bottlenecks, latent spaces, denoising diffusion, and fine-tuning a pretrained model |
| 4 | [GANs, generator versus discriminator](04-gans-generator-discriminator.md) | train a GAN with the non-saturating loss and diagnose mode collapse |
| 5 | [Latent diffusion and Stable Diffusion](05-latent-diffusion-stable-diffusion.md) | explain latent compression, cross-attention conditioning, and classifier-free guidance |
| 6 | [Flow matching and rectified flows](06-flow-matching-rectified-flows.md) | train a velocity field on straight noise-to-data paths and straighten it with reflow |
