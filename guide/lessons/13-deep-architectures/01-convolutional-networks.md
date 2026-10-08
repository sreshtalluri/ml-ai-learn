---
title: Convolutional networks
summary: Slide a kernel over an image to compute a feature map by hand, track output sizes with stride and padding, pool, and see why CNNs suit images.
skill: deep-learning
minutes: 35
prerequisites: [neural-network-forward-pass, training-and-regularization]
related: [recurrent-networks, autoencoders-diffusion-and-transfer, transformer-architecture]
---

# Convolutional networks

> **Mental model.** A convolution slides a small pattern detector (a kernel) across an image and records how strongly each location matches. The same detector is reused everywhere, so a CNN learns "what an edge looks like" once and finds it anywhere.

**You will learn to**
- Compute a convolution output cell by hand and a full feature map.
- Compute output sizes from input size, kernel size, stride, and padding.
- Apply max pooling and explain why it is used.
- Count parameters in a convolutional layer and compare with a dense layer.
- Explain the inductive biases (locality, weight sharing) that make CNNs data-efficient on images.

**Why it matters.** CNNs made modern computer vision work and remain strong, efficient backbones for images, audio spectrograms, and some sequence tasks. The ideas of local receptive fields and weight sharing reappear across deep learning.

## 1. Intuition

A dense layer connects every pixel to every unit: a 224×224 color image has 150,528 inputs, so each hidden unit needs 150,528 weights, and the layer ignores that nearby pixels are related.

A convolutional layer instead uses small kernels, say 3×3. Each kernel looks at one 3×3 patch at a time, computes a weighted sum, then slides over. The result is a **feature map** showing where the kernel's pattern appears. A layer has many kernels (channels), each learning a different pattern. Early layers learn edges and colors; deeper layers combine them into textures, parts, and objects.

**Stride** is how far the kernel moves each step (stride 2 halves the resolution). **Padding** adds zeros around the border so the output can keep the input's size. **Pooling** downsamples by taking the maximum (or average) of small windows, keeping the strongest responses and adding a little tolerance to small shifts.

## 2. Visualization

![A 5 by 5 image whose left three columns are 0 and right two columns are 9, with the top-left 3 by 3 window highlighted. A 3 by 3 vertical-edge kernel with columns −1, 0, 1. The 3 by 3 feature map: 0 in the first column and 27 in the others. A max-pooled 2 by 2 output of a separate 4 by 4 map.](../../figures/convolutional-networks.png)

*The kernel responds strongly (27) wherever a dark-to-bright vertical edge falls inside its window and gives 0 over flat regions.*

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $X$ | input of size $H \times W$ (one channel) |
| $K$ | kernel of size $k \times k$ |
| $s$, $p$ | stride and padding |
| $C_{\text{in}}, C_{\text{out}}$ | input and output channels |
| $Y$ | output feature map |

### Convolution (as deep-learning libraries compute it: cross-correlation)

```math
Y_{ij} = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} X_{\,i s + u,\; j s + v}\; K_{uv} \;+\; b
```

### Output size

```math
H_{\text{out}} = \left\lfloor \frac{H - k + 2p}{s} \right\rfloor + 1
```

### Parameters

A conv layer has $C_{\text{out}} \times (C_{\text{in}} \times k \times k + 1)$ parameters, independent of image size.

### Worked example 1: one output cell

Input (5×5): columns 1 to 3 are 0, columns 4 to 5 are 9. Kernel rows are all $[-1, 0, 1]$.

- **Top-left cell:** the window is all zeros, so $Y_{00} = 0$.
- **Top-middle cell:** the window covers columns 2 to 4, each row $[0, 0, 9]$. Per row: $(-1)(0) + (0)(0) + (1)(9) = 9$. Three rows: $Y_{01} = 27$.
- **Top-right cell:** columns 3 to 5, each row $[0, 9, 9]$: $0 + 0 + 9 = 9$ per row, so 27.

Full feature map: every row is $[0, 27, 27]$.

**Size check:** $(5 - 3 + 0)/1 + 1 = 3$. With padding 1: $(5 - 3 + 2)/1 + 1 = 5$. With stride 2 and no padding: $\lfloor 2/2 \rfloor + 1 = 2$.

### Worked example 2: max pooling

A 4×4 map with 2×2 windows and stride 2:

```math
\begin{bmatrix}1&3&2&1\\4&6&5&0\\1&2&9&8\\3&1&4&7\end{bmatrix} \;\to\; \begin{bmatrix}\max(1,3,4,6) & \max(2,1,5,0)\\ \max(1,2,3,1) & \max(9,8,4,7)\end{bmatrix} = \begin{bmatrix}6&5\\3&9\end{bmatrix}
```

### Worked example 3: parameter count

A layer with 64 input channels, 128 output channels, and 3×3 kernels: $128 \times (64 \times 9 + 1) = 128 \times 577 = 73{,}856$ parameters. A dense layer from a $32 \times 32 \times 64$ input to an output of the same size would need over 4 billion weights.

## 4. Implementation

```python
import numpy as np

def conv2d(x, k, stride=1, pad=0):
    x = np.pad(x, pad)
    kh, kw = k.shape
    oh, ow = (x.shape[0] - kh) // stride + 1, (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i*stride:i*stride+kh, j*stride:j*stride+kw] * k)
    return out
```

A small image classifier in PyTorch:

```python
import torch
from torch import nn

cnn = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # [B,3,32,32] -> [B,32,16,16]
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> [B,64,8,8]
    nn.Flatten(), nn.Linear(64 * 8 * 8, 10),                                  # -> [B,10] logits
)
print(cnn(torch.randn(4, 3, 32, 32)).shape)   # torch.Size([4, 10])
```

Runnable script (the feature map, size formula, pooling, and figure): [`code/13-deep-architectures/architectures.py`](../../code/13-deep-architectures/architectures.py).

## 5. Engineering

**Use pretrained models.** For most image tasks, start from a pretrained backbone (ResNet, EfficientNet, ConvNeXt, or a vision transformer) and fine-tune; training from scratch needs far more data (see [transfer learning](03-autoencoders-diffusion-and-transfer.md)).

**Augmentation** (random crops, flips, color jitter) is essential for CNNs on small datasets, as long as it preserves the label.

**Architecture pieces.** Batch normalization after convolutions; residual connections for depth; global average pooling instead of large dense heads; 1×1 convolutions to mix channels cheaply.

**CNNs versus vision transformers.** CNNs' built-in locality makes them data-efficient and fast; vision transformers scale well with very large data. Hybrids are common.

> [!WARNING]
> **Failure modes.** Off-by-one size errors after stride and padding; augmentations that change labels; shortcuts (the model detects a watermark or background instead of the object); distribution shift between training photos and production cameras.

### Common mistakes

- Forgetting that channels are a dimension: inputs are $[B, C, H, W]$ in PyTorch.
- Computing the flattened size by hand wrong; run a dummy tensor through and print shapes.
- Training a large CNN from scratch on a few thousand images.

## 6. Knowledge check

<!-- quiz:convolutional-networks -->
**[Take the CNN quiz](../../quizzes/convolutional-networks.md)**
<!-- /quiz -->

**Practice exercise.** A 32×32 input passes through a 5×5 convolution with stride 1 and no padding, then 2×2 max pooling with stride 2. What is the output size?

<details>
<summary>Solution</summary>

Convolution: $(32 - 5)/1 + 1 = 28$, so 28×28. Pooling: $28/2 = 14$, so 14×14.
</details>

**Implementation challenge.** Apply the vertical-edge kernel and its transpose (a horizontal-edge kernel) to a real grayscale photo with your `conv2d`, display both feature maps, and compare with `scipy.signal.correlate2d`.

## Summary

- A convolution slides a shared kernel over the input; each output cell is a patch-kernel dot product plus bias.
- Output size $= \lfloor (H - k + 2p)/s \rfloor + 1$; parameters don't depend on image size.
- Pooling downsamples and adds small-shift tolerance.
- Locality and weight sharing make CNNs efficient for images; in practice, fine-tune pretrained backbones.

**Next:** [Recurrent networks](02-recurrent-networks.md)

**Related:** [Training and regularization](../12-training-regularization/01-training-and-regularization.md) · [Model card: CNN](../../models/cnn.md)
