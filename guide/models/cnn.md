---
name: Convolutional neural network
tags: [supervised, classification, vision]
lessons: [convolutional-networks, autoencoders-diffusion-and-transfer]
labs: []
---

# Convolutional neural network (CNN)

## Problem type

Supervised image classification, detection, segmentation; also audio spectrograms and some sequences.

## Input

Grid-structured tensors, e.g. images $[B, C, H, W]$.

## Output

Class logits, bounding boxes, or per-pixel labels depending on the head.

## Mental model

Slide small, shared pattern detectors across the image; stack layers so edges become textures, textures become parts, parts become objects.

## Core objective

Minimize a task loss (cross-entropy for classification) over convolution, pooling, normalization, and dense layers.

## Training process

Mini-batch SGD or AdamW with heavy data augmentation; usually fine-tuned from an ImageNet-pretrained backbone.

## Preprocessing

Resize and normalize with the pretrained model's statistics; label-preserving augmentation (crops, flips, color jitter).

## Assumptions

Local patterns matter and can appear anywhere (translation equivariance).

## Key hyperparameters

Architecture (depth, width, kernel sizes), learning rate and schedule, augmentation strength, weight decay, input resolution.

## Good use cases

Image classification and detection, especially with transfer learning; efficient on-device vision.

## Poor use cases

Tabular data; long-range dependencies without additional mechanisms.

## Strengths

Data-efficient inductive bias; parameter sharing; mature pretrained backbones; efficient inference.

## Weaknesses

Needs labeled images; can learn shortcuts (backgrounds, watermarks); sensitive to distribution shift.

## Computational cost

Per layer about $H_{\text{out}}W_{\text{out}} \times C_{\text{out}} \times C_{\text{in}} k^2$ multiply-adds.

## Evaluation metrics

Accuracy, top-5 accuracy, mAP (detection), IoU (segmentation); per-class and per-condition breakdowns.

## Failure modes

Shortcut learning; label-changing augmentations; camera or lighting shift in production.

## Minimal implementation

```python
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)   # fine-tune for your classes
```

## Compared with neighbors

- **Vision transformer:** global attention, scales well with very large data.
- **MLP:** ignores spatial structure; far more parameters for images.

## Learn more

[Convolutional networks](../lessons/13-deep-architectures/01-convolutional-networks.md)
