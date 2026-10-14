---
title: Training at scale
summary: "Train models too big or too slow for one GPU: mixed precision and clipping, gradient checkpointing, collective communication, data/tensor/pipeline parallelism, and ZeRO/FSDP sharding."
skill: deep-learning
---

# Module 27: Training at scale

A training run is limited by memory, then by communication. This module builds each technique that pushes those limits from scratch, and keeps one set of numbers throughout: a synthetic 7B-style model needs 16 bytes per parameter with mixed-precision AdamW, 108.63 GB in total, and every lesson shows a different way to make that fit.

| Lever | What it saves | What it costs |
|---|---|---|
| Mixed precision (bf16 / fp16) | half the weight and activation bytes, faster matmuls | range or precision limits; fp16 needs loss scaling |
| Gradient clipping | runs that would diverge on one bad batch | nothing, unless it clips every step |
| Gradient checkpointing | most activation memory | about 33% more compute per step |
| Data parallelism | wall-clock time | one gradient all-reduce per step; no memory saving |
| Tensor / pipeline parallelism | weight memory per GPU | per-layer communication / pipeline bubble |
| ZeRO / FSDP | duplicate optimizer state, gradients, weights | up to 1.5 times DDP's communication (stage 3) |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Gradient clipping and mixed precision](01-gradient-clipping-amp.md) | clip by global norm, choose bf16 or fp16, and run and debug a loss scaler |
| 2 | [Gradient checkpointing](02-gradient-checkpointing.md) | size activation memory and trade it for recompute |
| 3 | [Collective operations from scratch](03-collective-ops-from-scratch.md) | build ring all-reduce and cost any collective |
| 4 | [Scaling and distributed training](04-scaling-distributed.md) | size training memory and combine data, tensor, and pipeline parallelism |
| 5 | [ZeRO and FSDP](05-zero-parameter-sharding.md) | compute per-GPU memory and communication for DDP and ZeRO stages 1 to 3 |

**Interactive lab:** [Data parallelism and ZeRO memory](https://sreshtalluri.github.io/ml-ai-learn/labs/zero-sharding/)
