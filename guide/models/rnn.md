---
name: Recurrent neural network
tags: [supervised, classification, regression, nlp]
lessons: [recurrent-networks]
labs: []
---

# Recurrent neural network (RNN)

## Problem type

Sequence modeling: classification, forecasting, tagging, generation.

## Input

Sequences of vectors $[B, T, d_x]$.

## Output

A hidden state per step; predictions from the last or every step.

## Mental model

Read one step at a time and keep a running summary (the hidden state), updated with the same weights at every step.

## Core objective

Task loss over outputs, with $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$; trained by backpropagation through time.

## Training process

Unroll over the sequence, backpropagate, clip gradients; truncated BPTT for long sequences.

## Preprocessing

Scale inputs; pad and mask variable-length sequences; time-ordered splits.

## Assumptions

Relevant information can be carried in a fixed-size state; dependencies are not too long.

## Key hyperparameters

Hidden size, layers, learning rate, gradient clipping threshold, sequence length.

## Good use cases

Short sequences; streaming with constant memory; teaching sequence modeling.

## Poor use cases

Long-range dependencies (vanishing gradients); large-scale language modeling.

## Strengths

Constant memory per step; natural for streaming; few parameters.

## Weaknesses

Vanishing/exploding gradients; sequential (slow) training; limited memory span.

## Computational cost

$O(T \cdot d_h(d_x + d_h))$ per sequence, sequential in $T$.

## Evaluation metrics

Task metrics on time-ordered validation data.

## Failure modes

NaN losses without clipping; forgetting early inputs; leakage from shuffled time-series splits.

## Minimal implementation

```python
import torch.nn as nn
rnn = nn.RNN(input_size=16, hidden_size=64, batch_first=True)
```

## Compared with neighbors

- **LSTM / GRU:** gated variants that remember longer.
- **Transformer:** parallel training and direct long-range attention.

## Learn more

[Recurrent networks, LSTMs, and GRUs](../lessons/13-deep-architectures/02-recurrent-networks.md)
