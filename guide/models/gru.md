---
name: GRU
tags: [supervised, classification, regression, nlp, low-latency]
lessons: [recurrent-networks]
labs: []
---

# GRU (gated recurrent unit)

## Problem type

Sequence modeling; a lighter alternative to the LSTM.

## Input

Sequences $[B, T, d_x]$.

## Output

A hidden state per step.

## Mental model

An LSTM simplified: an update gate decides how much of the old state to keep, and a reset gate decides how much of it to use when proposing a new state. No separate cell state.

## Core objective

Task loss with $z_t = \sigma(\cdot)$, $r_t = \sigma(\cdot)$, $\tilde{h}_t = \tanh(W x_t + U(r_t \odot h_{t-1}))$, $h_t = (1 - z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t$.

## Training process

Backpropagation through time with gradient clipping.

## Preprocessing

Scaling, padding with masks, time-ordered splits.

## Assumptions

Sequential structure with moderate-length dependencies.

## Key hyperparameters

Hidden size, layers, dropout, learning rate, clipping.

## Good use cases

Streaming and on-device sequence models; small time-series tasks.

## Poor use cases

Long-context language tasks.

## Strengths

Fewer parameters than LSTM (3 gate blocks versus 4); fast; good memory for moderate lengths.

## Weaknesses

Sequential training; limited long-range memory compared with attention.

## Computational cost

About $3 \times$ a vanilla RNN per step.

## Evaluation metrics

Task metrics on time-ordered validation data.

## Failure modes

Same as LSTMs: leakage, exploding gradients, overestimated memory.

## Minimal implementation

```python
import torch.nn as nn
gru = nn.GRU(input_size=16, hidden_size=64, batch_first=True)
```

## Compared with neighbors

- **LSTM:** an extra cell state and gate; sometimes slightly better on long dependencies.
- **Transformer:** better for long contexts, more compute per token.

## Learn more

[Recurrent networks, LSTMs, and GRUs](../lessons/13-deep-architectures/02-recurrent-networks.md)
