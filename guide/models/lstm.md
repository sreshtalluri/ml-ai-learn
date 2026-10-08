---
name: LSTM
tags: [supervised, classification, regression, nlp]
lessons: [recurrent-networks]
labs: []
---

# LSTM (long short-term memory)

## Problem type

Sequence modeling with longer dependencies than a vanilla RNN.

## Input

Sequences $[B, T, d_x]$.

## Output

Hidden states $h_t$ (and cell states $c_t$) per step.

## Mental model

An RNN with a conveyor belt (the cell state) and gates that decide what to forget, what to write, and what to reveal, so information and gradients can travel many steps.

## Core objective

Task loss with gated updates: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, $h_t = o_t \odot \tanh(c_t)$, where $f, i, o$ are sigmoid gates.

## Training process

Backpropagation through time with gradient clipping; forget-gate bias often initialized to 1.

## Preprocessing

Scaling, padding with masks, time-ordered splits.

## Assumptions

Sequential structure; dependencies up to hundreds of steps.

## Key hyperparameters

Hidden size, layers, dropout, learning rate, clipping, sequence length.

## Good use cases

Time series, streaming signals, speech and handwriting (historically), small on-device sequence models.

## Poor use cases

Very long contexts and large-scale language tasks, where transformers dominate.

## Strengths

Much better long-range memory than vanilla RNNs; streaming-friendly.

## Weaknesses

Sequential training; about 4× the parameters of a vanilla RNN of the same width; still limited memory.

## Computational cost

About $4 \times$ a vanilla RNN per step.

## Evaluation metrics

Task metrics on time-ordered validation sets.

## Failure modes

Leakage from shuffled splits; exploding gradients; underestimating how far back it can actually remember.

## Minimal implementation

```python
import torch.nn as nn
lstm = nn.LSTM(input_size=16, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True)
```

## Compared with neighbors

- **GRU:** fewer gates and parameters, similar performance.
- **Transformer:** parallel and stronger for long-range dependencies.

## Learn more

[Recurrent networks, LSTMs, and GRUs](../lessons/13-deep-architectures/02-recurrent-networks.md)
