---
title: Recurrent networks, LSTMs, and GRUs
summary: Unroll an RNN through time by hand, explain why long dependencies are hard, see how LSTM and GRU gates help, and understand why transformers replaced them for language.
skill: deep-learning
minutes: 35
prerequisites: [backpropagation, neural-network-forward-pass]
related: [self-attention, convolutional-networks, backpropagation]
---

# Recurrent networks, LSTMs, and GRUs

> **Mental model.** A recurrent network reads a sequence one step at a time and keeps a running summary, its hidden state. Each step combines the new input with the previous summary using the same weights.

**You will learn to**
- Write the RNN recurrence and unroll it for a short sequence by hand.
- Explain backpropagation through time and why gradients vanish or explode over long sequences.
- Describe the LSTM's cell state and gates, and the GRU's simplification.
- Compare RNNs with transformers on parallelism and long-range dependencies.

**Why it matters.** RNNs, LSTMs, and GRUs powered speech recognition and machine translation before transformers, and are still used for streaming and small-footprint time-series models. Their failure mode, vanishing gradients through time, motivated attention.

## 1. Intuition

To read "the cat that the dog chased ran away," you carry context forward: who is doing what. An RNN does the same with a vector $h_t$. At each step it mixes the current word's embedding with the previous $h_{t-1}$ and squashes the result. The same weight matrices are used at every step (weight sharing across time, just like a CNN shares across space).

The problem: information from early steps has to survive many multiplications to influence a late step, and so does the gradient flowing back. If the effective multiplier per step is below 1, early information fades (vanishing gradients); above 1, it blows up.

**LSTMs** add a separate **cell state**, a conveyor belt that information can ride along with little change, controlled by gates that decide what to forget, what to write, and what to output. **GRUs** merge some gates for a simpler, similarly effective design.

## 2. Visualization

![Left: relative gradient after k steps back in time on a log scale; it decays for a recurrent weight of 0.5 or 0.9 and grows for 1.1. Right: a random input sequence as bars and the RNN hidden state as a line that integrates recent inputs.](../../figures/recurrent-networks.png)

*Synthetic. With a recurrent weight of 0.9, the gradient from 40 steps back is about $0.9^{40} = 0.015$ of its original size; with 0.5 it is effectively zero after 15 steps.*

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $x_t$ | input at step $t$ | $[d_x]$ |
| $h_t$ | hidden state at step $t$ | $[d_h]$ |
| $W_x, W_h, b$ | input weights, recurrent weights, bias (shared across time) | $[d_h, d_x]$, $[d_h, d_h]$, $[d_h]$ |
| $c_t$ | LSTM cell state | $[d_h]$ |
| $f_t, i_t, o_t$ | forget, input, and output gates, each in $(0, 1)$ | $[d_h]$ |
| $\odot$ | element-wise multiplication | |

### Vanilla RNN

```math
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
```

### Backpropagation through time

Unroll the network across all steps and backpropagate. The gradient from step $t$ to step $t - k$ includes a product of $k$ Jacobians, roughly $\prod W_h^\top \operatorname{diag}(\tanh')$. Its size behaves like the $k$-th power of the recurrent weight's largest singular value times activation derivatives, which is why long dependencies vanish or explode.

### LSTM

```math
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i[h_{t-1}, x_t] + b_i), \quad o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)
```

```math
\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c), \quad c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad h_t = o_t \odot \tanh(c_t)
```

The key line is $c_t = f_t \odot c_{t-1} + \dots$: when the forget gate is near 1, the cell state (and its gradient) passes through almost unchanged, like a residual connection through time.

### Worked example: unrolling a scalar RNN

Scalar weights $w_x = 0.5$, $w_h = 0.8$, $b = 0$; inputs $x = [1, 0, 2]$; $h_0 = 0$.

| $t$ | Pre-activation $w_x x_t + w_h h_{t-1}$ | $h_t = \tanh(\cdot)$ |
|---|---|---|
| 1 | $0.5(1) + 0.8(0) = 0.5$ | $0.4621$ |
| 2 | $0.5(0) + 0.8(0.4621) = 0.3697$ | $0.3537$ |
| 3 | $0.5(2) + 0.8(0.3537) = 1.2830$ | $0.8573$ |

At $t = 2$ there is no input, but the hidden state still remembers the first input (decaying from 0.46 to 0.35).

**Gradient through time, linear simplification.** Ignoring tanh, $\partial h_3/\partial h_0 = w_h^3 = 0.8^3 = 0.512$. After 30 steps, $0.8^{30} = 0.0012$; with $w_h = 1.1$ it would be $1.1^{30} = 17.4$.

## 4. Implementation

```python
import numpy as np

def rnn_forward(xs, Wx, Wh, b, h0):
    h, hs = h0, []
    for x in xs:                         # sequential: step t needs step t-1
        h = np.tanh(Wx @ x + Wh @ h + b)
        hs.append(h)
    return np.stack(hs)                  # [T, d_h]
```

```python
import torch
from torch import nn

lstm = nn.LSTM(input_size=16, hidden_size=64, num_layers=2, batch_first=True)
out, (h_n, c_n) = lstm(torch.randn(8, 50, 16))   # out: [8, 50, 64]; h_n, c_n: [2, 8, 64]
```

Runnable script (unrolled RNN numbers and figure): [`code/13-deep-architectures/architectures.py`](../../code/13-deep-architectures/architectures.py).

## 5. Engineering

**Where RNNs still fit.** Streaming inputs where you must update a state per step with constant memory; tiny on-device models; some time-series forecasting. For tabular time series, gradient boosting on lagged features is often a stronger baseline.

**Why transformers replaced them for language.** RNNs must process tokens sequentially, so training can't be parallelized across time, and long-range information must survive many steps. Self-attention connects any two positions in one step and processes all positions in parallel during training. (Recent state-space models revisit recurrence with better long-range behavior.)

**Training tips.** Gradient clipping is standard; use LSTM or GRU cells rather than vanilla RNNs; truncated backpropagation through time limits memory for long sequences.

> [!WARNING]
> **Failure modes.** Exploding gradients (NaN losses) without clipping; forgetting information beyond a few dozen steps; slow training due to sequential computation; leakage in time-series evaluation through shuffled splits.

### Common mistakes

- Using random splits for sequential data.
- Confusing `batch_first` layouts (`[B, T, F]` versus `[T, B, F]`).
- Assuming an LSTM's hidden state remembers arbitrarily long histories.

## 6. Knowledge check

<!-- quiz:recurrent-networks -->
**[Take the recurrent networks quiz](../../quizzes/recurrent-networks.md)**
<!-- /quiz -->

**Practice exercise.** Continue the worked example one more step with $x_4 = -1$.

<details>
<summary>Solution</summary>

Pre-activation $= 0.5(-1) + 0.8(0.8573) = -0.5 + 0.6858 = 0.1858$. $h_4 = \tanh(0.1858) = 0.1837$.
</details>

**Implementation challenge.** Train a small LSTM and a vanilla RNN to remember the first token of a sequence and output it at the end, for sequence lengths 10, 50, and 100. Plot accuracy versus length for both.

## Summary

- An RNN updates a hidden state with shared weights: $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$.
- Backpropagation through time multiplies many factors, so gradients vanish or explode over long sequences.
- LSTMs and GRUs add gated paths (the cell state) that preserve information and gradients.
- Transformers replaced RNNs for language thanks to parallel training and direct long-range connections.

**Next:** [Autoencoders, diffusion, and transfer learning](03-autoencoders-diffusion-and-transfer.md)

**Related:** [Backpropagation](../11-gradient-descent-backprop/02-backpropagation.md) · [Self-attention](../14-transformers/01-self-attention.md) · [Model card: LSTM](../../models/lstm.md)
