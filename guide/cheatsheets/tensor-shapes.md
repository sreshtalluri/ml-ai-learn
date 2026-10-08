---
title: Tensor shapes
summary: The shapes you will see in every model, and the bugs that come from losing track of them.
---

# Tensor shapes

**Rule:** $[m, n] \times [n, p] \to [m, p]$. Inner dimensions match and disappear.

| Operation | Shapes |
|---|---|
| Linear layer (PyTorch) | `x [B, d_in]` @ `W.T [d_in, d_out]` + `b [d_out]` → `[B, d_out]` |
| MLP 20 → 128 → 3 | `[B, 20]` → `[B, 128]` → `[B, 3]` |
| Embedding lookup | IDs `[B, T]` → `[B, T, d]` via table `[V, d]` |
| Image batch (PyTorch) | `[B, C, H, W]` |
| Conv2d, k×k, stride s, pad p | `[B, C_in, H, W]` → `[B, C_out, ⌊(H−k+2p)/s⌋+1, …]` |
| RNN / LSTM (batch_first) | `[B, T, d_x]` → `[B, T, d_h]`; final state `[layers, B, d_h]` |
| Self-attention, one head | `X [n, d]` → `Q, K, V [n, d_k]` → scores `[n, n]` → output `[n, d_k]` |
| Multi-head attention | `[B, h, n, d_k]` per head; concat → `[B, n, d]` |
| Transformer block | `[B, n, d]` → `[B, n, d]` (shape preserved) |
| LM logits | `[B, T, V]`; targets `[B, T]` shifted by one |
| Cross-entropy input (PyTorch) | logits `[N, C]`, targets `[N]` (class indices) |
| Embedding index | `[num_chunks, d]`; query `[d]`; scores `[num_chunks]` |

**Classic bugs:**
- `[n] - [n, 1]` broadcasts to `[n, n]`: silent wrong loss. Flatten first.
- Transposing to "fix" a shape error and silently changing the meaning.
- Forgetting the batch dimension at inference (`[d]` instead of `[1, d]`).
- `batch_first` mismatch in RNNs: `[T, B, d]` versus `[B, T, d]`.

**Habit:** comment shapes on every line and run a dummy tensor through the model.

Lessons: [Vectors and matrices](../lessons/00-foundations/01-vectors-and-matrices.md) · [Forward pass](../lessons/10-neural-networks/01-neural-network-forward-pass.md) · [Self-attention](../lessons/14-transformers/01-self-attention.md)
