---
title: Transformer architecture
summary: Attention, blocks, shapes, families, and costs on one page.
---

# Transformer architecture

**Attention:** $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$, with $Q = XW_Q$, $K = XW_K$, $V = XW_V$.

| Stage | Purpose |
|---|---|
| Token + position embeddings | identity and order |
| Layer normalization | stable activations (pre-norm in modern models) |
| Multi-head self-attention | mix information across positions |
| Residual connection | preserve signal, help gradients |
| Feed-forward network ($d \to 4d \to d$) | transform each position; most parameters |
| Final norm + output head | logits over the vocabulary |

**Shapes** (sequence length $n$, width $d$, $h$ heads, $d_k = d/h$):

| Tensor | Shape |
|---|---|
| $X$ | $[n, d]$ |
| $Q, K, V$ per head | $[n, d_k]$ |
| $QK^\top$, attention weights | $[n, n]$ per head |
| Block output | $[n, d]$ |

**Parameters per block:** attention $4d^2 + 4d$; FFN $2 d\,d_{\text{ff}} + d_{\text{ff}} + d$; two layer norms $4d$. GPT-2 small ($d=768$, 12 layers): 124M total.

| Family | Mask | Use |
|---|---|---|
| Encoder-only | none (bidirectional) | classification, embeddings, rerankers |
| Decoder-only | causal | generation, chat LLMs |
| Encoder-decoder | causal decoder + cross-attention | translation, summarization |

**Costs:** attention $O(n^2 d)$ per layer; KV cache grows with $n$ × layers × width; generation with a KV cache costs $O(n)$ per new token.

Lessons: [Self-attention](../lessons/14-transformers/01-self-attention.md) · [The transformer architecture](../lessons/14-transformers/02-transformer-architecture.md) · Lab: [attention](https://sreshtalluri.github.io/ml-ai-learn/labs/attention/)
