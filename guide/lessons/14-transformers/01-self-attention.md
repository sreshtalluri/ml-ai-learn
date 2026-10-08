---
title: Self-attention
summary: Compute scaled dot-product attention by hand, track the shape of Q, K, V and every intermediate matrix, and explain scaling, masking, and multiple heads.
skill: transformers
minutes: 45
prerequisites: [vectors-and-matrices, neural-network-forward-pass, word-embeddings]
related: [transformer-architecture, tokenization-and-pretraining, recurrent-networks]
---

# Self-attention

> **Mental model.** Attention is a soft, learned lookup. Each token asks a question (its query), every token advertises what it contains (its key), and the asking token receives a blend of everyone's content (their values), weighted by how well each key matches the question.

**You will learn to**
- Explain queries, keys, and values first without equations, then with them.
- Compute the full attention output for a small example by hand.
- Write the shape of every matrix in $\text{softmax}(QK^\top/\sqrt{d_k})V$.
- Explain why scores are divided by $\sqrt{d_k}$ and why decoders use a causal mask.
- Describe what multi-head attention adds and what attention costs as sequences grow.

**Why it matters.** Self-attention is the core operation of the transformer, and therefore of every modern LLM, embedding model, and most vision and speech models. It replaced recurrence because it lets every position look at every other position directly, and because the whole sequence can be processed in parallel during training.

## 1. Intuition

Read the sentence "The animal didn't cross the street because **it** was too tired." To understand "it," you look back and decide that "animal" is the relevant word, not "street." Attention gives a model a mechanism for exactly that: when building the new representation of "it," draw heavily on "animal."

Each token's embedding is turned into three different vectors by three learned linear maps:

- **Query (Q):** what this token is looking for. ("I'm a pronoun; I need my referent.")
- **Key (K):** what this token offers to others who are searching. ("I'm a noun, singular, animate.")
- **Value (V):** the information this token actually passes on if chosen.

For each token, compare its query with every key (a dot product: large when they point the same way). Turn those similarity scores into weights that sum to 1 with a softmax. The token's output is the weighted average of all the values.

Compare with a recurrent network, which reads left to right and squeezes everything so far into one hidden state. In attention, "it" can reach "animal" in a single step, however far apart they are, and all positions are computed at once.

## 2. Visualization

<!-- lab:attention -->
![Left: a 6 by 6 attention heatmap for "the cat sat on the mat" where every token attends to every other. Middle: the same with a causal mask, so the upper triangle is zero and the first token attends only to itself. Right: for random 512-dimensional vectors, softmax of unscaled dot products puts nearly all weight on one key, while scaling by the square root of d_k spreads it out.](../../figures/self-attention.png)

*Synthetic: random embeddings and projections, so the weights show the mechanics, not linguistic meaning. In a trained model the patterns become interpretable (pronouns attending to referents, verbs to subjects).*

*Interactive version: type a sentence and step through Q, K, V, $QK^\top$, scaling, the causal mask, softmax, and the weighted sum, with tensor shapes at every stage. It includes the worked example below. [Open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/attention/).*
<!-- /lab -->

What to notice:
- Every row of an attention matrix sums to 1: each query distributes one unit of attention.
- With the causal mask, row $i$ has zeros after column $i$. The first token can only attend to itself, so its weight is exactly 1.
- Without scaling, large dot products make softmax nearly one-hot. Gradients through a saturated softmax are tiny, which stalls training.

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $n$ | sequence length (number of tokens) | scalar |
| $d_{\text{model}}$ | embedding width | scalar |
| $X$ | token embeddings (plus positional encodings), one row per token | $[n, d_{\text{model}}]$ |
| $W_Q, W_K, W_V$ | learned projection matrices | $[d_{\text{model}}, d_k]$, $[d_{\text{model}}, d_k]$, $[d_{\text{model}}, d_v]$ |
| $Q = XW_Q$ | queries | $[n, d_k]$ |
| $K = XW_K$ | keys | $[n, d_k]$ |
| $V = XW_V$ | values | $[n, d_v]$ |
| $S = QK^\top$ | raw scores: $S_{ij} = q_i \cdot k_j$ | $[n, n]$ |
| $A$ | attention weights, each row a probability distribution | $[n, n]$ |
| $M$ | mask: 0 where allowed, $-\infty$ where forbidden | $[n, n]$ |

### Scaled dot-product attention

```math
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
```

with the softmax applied to each row separately:

```math
A_{ij} = \frac{\exp(S'_{ij})}{\sum_{l} \exp(S'_{il})}, \qquad \text{output}_i = \sum_j A_{ij}\, v_j
```

### Why divide by the square root of d_k

If the entries of $q$ and $k$ are independent with mean 0 and variance 1, then $q \cdot k = \sum_{m=1}^{d_k} q_m k_m$ is a sum of $d_k$ terms each with variance 1, so its variance is $d_k$ and its standard deviation is $\sqrt{d_k}$. The script measures 2.00, 7.98, and 22.42 for $d_k = 4, 64, 512$, against $\sqrt{d_k} = 2, 8, 22.6$. Dividing by $\sqrt{d_k}$ brings scores back to unit scale, so softmax stays in its responsive range.

### Causal masking

A decoder that predicts the next token must not see future tokens during training. Setting $M_{ij} = -\infty$ for $j > i$ makes $\exp(-\infty) = 0$, so token $i$ attends only to positions $1..i$. Encoders (BERT-style) omit the mask and see the whole sequence in both directions.

### Multi-head attention

Run $h$ attention operations in parallel, each with its own smaller projections ($d_k = d_{\text{model}}/h$), concatenate the $h$ outputs, and mix them with an output matrix $W_O$:

```math
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
```

Each head can specialize (one tracks the previous token, another a syntactic relation, another a coreference) at about the same total cost as one full-width head. With $d_{\text{model}} = 512$ and $h = 8$, each head has $d_k = 64$. $W_Q$, $W_K$, $W_V$, and $W_O$ together hold $4 \times 512^2 = 1{,}048{,}576$ parameters per layer (ignoring biases).

### Worked example

One query and two keys, $d_k = 2$:

```math
q = [1, 0],\quad k_1 = [1, 0],\quad k_2 = [0, 1],\quad v_1 = [10, 0],\quad v_2 = [0, 6]
```

**Step 1: dot-product scores.** $q \cdot k_1 = 1 \times 1 + 0 \times 0 = 1$ and $q \cdot k_2 = 1 \times 0 + 0 \times 1 = 0$.

**Step 2: scale.** $\sqrt{d_k} = \sqrt{2} = 1.4142$, so the scaled scores are $[1/1.4142,\; 0] = [0.7071,\; 0]$.

**Step 3: exponentiate.** $e^{0.7071} = 2.0281$ and $e^{0} = 1$. Their sum is $3.0281$.

**Step 4: normalize.** Weights are $[2.0281/3.0281,\; 1/3.0281] = [0.6698,\; 0.3302]$. They sum to 1.

**Step 5: weighted sum of values.**

```math
\text{output} = 0.6698\,[10, 0] + 0.3302\,[0, 6] = [6.698,\; 0] + [0,\; 1.981] = [6.698,\; 1.981]
```

The query matched $k_1$ better, so the output is mostly $v_1$. Notice it is still a *blend*: $v_2$ contributes a third, because a scaled score of 0.71 versus 0 is not a decisive gap. Learned projections produce sharper or softer patterns as needed.

**Shapes, for a full sequence.** With $n = 6$ tokens, $d_{\text{model}} = 8$ and $d_k = d_v = 4$: $X$ is $[6, 8]$; $W_Q$ is $[8, 4]$; $Q$, $K$, $V$ are $[6, 4]$; $QK^\top$ is $[6, 4] \times [4, 6] = [6, 6]$; $A$ is $[6, 6]$; $AV$ is $[6, 6] \times [6, 4] = [6, 4]$.

## 4. Implementation

**From scratch (NumPy):**

```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)          # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, causal=False):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                   # [n, n]
    if causal:
        future = np.triu(np.ones_like(scores, dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)
    A = softmax(scores)                               # rows sum to 1
    return A @ V, A                                   # [n, d_v], [n, n]
```

**In PyTorch** (batched, multi-head, fused kernels):

```python
import torch
import torch.nn.functional as F

# q, k, v: [batch, heads, n, d_head]
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# or a full multi-head layer with projections:
mha = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
out, weights = mha(x, x, x, need_weights=True)   # self-attention: Q, K, V all come from x
```

Runnable script (the worked example, masked self-attention over a toy sentence, the $\sqrt{d_k}$ measurement, and the figure): [`code/14-transformers/self_attention.py`](../../code/14-transformers/self_attention.py).

## 5. Engineering

**Cost.** Computing $QK^\top$ and $AV$ takes $O(n^2 d)$ time, and storing $A$ takes $O(n^2)$ memory per head per layer. Doubling the context length quadruples attention cost. That is why long context is expensive, and why techniques like FlashAttention (which never materializes the full $n \times n$ matrix in slow memory), sliding-window attention, and grouped-query attention exist.

**KV caching at inference.** When generating token by token, the keys and values of earlier tokens don't change (thanks to the causal mask). Caching them means each new token computes only its own query against the cached keys: $O(n)$ per step instead of recomputing the whole prefix. The cache grows with context length and batch size and is often the main GPU-memory cost of serving LLMs.

**Position information.** Attention by itself is permutation-invariant: shuffle the tokens and each output is the same weighted sum, just reordered. Order has to be injected through positional encodings (sinusoidal, learned, or rotary embeddings, RoPE, applied to Q and K).

**Interpreting attention.** Heatmaps are useful for debugging and intuition, but attention weights are not a faithful explanation of a model's decision: there are many heads and layers, and values and the feed-forward layers transform information too.

> [!WARNING]
> **Failure modes.** Forgetting the causal mask during training lets the model "cheat" by seeing the answer, giving excellent training loss and useless generation. Padding tokens must be masked too, or real tokens will attend to padding. Softmax over very large unscaled scores saturates. Memory blows up quadratically with long inputs.

### Common mistakes

- Dividing by $d_k$ instead of $\sqrt{d_k}$, or forgetting the scaling entirely.
- Applying softmax over the wrong axis (columns instead of rows).
- Masking with a large negative number that overflows in half precision; use the framework's mask support.
- Assuming the attention matrix size depends on $d_k$. It is $[n, n]$ regardless.
- Reading a single head's heatmap as "what the model thinks."

## 6. Knowledge check

<!-- quiz:self-attention -->
**[Take the self-attention quiz](../../quizzes/self-attention.md)**: compute weights and outputs, check shapes, and reason about masking and cost.
<!-- /quiz -->

**Practice exercise.** Change the query in the worked example to $q = [0, 2]$, keeping the keys and values. Compute the weights and the output.

<details>
<summary>Solution</summary>

Scores: $q \cdot k_1 = 0$, $q \cdot k_2 = 2$. Scaled: $[0,\; 2/1.4142] = [0,\; 1.4142]$.
Exponentials: $1$ and $e^{1.4142} = 4.1133$; sum $5.1133$. Weights $[0.1956,\; 0.8044]$.
Output $= 0.1956\,[10, 0] + 0.8044\,[0, 6] = [1.956,\; 4.827]$. Now the second value dominates.
</details>

**Implementation challenge.** Implement multi-head self-attention from scratch: project $X$ to $Q, K, V$ of width $d_{\text{model}}$, reshape to $[h, n, d_{\text{model}}/h]$, run `attention` per head, concatenate, and apply $W_O$. Compare your output with `torch.nn.MultiheadAttention` after copying in the same weights.

## Summary

- Each token produces a query, key, and value through learned projections.
- Scores are query-key dot products, scaled by $\sqrt{d_k}$, masked if causal, and softmaxed row by row into weights.
- Each output is the attention-weighted average of the values; every row of weights sums to 1.
- Multi-head attention runs several smaller attentions in parallel so heads can specialize.
- Cost grows as $O(n^2)$ in sequence length; KV caching makes generation $O(n)$ per new token.

**Next:** [The transformer architecture](02-transformer-architecture.md)

**Related:** [Tokenization and pretraining](../15-llms/01-tokenization-and-pretraining.md) · [Recurrent networks](../13-deep-architectures/02-recurrent-networks.md) · [Model card: transformer decoder](../../models/transformer-decoder.md)
