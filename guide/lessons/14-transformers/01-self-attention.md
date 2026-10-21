---
title: Self-attention
summary: Compute scaled dot-product attention by hand, track the shape of Q, K, V and every intermediate matrix, and explain scaling, masking, and multiple heads.
skill: transformers
minutes: 45
prerequisites: [vectors-and-matrices, neural-network-forward-pass, word-embeddings]
related: [transformer-architecture, tokenization-and-pretraining, recurrent-networks]
video: { id: self-attention-explained, minutes: 2 }
adapted_from: [phases/05-nlp-foundations-to-advanced/10-attention-mechanism, phases/07-transformers-deep-dive/01-why-transformers, phases/07-transformers-deep-dive/02-self-attention-from-scratch, phases/07-transformers-deep-dive/03-multi-head-attention, phases/19-capstone-projects/33-multihead-self-attention]
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

> [!NOTE]
> Attention is older than the transformer. It was added to RNN translation models in 2014 (Bahdanau et al.): at each output step the decoder scored every encoder hidden state, took a softmax, and read a weighted average of them, instead of relying on one fixed summary vector. That version scored with a small neural network ("additive" attention); Luong et al. (2015) showed a plain dot product works as well and is cheaper. The 2017 transformer kept the dot product, added the $\sqrt{d_k}$ scale, and removed the recurrence entirely, so a sequence attends to *itself*.

## 2. Visualization

<!-- lab:attention -->
![Left: a 6 by 6 attention heatmap for "the cat sat on the mat" where every token attends to every other. Middle: the same with a causal mask, so the upper triangle is zero and the first token attends only to itself. Right: for random 512-dimensional vectors, softmax of unscaled dot products puts nearly all weight on one key, while scaling by the square root of d_k spreads it out.](../../figures/self-attention.png)

*Synthetic: random embeddings and projections, so the weights show the mechanics, not linguistic meaning. In a trained model the patterns become interpretable (pronouns attending to referents, verbs to subjects).*

*Interactive version: type a sentence and step through Q, K, V, $QK^\top$, scaling, the causal mask, softmax, and the weighted sum, with tensor shapes at every stage. It includes the worked example below. [Open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/attention/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. In the course example, predict which value the output mostly copies before stepping to the end.
2. Turn off the causal mask. Which cells of the heatmap change, and why must a decoder keep them masked?

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

The full script is [`code/14-transformers/attention_scratch.py`](../../code/14-transformers/attention_scratch.py) (NumPy only). It builds attention in four pieces, then checks them against the worked example above.

### Build it from scratch

**Step 1: a softmax that cannot overflow.** Attention turns scores into weights with a softmax. `np.exp(1000)` is infinity, so subtract each row's maximum first. That changes nothing mathematically, because softmax is unchanged when you add the same constant to every input.

```python
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)    # shift so the largest entry is 0
    e = np.exp(z)                              # every value now in (0, 1]
    return e / e.sum(axis=axis, keepdims=True)
```

*Predict:* what does `softmax([1000, 1001, 1002])` return? (Hint: the same as `softmax([0, 1, 2])`.)

**Step 2: scaled dot-product attention.** This is the formula from section 3, line for line. `np.swapaxes(K, -1, -2)` transposes only the last two axes, so the same function works for one head `[n, d]` or many heads `[h, n, d]`. The causal mask writes $-\infty$ above the diagonal before the softmax, never after.

```python
def attention(Q, K, V, causal=False):
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)   # [..., n, n]
    if causal:
        n = scores.shape[-1]
        future = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)       # exp(-inf) = 0
    A = softmax(scores)                                  # each row sums to 1
    return A @ V, A                                      # [..., n, d_v], [..., n, n]
```

*Predict:* with `causal=True`, what is row 0 of `A`, whatever the inputs are?

**Step 3: multi-head attention, the readable way.** Each head uses its own slice of the projection columns, so with $d_{\text{model}} = 8$ and $h = 2$ each head works in 4 dimensions. Run attention per head, concatenate, and mix with $W_O$.

```python
def mha_loop(X, Wq, Wk, Wv, Wo, h, causal=True):
    d_head = X.shape[-1] // h
    outs = []
    for i in range(h):
        cols = slice(i * d_head, (i + 1) * d_head)       # this head's slice of the projections
        out, _ = attention(X @ Wq[:, cols], X @ Wk[:, cols], X @ Wv[:, cols], causal)
        outs.append(out)                                 # [n, d_head]
    return np.concatenate(outs, axis=-1) @ Wo            # [n, d_model]
```

*Predict:* if you set `h = 8` with $d_{\text{model}} = 8$, what is each head's $d_k$, and what happens to the $\sqrt{d_k}$ scale?

**Step 4: the same thing without the loop.** Real implementations never loop over heads. They project once, reshape `[n, d_model]` into `[h, n, d_head]`, and let the head axis act like a batch axis. Merging reverses the reshape. The script checks that this gives exactly the loop's answer.

```python
def split_heads(M, h):                                   # [n, d] -> [h, n, d/h]
    n, d = M.shape
    return M.reshape(n, h, d // h).transpose(1, 0, 2)

def mha_batched(X, Wq, Wk, Wv, Wo, h, causal=True):
    Q, K, V = (split_heads(X @ W, h) for W in (Wq, Wk, Wv))
    out, A = attention(Q, K, V, causal)                  # all heads in one call
    merged = out.transpose(1, 0, 2).reshape(X.shape[0], -1)   # [n, d_model]
    return merged @ Wo, A
```

*Predict:* if you forgot the `transpose(1, 0, 2)` before the final `reshape`, would the shapes still work? Would the answer still match the loop?

### Use the library

PyTorch does all four steps in fused kernels. `F.scaled_dot_product_attention` is step 2 (and picks a FlashAttention-style kernel when it can, so the $[n, n]$ matrix is never stored in full). `nn.MultiheadAttention` is steps 3 and 4 plus the four projection matrices and their biases.

```python
import torch
import torch.nn.functional as F

# q, k, v: [batch, heads, n, d_head]
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# or a full multi-head layer with projections:
mha = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
out, weights = mha(x, x, x, need_weights=True)   # self-attention: Q, K, V all come from x
```

### Run it

```bash
uv run code/14-transformers/attention_scratch.py
```

Key lines of the output (the multi-head weights come from synthetic random matrices):

```text
== Course worked example (one query, two keys, d_k = 2) ==
weights: [0.6698 0.3302]  output: [6.6976 1.9814]
softmax([1000, 1001, 1002]) = [0.09   0.2447 0.6652]
row 0 (first token can only see itself): [1. 0. 0. 0. 0. 0.]
upper triangle all zero: True
output shape: (6, 8)  per-head weights shape: (2, 6, 6)
loop and batched agree: True
```

The first two lines reproduce the worked example in section 3. The figure script, with the masked sentence and the $\sqrt{d_k}$ measurement, is [`code/14-transformers/self_attention.py`](../../code/14-transformers/self_attention.py).

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

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Attention Mechanism — The Breakthrough](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/10-attention-mechanism), [Why Transformers — The Problems with RNNs](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/01-why-transformers), [Self-Attention from Scratch](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/02-self-attention-from-scratch), [Multi-Head Attention](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/03-multi-head-attention), [Multi-Head Self-Attention](https://aiengineeringfromscratch.com/lesson?path=phases/19-capstone-projects/33-multihead-self-attention).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Why do we scale attention scores by the square root of d_k?</strong></summary>

To keep the scores at unit scale so the softmax doesn't saturate. If the entries of $q$ and $k$ are independent with mean 0 and variance 1, then $q \cdot k = \sum_{m=1}^{d_k} q_m k_m$ is a sum of $d_k$ terms each with variance 1, so its variance is $d_k$ and its standard deviation $\sqrt{d_k}$. With $d_k = 512$, raw scores have a standard deviation around 22.6. Softmax of values that far apart is nearly one-hot: one key gets almost all the weight, and the gradient of a saturated softmax is close to zero, so the query and key projections barely learn. Dividing by $\sqrt{d_k}$ restores standard deviation 1 regardless of head size. Dividing by $d_k$ instead would over-shrink scores, making attention nearly uniform for large heads. Scaling doesn't change which key scores highest, only how peaked the distribution is.

</details>

<details>
<summary><strong>Walk me through what happens to memory when context length doubles during decoding.</strong></summary>

KV-cache memory doubles, per-token attention compute doubles, and prefill attention compute roughly quadruples. During decoding each new token attends to the keys and values of all earlier tokens, which are cached. The cache holds $2 \times \text{layers} \times \text{KV heads} \times d_{\text{head}} \times \text{bytes}$ per token. For 32 layers, 32 KV heads, $d_{\text{head}} = 128$, in fp16: $2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ bytes, 0.5 MiB per token. At 4,096 tokens that is 2 GiB per sequence; at 8,192 it is 4 GiB, multiplied by batch size. That is why the cache, not the weights, limits batch size at long contexts. Prefill builds $n \times n$ scores, so it scales as $O(n^2)$, though FlashAttention avoids storing that matrix. Grouped-query attention with 8 KV heads cuts the cache 4x, and paged attention reduces fragmentation waste.

</details>

<details>
<summary><strong>Your decoder-only model reaches a suspiciously low training loss but generates garbage. What do you check?</strong></summary>

First suspect is the causal mask: if it is missing or wrong, position $i$ can attend to position $i+1$, which is the token it is supposed to predict. The model learns to copy the answer, training loss collapses, and at generation time, when the future doesn't exist, it has learned nothing useful. Test it directly: change a future token and check that logits at earlier positions don't move, or inspect an attention matrix and confirm the upper triangle is exactly zero. Second, check the target shift: logits at position $t$ must be scored against token $t + 1$, not token $t$. Third, padding: padding tokens must be masked as keys and excluded from the loss. Fourth, inference parity: the generation loop must use the same tokenizer, position handling, and mask as training, and the KV-cache path should match a full recompute on a short prompt.

</details>

<details>
<summary><strong>What does multi-head attention buy you, and why do modern LLMs use grouped-query attention instead of full multi-head attention?</strong></summary>

Multi-head attention runs $h$ smaller attentions in parallel, each with $d_k = d_{\text{model}}/h$, so different heads can track different relations (previous token, syntax, coreference) at about the cost of one full-width head. With $d_{\text{model}} = 512$ and $h = 8$, each head has $d_k = 64$. The cost appears at inference: every head has its own keys and values in the KV cache, and decoding is usually limited by memory bandwidth for reading that cache. Grouped-query attention keeps all query heads but shares each key/value head across a group of query heads. With 32 query heads and 8 KV heads, the cache shrinks 4x, which allows bigger batches and longer contexts with a small quality cost. Multi-query attention is the extreme case, a single KV head, with the largest savings and a larger quality risk. GQA is the common middle ground.

</details>
