---
title: Attention variants
summary: Draw causal, sliding-window, strided, and global-token attention masks, count what each query can see, and size the compute, KV cache, and receptive field each one buys.
skill: transformers
minutes: 40
prerequisites: [self-attention, transformer-architecture]
related: [llm-inference, open-models-architecture-walkthroughs, bert-masked-language-modeling]
adapted_from: [phases/07-transformers-deep-dive/15-attention-variants]
---

# Attention variants

> **Mental model.** An attention mask is a seating chart. Full attention lets every guest talk to every other guest, which gets loud and expensive as the party grows. Variants redraw the chart: talk only to your neighbours (sliding window), to your neighbours plus every fourth seat (strided), or to your neighbours plus a few hosts who talk to everyone (global tokens). News still travels across the room, one table per round.

**You will learn to**
- Write the boolean mask for full, causal, sliding-window, local + strided, and global-token attention.
- Count attended cells and explain why sliding window costs $O(n \cdot w)$ instead of $O(n^2)$.
- Compute the receptive field of a stacked sliding-window model, $L(w-1)+1$.
- Size the KV cache of a model that mixes local and global layers.
- Choose a pattern for a long-context model and say what it gives up.

**Why it matters.** Full attention scores every pair of tokens, so doubling the context quadruples the score matrix. Long-context models (Gemma 2 and 3, early Mistral, Longformer, BigBird) change *which* pairs are scored. The mask is one line of code, but it decides compute, KV-cache memory, and what the model can and cannot connect.

## 1. Intuition

In [self-attention](01-self-attention.md) each query scores every key, and a mask sets some scores to $-\infty$ before the softmax so those keys get zero weight. A decoder uses the **causal** mask: token $i$ reads keys $0 \dots i$, never the future. An encoder like [BERT](03-bert-masked-language-modeling.md) uses **no** mask: every token reads every token.

Every variant in this lesson is just a different mask:

1. **Sliding window (local).** Token $i$ reads only the last $w$ tokens, itself included. Cost per query is $w$, not $i$. A sliding-window layer only needs to keep the last $w$ keys and values in its KV cache.
2. **Local + strided (sparse).** Add a few long-range links, such as every 4th earlier token, so distant information has a shortcut ([Child et al., 2019](https://arxiv.org/abs/1904.10509)).
3. **Global tokens.** Pick a handful of positions (often a `[CLS]` token) that read everything and that everyone reads. Longformer and BigBird combined a local window with global tokens ([Beltagy et al., 2020](https://arxiv.org/abs/2004.05150); [Zaheer et al., 2020](https://arxiv.org/abs/2007.14062)).
4. **Interleaving.** Most current long-context models keep some layers fully global. Gemma 2 alternated local and global layers 1:1 with a 4,096-token window; Gemma 3 uses five local layers (window 1,024) per global layer ([Gemma Team, 2025](https://arxiv.org/abs/2503.19786)).

A local mask doesn't make distant tokens invisible forever. Layer 1 lets token 15 read tokens 12 to 15. But token 12 already read 9 to 12 in the layer before, so after two layers token 15 carries information from 9 onward. Each layer extends the reach by $w - 1$ positions. The catch: information that travels many hops gets blurred, so pure local models are weak at retrieving one exact fact from far back. That is why global layers stay in the mix.

> [!NOTE]
> These are changes to *which* pairs get scored. [FlashAttention](../../glossary.md#flashattention) is different: it computes exactly the same full attention, only faster and with less memory traffic. Multi-query and grouped-query attention are different again: they share keys and values across heads to shrink the KV cache (see [LLM inference](../18-llm-inference/01-llm-inference.md)).

## 2. Visualization

<!-- lab:attention-masks -->
![Five panels. Four 16-by-16 query-by-key grids: causal (136 cells, query 12 sees 13 keys), sliding window w=4 (58 cells, query 12 sees 4), local w=2 plus stride 4 (52 cells, query 12 sees 4: keys 3, 7, 11, 12), and local radius 2 plus global token 0 (100 cells, query 12 sees 6). The fifth panel plots the receptive field of the last token against layers for windows 2, 4, and 8, rising in straight lines of slope w minus 1.](../../figures/attention-variants.png)

*Row 12 is drawn in full colour in each grid: those are the keys query 12 can read. The right panel shows how far back the last token can see after stacking sliding-window layers, on a 64-token sequence.*

*Interactive version: [open the lab on the website](https://sreshtalluri.github.io/ml-ai-learn/labs/attention-masks/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Choose **causal**, then **sliding window** with $w = 4$ on 16 tokens. Predict the attended-cell count for each before reading it. (136 and 58.)
2. Keep sliding window and raise the window to 16. What mask do you get, and why?
3. Raise the layers slider for sliding window $w = 4$. After how many layers can the last of 16 tokens see the first?
4. Switch to **global tokens** and pick query 0. Why does it see every key, while query 12 sees only 6?

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $n$ | sequence length (tokens) | scalar |
| $i, j$ | query position, key position | indices in $0 \dots n-1$ |
| $M$ | mask: $M_{ij} = 1$ if query $i$ may read key $j$ | $[n, n]$ |
| $S$ | raw scores $QK^\top/\sqrt{d_k}$ | $[n, n]$ |
| $w$ | window size (keys per query, self included) | scalar |
| $s$ | stride for long-range links | scalar |
| $L$ | number of stacked layers | scalar |

### Masked attention

```math
A = \operatorname{softmax}\big(S + \log M\big)V, \qquad \log M_{ij} = \begin{cases} 0 & M_{ij}=1 \\ -\infty & M_{ij}=0 \end{cases}
```

Each pattern is a rule for $M_{ij}$:

| Pattern | $M_{ij} = 1$ when | Cells for $n \ge w$ |
|---|---|---|
| Full (encoder) | always | $n^2$ |
| Causal | $j \le i$ | $n(n+1)/2$ |
| Sliding window | $j \le i$ and $i - j < w$ | $nw - w(w-1)/2$ |
| Local + strided | in the window, or $j \le i$ and $j \bmod s = s - 1$ | about $nw + n^2/(2s)$ |
| Global tokens $G$ | local rule, or $i \in G$, or $j \in G$ | about $n \cdot (\text{local}) + 2n\lvert G\rvert$ |

Strided attention with $s \approx \sqrt{n}$ costs $O(n\sqrt{n})$; sliding window costs $O(nw)$, linear in $n$ for fixed $w$.

**Receptive field.** One sliding-window layer lets position $i$ read back $w - 1$ positions. Stack $L$ layers and the reach adds:

```math
\text{reach}(L) = \min\big(i + 1,\; L(w-1) + 1\big)
```

### Worked example: 16 tokens

Take $n = 16$, $w = 4$.

- **Causal:** $16 \cdot 17 / 2 = 272/2 = 136$ cells, 53% of $16^2 = 256$.
- **Sliding window:** rows 0, 1, 2 have only 1, 2, 3 earlier keys, so $1 + 2 + 3 = 6$. Rows 3 to 15 are 13 rows of 4 keys: $13 \cdot 4 = 52$. Total $6 + 52 = 58$. The formula agrees: $16 \cdot 4 - 4 \cdot 3/2 = 64 - 6 = 58$.
- **Local $w = 2$ + stride 4:** long-range keys are $j \in \{3, 7, 11, 15\}$. Query 12 reads its window $\{11, 12\}$ plus $\{3, 7, 11\}$, which is $\{3, 7, 11, 12\}$: 4 keys. Summing over all 16 rows gives 52 cells.
- **Local radius 2 + global token 0:** query 12 reads $\{10, \dots, 14\}$ (5 keys) plus key 0: 6 keys. Query 0 is global and reads all 16. Total 100 cells.

**Receptive field of the last token (position 15), $w = 4$:**

| Layers $L$ | $L(w-1)+1$ | Reach (capped at 16) |
|---|---|---|
| 1 | $1 \cdot 3 + 1 = 4$ | 4 |
| 2 | $2 \cdot 3 + 1 = 7$ | 7 |
| 3 | $3 \cdot 3 + 1 = 10$ | 10 |
| 5 | $5 \cdot 3 + 1 = 16$ | 16 |

Five layers of a 4-token window connect the whole 16-token sequence. A Gemma-3-style window of 1,024 over a few dozen layers reaches tens of thousands of tokens in principle, but each hop dilutes the signal, which is why exact long-range recall still needs global layers.

### Worked example: KV cache for a long context

A synthetic 80-layer model with 8 KV heads, $d_{\text{head}} = 128$, BF16 (2 bytes), one 131,072-token sequence. KV bytes $= 2 \cdot \text{layers} \cdot n_{kv} \cdot d_{\text{head}} \cdot \text{tokens} \cdot 2$.

- All 80 layers global: $2 \cdot 80 \cdot 8 \cdot 128 \cdot 131{,}072 \cdot 2 = 42{,}949{,}672{,}960$ bytes, **42.95 GB**.
- About 5:1 local:global, so 13 global and 67 local layers with $w = 1{,}024$:
  - Global part: $42.95 \cdot 13/80 = 6.98$ GB.
  - Local part: $2 \cdot 67 \cdot 8 \cdot 128 \cdot 1{,}024 \cdot 2 = 281{,}018{,}368$ bytes, 0.28 GB.
  - Total 7.26 GB, which is $42.95 / 7.26 = 5.9\times$ smaller.

The global layers carry almost all of the cost. The ratio of global to local layers is the first number to check on a long-context model.

## 4. Implementation

### Build it from scratch

The full script is [`code/14-transformers/attention_variants.py`](../../code/14-transformers/attention_variants.py) (NumPy only). Every mask is a boolean matrix built from two index grids: `i` holds each cell's query position and `j` its key position. A rule over `i` and `j` is the whole mask.

```python
def causal(n):
    i, j = np.indices((n, n))
    return j <= i                                   # [n, n], lower triangle

def sliding_window(n, w):
    i, j = np.indices((n, n))
    return (j <= i) & (i - j < w)                   # the last w tokens, self included
```

*Predict:* what does `sliding_window(n, n)` return, and what about `sliding_window(n, 1)`? (The script asserts the first equals `causal(n)`.)

Sparse patterns are unions of simple rules. The strided mask adds every `stride`-th earlier key to the local window. Global tokens set whole rows and columns to `True`.

```python
def local_strided(n, w, stride):
    i, j = np.indices((n, n))
    return sliding_window(n, w) | ((j <= i) & (j % stride == stride - 1))

def with_global(mask, global_ids):
    m = mask.copy()
    m[:, global_ids] = True                         # everyone reads the global tokens
    m[global_ids, :] = True                         # global tokens read everyone
    return m
```

*Predict:* with `with_global` applied to a causal mask, token 0 now reads token 15. Is that still a valid decoder for generation? (No: it leaks the future. Global tokens belong in encoders, or must stay causal in a decoder.)

The receptive field is a reachability question. Treat the mask as a graph: one layer is one hop. Multiplying the mask by the current reach matrix adds one hop.

```python
def reach_after(mask, layers):
    """Which keys can influence each query after stacking `layers` copies of the mask."""
    R = np.eye(len(mask), dtype=bool)
    for _ in range(layers):
        R = (mask.astype(int) @ R.astype(int)) > 0  # one more attention hop
    return R
```

*Predict:* for a causal mask, how many hops until the last token reaches everything? (One. Causal attention is already fully connected to the past.)

The KV cache counts keys and values per layer. A sliding-window layer stores `w` tokens instead of the whole context.

```python
def kv_bytes(layers, n_kv, d_head, tokens, bytes_per=2):
    return 2 * layers * n_kv * d_head * tokens * bytes_per   # K and V
```

*Predict:* if all 80 layers used $w = 1{,}024$ at a 131,072-token context, how much smaller would the cache be than all-global? ($131{,}072 / 1{,}024 = 128\times$; the script prints 0.34 GB.)

### Use the library

PyTorch's `scaled_dot_product_attention` takes a boolean mask where `True` means "may attend", the same convention as above. For long sequences, materialising an $n \times n$ mask wastes memory, so FlexAttention (PyTorch 2.5+) takes the rule as a function and compiles a kernel that skips fully masked blocks:

```python
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

out = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.from_numpy(sliding_window(n, 4)))

def swa(b, h, q_idx, kv_idx):                       # same rule, as a function
    return (kv_idx <= q_idx) & (q_idx - kv_idx < 1024)

block_mask = create_block_mask(swa, B=None, H=None, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=block_mask)   # q, k, v: [batch, heads, n, d_head]
```

What the library adds: block-level skipping (the speed comes from never loading zero blocks, not from the mask itself), fused softmax, and GPU kernels. In Hugging Face, a model's config tells you its pattern: look for `sliding_window` and, in recent models, `layer_types` (a list such as `"sliding_attention"` and `"full_attention"` per layer).

### Run it

From `guide/`:

```bash
uv run code/14-transformers/attention_variants.py
```

```text
=== attended cells on a 16-token sequence ===
  full (bidirectional)                             256 of 256
  causal                                           136 of 256
  sliding window w=4                                58 of 256
  local w=2 + stride 4                              52 of 256
  local radius 2 + global token 0 (bidirectional)  100 of 256
...
  3 layer(s): 10 tokens   (formula L*(w-1)+1 = 10, capped at 16 -> 10)
  5 layer(s): 16 tokens   (formula L*(w-1)+1 = 16, capped at 16 -> 16)
...
  every layer full          :  42.95 GB
  every layer window 1024   :   0.34 GB
  5:1 local:global, W=1024  :   7.26 GB   (13 global + 67 local layers, 5.9x smaller)
```

The reachability matrix agrees with $L(w-1)+1$ at every depth (the script asserts it), and the KV numbers match the worked example.

## 5. Engineering

**Where each pattern fits.**

| Pattern | Compute per layer | KV cache per layer | Typical use |
|---|---|---|---|
| Full / causal | $O(n^2)$ | $n$ tokens | default; short and medium context; every global layer |
| Sliding window | $O(nw)$ | $w$ tokens | most layers of long-context decoders (Gemma 2/3, Mistral 7B v0.1) |
| Local + strided | about $O(n\sqrt{n})$ | mixed | research and specialised kernels |
| Local + global tokens | $O(n(w + \lvert G\rvert))$ | encoder (no cache) | long-document encoders (Longformer, BigBird, ModernBERT's local/global alternation) |

**Interleave, don't replace.** A model made only of local layers is cheap but weak at "find the one sentence 50,000 tokens back." Keeping a fraction of global layers restores direct long-range paths. The global layers then dominate the KV cache, as the worked example shows.

**The kernel matters more than the mask.** A boolean mask applied to a dense score matrix still computes all $n^2$ scores, so nothing gets faster. Real savings need a kernel that skips masked blocks (FlexAttention, FlashAttention's sliding-window support, or a serving engine that implements the model's pattern).

**Learned sparsity.** Newer work selects which blocks to attend per query instead of fixing the pattern in advance, for example Native Sparse Attention ([Yuan et al., 2025](https://arxiv.org/abs/2502.11089)). Differential attention ([Ye et al., 2024](https://arxiv.org/abs/2410.05258)) is a different idea: it subtracts two softmax maps to cancel weight that piles onto uninformative tokens such as the first one (the "attention sink"). Both are worth knowing; neither is a default yet.

**Serving.** Inference engines allocate KV memory per layer type. Check that your engine supports the model's sliding window; one that silently treats every layer as global still gives correct outputs but uses far more memory.

### Common mistakes

- Off-by-one windows: deciding whether $w$ counts the query itself. Here it does ($i - j < w$). Check against the model's reference code.
- Applying an encoder pattern (global tokens reading everything) inside a decoder, which leaks future tokens during training.
- Expecting speed from a mask alone, without a block-sparse kernel.
- Assuming a large theoretical receptive field means good long-range recall.
- Sizing KV memory for a hybrid model as if every layer were global, or as if every layer were local.

## 6. Knowledge check

<!-- quiz:attention-variants -->
**[Take the attention variants quiz](../../quizzes/attention-variants.md)**
<!-- /quiz -->

**Practice exercise.** A 32-token sequence uses causal sliding-window attention with $w = 8$. How many cells are attended? How many layers until the last token can be influenced by the first?

<details>
<summary>Solution</summary>

Cells: rows 0 to 6 have 1 to 7 keys, $1 + 2 + \dots + 7 = 28$. Rows 7 to 31 are 25 rows of 8: $200$. Total $228$. Formula: $32 \cdot 8 - 8 \cdot 7/2 = 256 - 28 = 228$, versus $32 \cdot 33/2 = 528$ for causal.

Reach: $L(w-1)+1 \ge 32$ means $7L \ge 31$, so $L \ge 4.43$: **5 layers** ($5 \cdot 7 + 1 = 36 \ge 32$; 4 layers give only 29).
</details>

**Implementation challenge.** Add a `dilated(n, w, d)` mask to the script: each query reads $w$ earlier keys spaced $d$ apart ($i, i-d, i-2d, \dots$). Print its cell count for $n = 16, w = 4, d = 2$ and its receptive field per layer. Why does stacking a dilated layer on a plain window layer grow the reach faster than two window layers?

<details>
<summary>Solution</summary>

```python
def dilated(n, w, d):
    i, j = np.indices((n, n))
    gap = i - j
    return (gap >= 0) & (gap % d == 0) & (gap < w * d)
```

For $n = 16, w = 4, d = 2$: rows with $i \ge 6$ have 4 keys (10 rows, 40 cells); rows 0 to 5 have 1, 1, 2, 2, 3, 3 (12 cells). Total 52. A dilated layer alone only reaches positions of the same parity, so its receptive field never covers odd gaps. After a plain window layer fills the gaps, the dilated layer jumps $(w-1)d = 6$ positions per hop instead of 3. That is the WaveNet idea: increase dilation with depth to grow reach exponentially.
</details>

## Summary

- Every attention variant here is a mask: a rule for which query-key pairs get scored.
- Sliding window costs $O(nw)$ and keeps only $w$ tokens of KV cache per layer; causal costs $O(n^2)$.
- Stacked windows reach $L(w-1)+1$ tokens, but many-hop paths blur, so models keep some global layers.
- In a local/global hybrid the global layers dominate KV memory; the ratio is the key number.
- Masks save compute only with kernels that skip masked blocks.

**Next:** [MDPs, states, actions, and rewards](../26-reinforcement-learning/01-mdps-states-actions-rewards.md)

**Related:** [Self-attention](01-self-attention.md) · [LLM inference](../18-llm-inference/01-llm-inference.md) · [Open models: architecture walkthroughs](../15-llms/07-open-models-architecture-walkthroughs.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Differential Attention (V2)](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/16-differential-attention-v2) · [Native Sparse Attention (DeepSeek NSA)](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/17-native-sparse-attention)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Attention Variants — Sliding Window, Sparse, Differential](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/15-attention-variants).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>What does sliding-window attention change, and what does it cost?</strong></summary>

Each query attends only to the last w tokens, itself included, instead of the whole prefix. The score matrix becomes a band: about n·w cells instead of n²/2, so compute grows linearly with sequence length for a fixed window. The KV cache for that layer only holds the last w tokens, so memory stops growing with context. With n = 16 and w = 4 the mask has 58 cells against 136 for causal. Information still travels further than w, because each layer extends reach by w − 1: after L layers a token can be influenced by L(w − 1) + 1 positions. The cost is that long-range information must hop through many layers and gets diluted, so exact recall of a fact far back degrades. That is why production models such as Gemma 2 and 3 interleave sliding-window layers with full global layers rather than using windows everywhere.

</details>

<details>
<summary><strong>Gemma 3 uses five local layers per global layer instead of one to one. What's the trade-off?</strong></summary>

More local layers means a smaller KV cache and less attention compute at long context, because the global layers dominate memory. For a synthetic 80-layer model at 131k tokens, all-global costs about 43 GB of KV per sequence; at roughly 5:1 with a 1,024 window it is about 7.3 GB, close to the cost of the 13 global layers alone. That memory is what lets you serve more concurrent long requests. The price is fewer direct long-range paths: fewer layers where any token can read any other, so retrieval of distant facts relies on fewer layers. A 1:1 mix keeps more long-range capacity but costs more memory. The right ratio is an empirical choice made at training time with long-context evaluations such as needle-in-a-haystack and multi-document QA; you can't change it after training without retraining or adapting the model.

</details>

<details>
<summary><strong>You added a sliding-window mask to your model and training got no faster. Why?</strong></summary>

A mask applied to a dense score matrix still computes all n² scores and then throws most away by setting them to minus infinity, so the arithmetic and memory traffic are unchanged. Speed comes from never computing masked blocks. That needs a kernel that understands the pattern: FlexAttention with a block mask, FlashAttention's sliding-window option, or a custom block-sparse kernel. Also check the sequence length: if n is close to w, the band covers most of the matrix anyway and there is little to skip. And attention may not be the bottleneck at all; at short context the feed-forward layers dominate FLOPs. Profile before and after, confirm the kernel actually reports block skipping, and compare at a long sequence length such as 32k where attention dominates.

</details>

<details>
<summary><strong>A causal model uses a window of 4,096. How many layers until the last token of a 32,768-token context can see the first?</strong></summary>

Each layer extends the reach by w − 1 = 4,095 positions, and the token itself counts once, so after L layers the reach is 4,095·L + 1. We need that to be at least 32,768, so L ≥ 32,767 / 4,095 ≈ 8.0. Eight layers give 4,095 × 8 + 1 = 32,761, six short, so the answer is 9 layers. In a 32-layer model that is easy to satisfy on paper. The real question is whether information survives nine hops of mixing with everything else in each window, which it often does not for a single precise fact. That is why I would check the model's global layers and run a long-context retrieval evaluation rather than trusting the theoretical receptive field.

</details>
