---
title: The transformer architecture
summary: Trace a token through embeddings, positional information, attention, residual connections, layer normalization, and the feed-forward network; count a block's parameters; and compare encoder, decoder, and encoder-decoder models.
skill: transformers
minutes: 40
prerequisites: [self-attention, training-and-regularization]
related: [self-attention, tokenization-and-pretraining, decoding]
adapted_from: [phases/07-transformers-deep-dive/04-positional-encoding, phases/07-transformers-deep-dive/05-full-transformer, phases/07-transformers-deep-dive/14-build-a-transformer-capstone, phases/19-capstone-projects/32-token-positional-embeddings, phases/19-capstone-projects/34-transformer-block, phases/19-capstone-projects/35-gpt-model-assembly]
---

# The transformer architecture

> **Mental model.** A transformer is a stack of identical blocks. In each block, attention lets every token gather information from other tokens, then a small feed-forward network transforms each token on its own. Residual connections keep the original signal flowing, and layer normalization keeps it well scaled.

**You will learn to**
- List the stages of a transformer block and what each does.
- Explain residual connections and pre-norm versus post-norm.
- Count the parameters of a block and a full model.
- Distinguish encoder-only, decoder-only, and encoder-decoder models and their uses, including cross-attention.
- Reason about how cost grows with sequence length.

**Why it matters.** Every modern LLM is this architecture repeated dozens of times. Knowing where parameters and compute live explains model sizes, context-length limits, and inference costs.

## 1. Intuition

1. **Token embeddings** turn token IDs into vectors.
2. **Positional information** is added (or applied to queries and keys with rotary embeddings), because attention alone ignores order.
3. Each **block** then does two things, each wrapped in a residual connection:
   - **Multi-head self-attention:** mix information across positions ("communication").
   - **Feed-forward network (MLP):** process each position independently, expanding to a wider hidden size and back ("computation"). Much of a model's stored knowledge is thought to live here.
4. **Layer normalization** keeps activations in a stable range. Modern models normalize *before* each sublayer (pre-norm), which trains more stably in deep stacks.
5. A final layer norm and a linear **output head** produce logits over the vocabulary (often sharing weights with the input embedding).

**Residual connections** add a sublayer's input to its output: $x + \text{Attention}(x)$. Each block only has to learn a *change* to the representation, and gradients flow straight through the additions, which makes very deep stacks trainable.

## 2. Visualization

![Left: parameters in one GPT-2-small block: attention 2.36 million, feed-forward 4.72 million, layer norms 0.003 million. Right: relative cost versus sequence length on log axes; attention scores grow with the square of length while feed-forward compute grows linearly.](../../figures/transformer-architecture.png)

*The feed-forward network holds about two-thirds of each block's parameters. Attention's score matrix is what grows quadratically with context.*

*Interactive: step through self-attention, the core of each block, with shapes. [Open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/attention/).*

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $x$ | the sequence of token representations, shape $[n, d]$ |
| $d$ | model width ($d_{\text{model}}$) |
| $d_{\text{ff}}$ | feed-forward hidden width, typically $4d$ |
| $h$ | number of attention heads |
| $L$ | number of blocks |
| $V$ | vocabulary size |

### A pre-norm block

```math
x' = x + \text{MultiHeadAttention}(\text{LayerNorm}(x))
\qquad
y = x' + \text{FFN}(\text{LayerNorm}(x'))
```

```math
\text{FFN}(u) = W_2\,\text{GELU}(W_1 u + b_1) + b_2, \qquad W_1: [d_{\text{ff}}, d],\; W_2: [d, d_{\text{ff}}]
```

Shapes never change through a block: $[n, d]$ in, $[n, d]$ out. That is what makes blocks stackable.

### Positional information

Attention treats its input as a set: shuffle the tokens and each output row is shuffled the same way, with nothing else changed. Order has to be injected. Three schemes matter:

- **Learned absolute** (GPT-2): a trainable table `pos_emb` of shape $[\text{ctx}, d]$, added to the token embedding. Nothing exists past `ctx`.
- **Sinusoidal** (original transformer): a fixed table, no parameters. For position $p$ and feature pair $i$:

```math
PE(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad PE(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i/d}}\right)
```

  With $d = 4$ at $p = 1$: pair $i = 0$ has frequency $1/10000^{0} = 1$, giving $\sin 1 = 0.8415$ and $\cos 1 = 0.5403$. Pair $i = 1$ has frequency $1/10000^{2/4} = 0.01$, giving $\sin 0.01 = 0.0100$ and $\cos 0.01 = 1.0000$. Low pairs spin fast and tell neighbours apart; high pairs spin slowly and encode coarse position.
- **Rotary (RoPE)**, used by most current open LLMs: instead of adding anything, rotate each pair of query and key features by an angle proportional to the position. Rotating $q$ by angle $p\theta$ and $k$ by angle $m\theta$ makes $q \cdot k$ depend only on the offset $p - m$, so attention sees relative distance directly.

### Families

| Family | Attention pattern | Typical use | Examples |
|---|---|---|---|
| Encoder-only | bidirectional (no mask) | classification, embeddings, extraction | BERT-style encoders, embedding models |
| Decoder-only | causal mask | text generation, chat LLMs | GPT-style models |
| Encoder-decoder | encoder bidirectional; decoder causal plus **cross-attention** to the encoder's outputs | translation, summarization | T5-style models |

In **cross-attention**, queries come from the decoder and keys and values from the encoder, so each generated token can look at the whole input.

### Worked example: counting GPT-2-small's parameters

$d = 768$, $d_{\text{ff}} = 3072$, $L = 12$, $V = 50{,}257$, context 1,024.

- **Attention** per block: $W_Q, W_K, W_V, W_O$ are each $768 \times 768$ with biases: $4 \times 768^2 + 4 \times 768 = 2{,}359{,}296 + 3{,}072 = 2{,}362{,}368$.
- **Feed-forward** per block: $2 \times 768 \times 3072 + 3072 + 768 = 4{,}718{,}592 + 3{,}840 = 4{,}722{,}432$.
- **Layer norms** per block: two, each with scale and shift: $2 \times 2 \times 768 = 3{,}072$.
- **Block total:** $7{,}087{,}872$. Twelve blocks: $85{,}054{,}464$.
- **Embeddings:** tokens $50{,}257 \times 768 = 38{,}597{,}376$ plus positions $1{,}024 \times 768 = 786{,}432$, total $39{,}383{,}808$. The output head reuses the token embedding (weight tying).
- **Final layer norm:** $1{,}536$.

Total: $85{,}054{,}464 + 39{,}383{,}808 + 1{,}536 = 124{,}439{,}808$, the published 124M.

### Compute per token

A forward pass costs roughly $2 \times (\text{parameters})$ multiply-adds per token, plus the attention-score term that grows with context length $n$ (about $2 n d$ per layer per token).

## 4. Implementation

The full script is [`code/14-transformers/gpt_scratch.py`](../../code/14-transformers/gpt_scratch.py) (NumPy only). It reuses `attention`, `softmax`, and `split_heads` from the [self-attention lesson's script](../../code/14-transformers/attention_scratch.py) and assembles a complete GPT-style forward pass.

### Build it from scratch

**Step 1: one shape table for the whole model.** Every parameter's shape is written down once. The same table initializes the tiny model and counts GPT-2-small, so the count in section 3 is checked by code rather than trusted. The query, key, and value projections are fused into one $[d, 3d]$ matrix, as in GPT-2.

```python
def param_shapes(V, ctx, d, L, d_ff):
    block = {"ln1_g": (d,), "ln1_b": (d,), "W_qkv": (d, 3 * d), "b_qkv": (3 * d,),
             "W_o": (d, d), "b_o": (d,), "ln2_g": (d,), "ln2_b": (d,),
             "W_1": (d, d_ff), "b_1": (d_ff,), "W_2": (d_ff, d), "b_2": (d,)}
    top = {"tok_emb": (V, d), "pos_emb": (ctx, d), "lnf_g": (d,), "lnf_b": (d,)}
    return top, [block] * L          # no separate output head: it reuses tok_emb
```

*Predict:* if you untie the output head (add a separate `(d, V)` matrix), how many parameters does GPT-2-small gain?

**Step 2: the per-token operations.** LayerNorm normalizes each token's $d$ features to mean 0 and variance 1, then rescales with learned `g` and `b`. The MLP widens each token to $d_{\text{ff}}$, applies GELU, and projects back. Neither looks at other tokens.

```python
def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)                  # per token, over the d features
    var = x.var(axis=-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def gelu(x):                                             # tanh approximation used by GPT-2
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def mlp(x, p):
    return gelu(x @ p["W_1"] + p["b_1"]) @ p["W_2"] + p["b_2"]   # [n, d] -> [n, d_ff] -> [n, d]
```

*Predict:* LayerNorm here uses `axis=-1`. What would go wrong at inference if it normalized over the sequence axis instead?

**Step 3: causal multi-head attention.** One matrix multiply produces Q, K, and V side by side; `np.split` cuts them apart. Then it is the batched multi-head attention from the previous lesson, followed by the output projection.

```python
def causal_mha(x, p, h):
    q, k, v = np.split(x @ p["W_qkv"] + p["b_qkv"], 3, axis=-1)   # three [n, d]
    out, _ = attention(split_heads(q, h), split_heads(k, h), split_heads(v, h), causal=True)
    merged = out.transpose(1, 0, 2).reshape(x.shape[0], -1)       # [n, d]
    return merged @ p["W_o"] + p["b_o"]
```

*Predict:* attention is the only place tokens exchange information. If you removed the causal mask here, which check in the output below would fail?

**Step 4: one pre-norm block.** This is the two-line formula from section 3. Each sublayer sees a normalized copy of `x`, and its output is *added* to `x`. The residual path itself is never normalized, which is why gradients reach early layers cleanly.

```python
def block(x, p, h):
    x = x + causal_mha(layer_norm(x, p["ln1_g"], p["ln1_b"]), p, h)   # communicate
    x = x + mlp(layer_norm(x, p["ln2_g"], p["ln2_b"]), p)             # compute
    return x                                                          # still [n, d]
```

*Predict:* if every weight in a block were zero, what would the block output?

**Step 5: the whole model.** Look up a token vector and a position vector and add them. Run the blocks, apply a final LayerNorm, and multiply by the transposed token embedding to get one logit per vocabulary entry at every position.

```python
def gpt_forward(ids, P, h):
    n = len(ids)
    x = P["tok_emb"][ids] + P["pos_emb"][:n]           # [n, d]: what + where
    for p in P["blocks"]:
        x = block(x, p, h)
    x = layer_norm(x, P["lnf_g"], P["lnf_b"])
    return x @ P["tok_emb"].T                          # tied head: logits [n, V]
```

*Predict:* generation only needs the logits at the last position. What fraction of this computation does a [KV cache](../18-llm-inference/01-llm-inference.md) let you skip for each new token?

### Use the library

The PyTorch block below is the same computation as steps 2 to 4, with autograd, dropout, and fused GPU kernels. `nn.MultiheadAttention` holds separate Q, K, V weights where GPT-2 fuses them, but the parameter count is identical.

```python
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, d=768, heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d), nn.Dropout(dropout))

    def forward(self, x):                                  # x: [B, n, d]
        n = x.size(1)
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=causal, need_weights=False)[0]   # residual 1
        return x + self.ff(self.ln2(x))                                       # residual 2

print(sum(p.numel() for p in Block().parameters()))       # 7,087,872
```

To load real GPT-2 weights instead of random ones, use Hugging Face `transformers` (`AutoModelForCausalLM.from_pretrained("gpt2")`).

### Run it

```bash
uv run code/14-transformers/gpt_scratch.py
```

Key lines of the output (the tiny model's weights are synthetic and random):

```text
== GPT-2-small parameter count from the shape table ==
total: 124,439,808
one block: 7,087,872
ids shape: (5,) -> logits shape: (5, 12)
positions 0-3 unchanged: True
position 4 changed:      True
sequence: [3, 7, 1, 4, 9, 9, 9, 9, 9, 9]
```

The totals match the hand count in section 3. Changing the last token leaves earlier logits untouched, which proves the mask works. Random weights generate a repeated token, which is what an untrained model does. FLOP and memory figures are in [`code/15-llms/llms.py`](../../code/15-llms/llms.py).

## 5. Engineering

**Scaling.** Model quality improves predictably with parameters, data, and compute (scaling laws). Width and depth grow together; modern models add tweaks such as RMSNorm instead of LayerNorm, SwiGLU feed-forward layers, rotary position embeddings, and grouped-query attention to cut KV-cache memory.

**Memory and latency.** Weights dominate memory for short contexts; the KV cache dominates for long contexts and large batches. Attention cost grows with $n^2$ during prefill (processing the prompt); each new generated token costs $O(n)$ with a KV cache.

**Choosing a family.** Use encoder models for classification, retrieval embeddings, and token tagging (cheaper, bidirectional). Use decoder models for open-ended generation and instruction following. Encoder-decoders remain strong for translation and structured transformation tasks.

> [!WARNING]
> **Failure modes.** Missing or wrong causal masks; position handling that breaks beyond the trained context length; numerical instability in deep post-norm stacks; underestimating KV-cache memory when raising batch size or context.

### Common mistakes

- Assuming attention holds most of the parameters (the feed-forward layers do).
- Forgetting weight tying when counting parameters.
- Using a decoder-only model where a small encoder would do the job cheaper.

## 6. Knowledge check

<!-- quiz:transformer-architecture -->
**[Take the transformer architecture quiz](../../quizzes/transformer-architecture.md)**
<!-- /quiz -->

**Practice exercise.** Count the parameters of one block with $d = 512$ and $d_{\text{ff}} = 2048$ (with biases and two layer norms).

<details>
<summary>Solution</summary>

Attention: $4 \times 512^2 + 4 \times 512 = 1{,}048{,}576 + 2{,}048 = 1{,}050{,}624$. FFN: $2 \times 512 \times 2048 + 2048 + 512 = 2{,}097{,}152 + 2{,}560 = 2{,}099{,}712$. Layer norms: $2{,}048$. Total: $3{,}152{,}384$.
</details>

**Implementation challenge.** Stack 4 `Block`s with token and position embeddings and a tied output head, train it on a small text file for next-character prediction, and sample from it.

## Summary

- A block is attention (communication) plus a feed-forward network (per-token computation), each with a residual connection and layer normalization.
- Shapes stay $[n, d]$ through every block, so blocks stack.
- The feed-forward layers hold about two-thirds of block parameters; GPT-2 small totals 124M.
- Encoder-only for understanding and embeddings, decoder-only for generation, encoder-decoder (with cross-attention) for sequence-to-sequence tasks.

**Next:** [BERT and masked language modeling](03-bert-masked-language-modeling.md)

**Related:** [Self-attention](01-self-attention.md) · [Model card: transformer encoder](../../models/transformer-encoder.md) · [Model card: transformer decoder](../../models/transformer-decoder.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Machine Translation](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/11-machine-translation)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Positional Encoding — Sinusoidal, RoPE, ALiBi](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/04-positional-encoding), [The Full Transformer — Encoder + Decoder](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/05-full-transformer), [Build a Transformer from Scratch — The Capstone](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/14-build-a-transformer-capstone), [Token and Positional Embeddings](https://aiengineeringfromscratch.com/lesson?path=phases/19-capstone-projects/32-token-positional-embeddings), [Transformer Block from Scratch](https://aiengineeringfromscratch.com/lesson?path=phases/19-capstone-projects/34-transformer-block), [GPT Model Assembly](https://aiengineeringfromscratch.com/lesson?path=phases/19-capstone-projects/35-gpt-model-assembly).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Why do residual connections help deep networks train?</strong></summary>

They give both the signal and the gradient an identity path around every block. A residual block computes $y = x + F(x)$, so $\partial y/\partial x = I + \partial F/\partial x$. When gradients flow back through $L$ blocks, the product of these Jacobians always contains the identity term, so the gradient can't shrink to zero just because some $\partial F/\partial x$ are small. Without residuals, the gradient is a product of $L$ factors that tends to vanish or explode. Residuals also change what each block learns: only a change to the representation, and at initialization, when $F$ is near zero, the whole network is close to the identity, an easy starting point. In transformers the residual stream carries information through all layers while attention and the FFN read from and write to it. Pre-norm, $x + F(\text{LayerNorm}(x))$, keeps that identity path clean, which is why it trains stably in deep stacks.

</details>

<details>
<summary><strong>For ticket classification and semantic search, would you use an encoder-only or a decoder-only model?</strong></summary>

Usually an encoder-only model, at least as the first baseline. Encoders attend bidirectionally, so every token's representation uses context on both sides, which is what you want for classification, token tagging, and embeddings. They are small (100M to 400M parameters is common), so they are cheap to fine-tune and serve, and a forward pass gives you the answer with no generation loop. For semantic search, a bi-encoder produces one vector per document for a vector index. A decoder-only LLM can classify zero-shot or few-shot with no labeled data and handles open-ended outputs, but it costs far more per request and adds latency from autoregressive decoding. My order: with labeled data, fine-tune a small encoder; without it, prompt a decoder model to bootstrap labels, then distill into an encoder if volume makes the cost matter. Encoder-decoders fit sequence-to-sequence tasks such as translation.

</details>

<details>
<summary><strong>Estimate the parameter count and per-token compute of a decoder with d = 4096, 32 layers, d_ff = 4d, and a 32,000-token vocabulary.</strong></summary>

About 6.6 billion parameters and about 13 GFLOPs per token. Per block, attention has $4d^2$ weights ($W_Q, W_K, W_V, W_O$) and the FFN has $2 \times d \times 4d = 8d^2$, so a block is about $12d^2 = 12 \times 4096^2 = 201{,}326{,}592$; biases and layer norms are negligible. Thirty-two blocks give about $6.44 \times 10^9$. The token embedding adds $32{,}000 \times 4096 = 131{,}072{,}000$, and an untied output head adds the same again. Total is about 6.6 to 6.7 billion, a "7B-class" model. LLaMA-style models use a SwiGLU FFN with three matrices and $d_{\text{ff}} \approx \tfrac{8}{3}d$, which keeps the FFN near $8d^2$. A forward pass costs about $2 \times$ parameters FLOPs per token, about 13 GFLOPs, plus an attention term that grows with context. Two-thirds of the block parameters live in the FFN.

</details>

<details>
<summary><strong>Your model works well up to 4,000 tokens, but quality collapses on longer inputs. What is going on?</strong></summary>

Most likely the model was trained at a 4,096-token context and position handling doesn't extrapolate. Learned absolute position embeddings simply don't exist past the trained length. Rotary embeddings (RoPE) produce rotation angles at long positions the model never saw, and attention patterns break down. Check the training context length in the model config first. Then rule out pipeline bugs: silent truncation by the tokenizer or server, a wrong `max_position_embeddings`, or a chat template that pushes the instructions out of the window. If it really is extrapolation, the fixes are RoPE scaling methods such as position interpolation, which compress positions into the trained range and usually need a short fine-tune on long sequences, or switching to a model trained for long context. Even within the trained window, measure retrieval of facts placed in the middle of long inputs; models often attend to them less reliably.

</details>
