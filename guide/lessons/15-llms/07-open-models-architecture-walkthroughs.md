---
title: "Open models: architecture walkthroughs"
summary: Read the config.json of Llama 3, Mistral, Mixtral, Gemma 2, Qwen2.5, and DeepSeek-V3, name each design choice, and compute parameter counts and KV-cache memory from the config alone.
skill: llms
minutes: 45
prerequisites: [transformer-architecture, llm-inference, mixture-of-experts]
related: [attention-variants, scaling-laws, serving-llms, fine-tuning-in-practice]
adapted_from: [phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs]
---

# Open models: architecture walkthroughs

> **Mental model.** Modern open LLMs are the same car with different option packages. Underneath is the decoder-only transformer from GPT-2: embeddings, a stack of attention-plus-MLP blocks, a final norm, and an output head. Each model family then ticks a few boxes on the same short options list: RMSNorm, RoPE, a gated MLP, shared key-value heads, mixture-of-experts. If you can read the option sheet (the `config.json`), you know what you're buying.

**You will learn to**
- Name the six design choices that separate current open models from GPT-2, and say what each buys.
- Read a Hugging Face `config.json` and explain each field.
- Compute a model's total and active parameter count from its config, and check it against the published number.
- Compute KV-cache bytes per token for GQA, sliding-window, and MLA models.
- Pick an open model for a memory and context budget.

**Why it matters.** Choosing, fine-tuning, or serving an open model starts with sizing it: will the weights fit, how much context can each request hold, how many requests fit at once. Every one of those answers comes from a few config fields. Interviewers ask for exactly this arithmetic.

## 1. Intuition

Start from the [transformer architecture](../14-transformers/02-transformer-architecture.md): token embedding, $L$ pre-norm decoder blocks (norm, causal self-attention, residual, norm, MLP, residual), a final norm, and a linear head over the vocabulary. Every model in this lesson keeps that skeleton. What changed between GPT-2 (2019) and today's open models is a short list of choices:

1. **RMSNorm instead of LayerNorm.** Divide by the root-mean-square of the activations and scale. No mean subtraction, no bias. Slightly cheaper, works as well ([Zhang and Sennrich, 2019](https://arxiv.org/abs/1910.07467)).
2. **RoPE instead of learned position embeddings.** Rotate each query and key by an angle that depends on its position, so attention scores depend on *relative* distance. No lookup table that runs out at 1,024 positions, and the angles can be rescaled to extend context ([Su et al., 2021](https://arxiv.org/abs/2104.09864)).
3. **A gated MLP (SwiGLU or GeGLU) instead of GELU.** Two parallel projections, one passed through an activation and used as a gate on the other, then a down projection: three matrices instead of two ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)).
4. **Shared keys and values.** [Grouped-query attention](../../glossary.md#grouped-query-attention) (GQA) lets groups of query heads share one key-value head, cutting the [KV cache](../../glossary.md#kv-cache). DeepSeek's **multi-head latent attention** (MLA) goes further: it caches one small compressed vector per token and expands it into per-head keys and values when needed.
5. **Mixture-of-experts MLPs.** Replace the MLP with many expert MLPs and a router that picks a few per token. Memory follows *total* parameters, compute follows *active* ones (see [mixture of experts](05-mixture-of-experts.md)).
6. **Local attention layers.** Some models make part of the stack sliding-window layers that only see recent tokens (see [attention variants](../14-transformers/04-attention-variants.md)).

Everything else, such as data mix, learning-rate schedule, and context-extension recipe, is training, not architecture.

| Model | Params (total / active) | Norm | MLP | Positions | Attention | MoE |
|---|---|---|---|---|---|---|
| GPT-2 small (2019) | 124M | LayerNorm | GELU | learned | MHA, 12 heads | no |
| Llama 3 8B (2024) | 8.0B | RMSNorm | SwiGLU | RoPE | GQA 32 Q / 8 KV | no |
| Mistral 7B | 7.2B | RMSNorm | SwiGLU | RoPE | GQA 32 / 8 | no |
| Mixtral 8x7B | 46.7B / 12.9B | RMSNorm | SwiGLU | RoPE | GQA 32 / 8 | 8 experts, top-2 |
| Gemma 2 9B | 9.2B | RMSNorm (pre and post) | GeGLU | RoPE | GQA 16 / 8, local and global alternating | no |
| Qwen2.5 72B | 72.7B | RMSNorm | SwiGLU | RoPE | GQA 64 / 8, QKV bias | no |
| DeepSeek-V3 | 671B / 37B | RMSNorm | SwiGLU | RoPE (decoupled) | MLA, 128 heads | 256 routed + 1 shared, top-8 |

> [!NOTE]
> Mistral 7B v0.1 used sliding-window attention in every layer; the config used here (v0.3) does not. Families change between versions, so always read the config of the exact checkpoint you'll run.

## 2. Visualization

![Left: horizontal bars of total and active parameters on a log scale. Dense models (Llama 3 8B, Mistral 7B, Gemma 2 9B, Qwen2.5 72B) have equal total and active bars. Mixtral 8x7B has 46.7B total and 12.9B active; DeepSeek-V3 has 671B total and 37.6B active. Right: KV cache per sequence in BF16 at 32,768 tokens: Llama 3 8B, Mistral 7B and Mixtral 4.29 GB each, Qwen2.5 72B 10.74 GB, DeepSeek-V3 2.30 GB with MLA, and Gemma 2 9B 2.11 GB at its 8,192-token maximum.](../../figures/open-models-architecture-walkthroughs.png)

*Left: MoE models separate memory (grey, total) from per-token compute (blue, active). Right: KV cache per request depends on KV heads, head size, and layer types, not on total size. DeepSeek-V3, the largest model, has one of the smallest caches.*

**Try it** (in [`code/15-llms/open_model_configs.py`](../../code/15-llms/open_model_configs.py); predict first, then run):

1. Set Llama 3 8B's `num_key_value_heads` to 32 (plain multi-head attention). Predict the new KV per token and total parameter count.
2. Set `tie_word_embeddings=True` for Llama 3 8B. How many parameters disappear? (One $128{,}256 \times 4{,}096$ matrix.)
3. Change Gemma 2's `layer_types` so every layer is `"full_attention"`. What happens to its 8,192-token KV cache?
4. Raise Mixtral's `num_experts_per_tok` from 2 to 4. Which of total and active changes?

## 3. The math

### Symbols

| Symbol | Config field | Meaning |
|---|---|---|
| $h$ | `hidden_size` | width of the residual stream |
| $f$ | `intermediate_size` | MLP hidden width |
| $L$ | `num_hidden_layers` | number of blocks |
| $n_q$ | `num_attention_heads` | query heads |
| $n_{kv}$ | `num_key_value_heads` | key-value heads (GQA when $n_{kv} < n_q$) |
| $d$ | `head_dim`, else $h/n_q$ | size of one head |
| $V$ | `vocab_size` | vocabulary size |
| $E, k$ | `num_local_experts`, `num_experts_per_tok` | experts per MoE layer, experts used per token |
| $w$ | `sliding_window` | tokens kept by a local layer |
| $b$ | | bytes per value (2 for BF16) |

### Parameters per block

```math
P_{\text{attn}} = \underbrace{h \cdot n_q d}_{Q} + \underbrace{2 \cdot h \cdot n_{kv} d}_{K,\,V} + \underbrace{n_q d \cdot h}_{\text{out}}, \qquad P_{\text{MLP}}^{\text{gated}} = 3hf, \qquad P_{\text{norms}} = 2h
```

A plain GELU MLP has $2hf$; a gated MLP has three matrices, so models pick $f \approx \tfrac{8}{3}h$ to match the $4h$ GELU MLP's $8h^2$ parameters ($3h \cdot \tfrac{8}{3}h = 8h^2$). Llama 3 rounds up: $f = 14{,}336 = 3.5h$.

Total: $P = Vh\,(\times 2 \text{ if the output head is not tied}) + L(P_{\text{attn}} + P_{\text{MLP}} + P_{\text{norms}}) + h$.

For an MoE layer: total MLP $= E \cdot P_{\text{expert}}$ plus a router of $hE$; active MLP $= k \cdot P_{\text{expert}}$ plus the router.

### KV cache

```math
\text{KV bytes} = \underbrace{2\, n_{kv}\, d}_{\text{K and V per layer}} \cdot b \cdot \sum_{\text{layers}} \text{tokens kept}, \qquad \text{tokens kept} = \begin{cases} T & \text{global layer} \\ \min(T, w) & \text{local layer} \end{cases}
```

MLA stores one latent of size $r$ (`kv_lora_rank`) plus a shared RoPE key of size $d_R$ per layer instead of $2n_{kv}d$.

### Worked example: Llama 3 8B from its config

Config: $h = 4{,}096$, $f = 14{,}336$, $L = 32$, $n_q = 32$, $n_{kv} = 8$, $d = 4{,}096/32 = 128$, $V = 128{,}256$, untied head.

1. Q: $4{,}096 \cdot 4{,}096 = 16{,}777{,}216$. Output: the same, $16{,}777{,}216$.
2. K and V: $2 \cdot 4{,}096 \cdot 8 \cdot 128 = 8{,}388{,}608$ (a quarter of the 33.5M full multi-head attention would need).
3. Attention per layer: $16{,}777{,}216 \cdot 2 + 8{,}388{,}608 = 41{,}943{,}040$.
4. MLP: $3 \cdot 4{,}096 \cdot 14{,}336 = 176{,}160{,}768$. Norms: $2 \cdot 4{,}096 = 8{,}192$.
5. Per block: $41{,}943{,}040 + 176{,}160{,}768 + 8{,}192 = 218{,}112{,}000$. Times 32: $6{,}979{,}584{,}000$.
6. Embedding and output head: $2 \cdot 128{,}256 \cdot 4{,}096 = 1{,}050{,}673{,}152$. Final norm: $4{,}096$.
7. Total: $6{,}979{,}584{,}000 + 1{,}050{,}673{,}152 + 4{,}096 = 8{,}030{,}261{,}248$, exactly the published count.

MLPs are 81% of the block parameters, and the vocabulary matrices alone are 13% of the model.

**KV cache:** $2 \cdot 8 \cdot 128 \cdot 2 = 4{,}096$ bytes per token per layer, times 32 layers $= 131{,}072$ bytes (128 KiB) per token. At 8,192 tokens: $131{,}072 \cdot 8{,}192 = 1{,}073{,}741{,}824$ bytes, 1.07 GB per sequence. With 32 KV heads (no GQA) it would be 4.29 GB.

### Worked example: Mixtral 8x7B, total versus active

Same attention and width as Mistral 7B, but each layer has 8 experts of $176{,}160{,}768$ parameters and a router of $4{,}096 \cdot 8 = 32{,}768$.

- MoE layer total: $41{,}943{,}040 + 8{,}192 + 32{,}768 + 8 \cdot 176{,}160{,}768 = 1{,}451{,}270{,}144$.
- MoE layer active (top-2): $41{,}943{,}040 + 8{,}192 + 32{,}768 + 2 \cdot 176{,}160{,}768 = 394{,}305{,}536$.
- With 32 layers, embeddings $2 \cdot 32{,}000 \cdot 4{,}096 = 262{,}144{,}000$, and the final norm: **46,702,792,704 total** and **12,879,925,248 active**.

"8x7B" is not 56B: attention and embeddings are shared, only the MLPs are multiplied. All 46.7B must sit in memory (93 GB in BF16), but each token computes like a 13B model.

### Worked example: DeepSeek-V3's MLA cache

DeepSeek-V3 caches a 512-value latent plus a 64-value RoPE key per layer: $(512 + 64) \cdot 2$ bytes $\cdot$ 61 layers $= 70{,}272$ bytes per token. Caching full keys (192 values per head) and values (128 per head) for 128 heads would need $128 \cdot 320 \cdot 61 \cdot 2 = 4{,}997{,}120$ bytes, 71× more. That's why a 671B model has a smaller per-token cache (69 KiB) than Llama 3 8B (128 KiB).

## 4. Implementation

### Build it from scratch

The full script is [`code/15-llms/open_model_configs.py`](../../code/15-llms/open_model_configs.py) (no downloads; configs are copied in as dicts with Hugging Face field names). Attention parameters come first. GQA only changes the K and V projections, and some families add biases.

```python
    d = c.get("head_dim", h // nq)
    nkv = c["num_key_value_heads"]
    p = h * nq * d + 2 * h * nkv * d + nq * d * h             # Q, K and V, output
    if c.get("bias"):
        p += nq * d + 2 * nkv * d + h                         # GPT-2: bias on every projection
    elif c.get("qkv_bias"):
        p += nq * d + 2 * nkv * d                             # Qwen: bias on Q, K, V only
    return p
```

*Predict:* Gemma 2 9B has $h = 3{,}584$, 16 heads, and `head_dim = 256`. Is its Q projection square? ($16 \cdot 256 = 4{,}096 \ne 3{,}584$, so no. Ignoring an explicit `head_dim` is a classic miscount.)

MLA replaces the K and V projections with a down-projection to a small latent and an up-projection back to per-head keys and values. Queries get the same low-rank treatment.

```python
    if "kv_lora_rank" in c:                                   # MLA (DeepSeek)
        qk = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
        q = h * c["q_lora_rank"] + c["q_lora_rank"] + c["q_lora_rank"] * nq * qk
        kv = h * (c["kv_lora_rank"] + c["qk_rope_head_dim"]) + c["kv_lora_rank"] \
            + c["kv_lora_rank"] * nq * (c["qk_nope_head_dim"] + c["v_head_dim"])
        return q + kv + nq * c["v_head_dim"] * h
```

*Predict:* which of these matrices determines what goes in the KV cache? (The down-projection `h * (kv_lora_rank + qk_rope_head_dim)`: its 576 outputs per token are all that is cached.)

The MoE branch of `count` multiplies only the MLP. Dense layers at the bottom of the stack (DeepSeek-V3 keeps its first 3) and shared experts are handled the same way.

```python
        k, shared = c["num_experts_per_tok"], c.get("n_shared_experts", 0)
        expert = mlp_params(c, c.get("moe_intermediate_size", c["intermediate_size"]))
        n_dense = c.get("first_k_dense_replace", 0)
        router = h * experts
        moe_total = per_layer_fixed + router + (experts + shared) * expert
        moe_active = per_layer_fixed + router + (k + shared) * expert
        dense = per_layer_fixed + dense_mlp
        blocks = n_dense * dense + (L - n_dense) * moe_total
        blocks_active = n_dense * dense + (L - n_dense) * moe_active
```

*Predict:* the shared expert runs for every token. Does it count toward active parameters, total parameters, or both? (Both.)

The KV cache sums tokens kept per layer, so a hybrid local/global model is handled without special cases.

```python
def kv_cache_bytes(c, tokens, bytes_per=2):
    kinds = c.get("layer_types", ["full_attention"] * c["num_hidden_layers"])
    kept = [min(tokens, c["sliding_window"]) if k == "sliding_attention" else tokens for k in kinds]
    return kv_elems_per_token_layer(c) * sum(kept) * bytes_per
```

*Predict:* at 4,096 tokens, how much does Gemma 2's sliding window save? (Nothing. The window is 4,096, so local layers keep everything until the context exceeds it.)

### Use the library

Real configs come from the Hub, and the model can be built without allocating weights to count parameters exactly:

```python
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")       # gated: needs an accepted licence
print(cfg.hidden_size, cfg.num_key_value_heads, getattr(cfg, "head_dim", None), cfg.tie_word_embeddings)
with init_empty_weights():                                            # "meta" tensors, no memory used
    model = AutoModelForCausalLM.from_config(cfg)
print(sum(p.numel() for p in model.parameters()))                     # 8030261248
print(count(cfg.to_dict()))                                           # our function, after adding mlp/norm keys
```

What the library adds: exact module trees (so odd extras such as Qwen's QKV bias or Gemma's extra norms are never missed) and the model's own attention and RoPE code. Model cards often round counts and sometimes include extra modules; DeepSeek-V3's checkpoint is listed as 685B because it bundles a 14B multi-token-prediction module on top of the 671B main model.

### Run it

From `guide/`:

```bash
uv run code/15-llms/open_model_configs.py
```

```text
model                   total          active   KV/token  KV @ 32k, BF16
GPT-2 small       124,439,808     124,439,808      36 KiB         0.04 GB
Llama 3 8B      8,030,261,248   8,030,261,248     128 KiB         4.29 GB
Mistral 7B      7,241,732,096   7,241,732,096     128 KiB         4.29 GB
Mixtral 8x7B   46,702,792,704  12,879,925,248     128 KiB         4.29 GB
Gemma 2 9B      9,241,705,984   9,241,705,984     336 KiB         2.11 GB
Qwen2.5 72B    72,706,203,648  72,706,203,648     320 KiB        10.74 GB
DeepSeek-V3   671,026,404,352  37,552,282,624      69 KiB         2.30 GB

Gemma 2 9B at 8,192 tokens: 2.11 GB with sliding layers, 2.82 GB if every layer were global
DeepSeek-V3 MLA: 70,272 bytes/token vs 4,997,120 for uncompressed 128-head K and V (71x smaller)
```

The dense counts match the published figures (Llama 3 8B and GPT-2 small exactly, asserted in the script). DeepSeek-V3 lands at 671.0B total and 37.6B active against the reported 671B and 37B; the small gap comes from rounding and from tiny tensors we skip (the router's per-expert bias). Gemma 2's 32k column is at its 8,192-token maximum.

## 5. Engineering

**Choosing an open model.** Size the weights first: parameters × bytes per parameter (BF16 2, FP8 1, 4-bit about 0.5). Then add the KV cache for your context × concurrency. Then check the licence and the tooling support for that architecture in your serving engine.

| Constraint | Usually means |
|---|---|
| One 24 GB GPU | dense 7B to 9B in 4-bit or 8-bit, moderate context |
| One 80 GB GPU | dense 7B to 9B in BF16 with long context, or a ~70B model in 4-bit |
| A node of 8 × 80 GB | dense 70B in BF16, or large MoE in FP8 |
| Very long context | small $n_{kv}$, MLA, or many sliding-window layers |
| High throughput on big hardware | MoE: compute per token of a mid-size model with a larger model's capacity |

**MoE trade-offs.** All experts must be resident, so memory follows total parameters. At batch 1 decode is fast, since few experts are read per token; at large batches most experts are touched every step. Serving needs expert parallelism and good load balance.

**Context extension.** RoPE models extend context by rescaling rotary frequencies (`rope_theta`, `rope_scaling` with methods such as YaRN). A config's `max_position_embeddings` says what the model supports, not what it handles well; measure long-context quality.

**Fine-tuning.** LoRA targets are the projection names in the config's module tree (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). MLA and MoE models have different module names and more of them; check before you copy a recipe (see [fine-tuning in practice](04-fine-tuning-in-practice.md)).

### Common mistakes

- Computing $d$ as $h/n_q$ when the config sets `head_dim` explicitly (Gemma 2).
- Counting the output head twice for tied models, or once for untied ones.
- Using $n_q$ instead of $n_{kv}$ in the KV-cache formula.
- Treating every layer as global in a hybrid model, or every layer as local.
- Reading "8x7B" as 56B, or sizing MoE memory by active parameters.

## 6. Knowledge check

<!-- quiz:open-models-architecture-walkthroughs -->
**[Take the open models quiz](../../quizzes/open-models-architecture-walkthroughs.md)**
<!-- /quiz -->

**Practice exercise.** Llama 3 70B: $h = 8{,}192$, $f = 28{,}672$, $L = 80$, $n_q = 64$, $n_{kv} = 8$, $V = 128{,}256$, untied. Compute the KV cache per token in BF16 and for one 131,072-token sequence. Then estimate how many such sequences fit on 8 × 80 GB GPUs alongside BF16 weights of about 141 GB.

<details>
<summary>Solution</summary>

$d = 8{,}192/64 = 128$. Per token: $2 \cdot 8 \cdot 128 \cdot 2 \cdot 80 = 327{,}680$ bytes (320 KiB). One 131,072-token sequence: $327{,}680 \cdot 131{,}072 = 42{,}949{,}672{,}960$ bytes, 42.9 GB.

Memory: $640 - 141 = 499$ GB left, minus activations and engine overhead (say 10%): about 450 GB, so about $450/42.9 \approx 10$ full-length sequences at once. With an FP8 KV cache, about 20.
</details>

**Implementation challenge.** Add Llama 3 70B and a synthetic MoE of your own design to `CONFIGS`, then add a `fits(c, gpu_gb, n_gpus, weight_bytes, kv_bytes, context)` function that returns the maximum concurrent sequences. Check Llama 3 70B's total against the published 70.6B.

<details>
<summary>Solution</summary>

```python
def fits(c, gpu_gb, n_gpus, weight_bytes, kv_bytes, context, overhead=0.10):
    free = gpu_gb * 1e9 * n_gpus * (1 - overhead) - count(c)[0] * weight_bytes
    per_seq = kv_cache_bytes(c, context, kv_bytes)
    return max(0, int(free // per_seq))
```

Llama 3 70B with the config above counts to 70,553,706,496 parameters. Note that weights use *total* parameters for an MoE, even though compute uses active ones.
</details>

## Summary

- Current open models are GPT-2's decoder with RMSNorm, RoPE, gated MLPs, shared or compressed KV, and optionally MoE and local layers.
- Parameters per block: attention $h \cdot n_q d + 2h \cdot n_{kv} d + n_q d \cdot h$, gated MLP $3hf$; embeddings count twice when untied.
- KV cache per token: $2 n_{kv} d \cdot b$ per global layer; local layers cap tokens at $w$; MLA caches a small latent.
- MoE: memory follows total parameters, compute follows active parameters.
- Always read the exact checkpoint's config; families change between versions.

**Next:** [Prompt engineering](../28-prompting-and-context/01-prompt-engineering.md)

**Related:** [Mixture of experts](05-mixture-of-experts.md) · [Attention variants](../14-transformers/04-attention-variants.md) · [LLM inference](../18-llm-inference/01-llm-inference.md) · [Serving LLMs](../18-llm-inference/02-serving-llms.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Multi-Token Prediction (MTP)](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/18-multi-token-prediction) · [DeepSeek-V3 Architecture Walkthrough](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/20-deepseek-v3-walkthrough) · [Jamba — Hybrid SSM-Transformer](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Open Models: Architecture Walkthroughs](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>What changed between GPT-2 and Llama 3, and why?</strong></summary>

The skeleton is the same decoder-only transformer; a handful of components changed. LayerNorm became RMSNorm: no mean subtraction or bias, slightly cheaper and just as stable. Learned position embeddings became RoPE, which rotates queries and keys by position so scores depend on relative distance, and which can be rescaled to extend context. The GELU MLP became SwiGLU, a gated MLP with three matrices, which gives better loss per parameter. Multi-head attention became grouped-query attention, with 32 query heads sharing 8 key-value heads, which cuts the KV cache four times with little quality loss. Llama 3 also uses a 128k-token vocabulary, an untied output head, and far more training data, about 15 trillion tokens. None of these are radical; together they make the model cheaper to serve and better per parameter.

</details>

<details>
<summary><strong>How do you compute Llama 3 8B's parameter count from its config?</strong></summary>

Head size is 4,096 / 32 = 128. Attention per layer: Q and output are 4,096 × 4,096 each, about 16.8M; K and V with 8 KV heads are 2 × 4,096 × 1,024, about 8.4M; total about 41.9M. The SwiGLU MLP has three matrices of 4,096 × 14,336, about 176.2M. Two RMSNorms add 8,192. That is 218.1M per block, times 32 layers, about 6.98B. The embedding and the untied output head are each 128,256 × 4,096, about 525M, so 1.05B together. Adding the final norm gives 8,030,261,248, exactly the published number. Two things to notice: MLPs are about 80 percent of block parameters, and with a large vocabulary the embedding matrices are 13 percent of a small model.

</details>

<details>
<summary><strong>A 9B model OOMs at 32k context while an 8B model fits on the same GPU. What do you check?</strong></summary>

The KV cache, not the weights. KV per token depends on KV heads, head size, and how many layers keep the full context, not on total parameters. Gemma 2 9B, for example, has 8 KV heads of size 256 across 42 layers: 336 KiB per token if every layer kept everything, versus 128 KiB for Llama 3 8B. I would read the config for num_key_value_heads, head_dim, layer_types and sliding_window, then compute 2 × n_kv × d × bytes × tokens kept, summed over layers. Then check the serving engine: if it doesn't implement the model's sliding-window layers, it may allocate full-context cache for all of them. Fixes are an FP8 KV cache, shorter max context, lower concurrency, or an engine that supports the layer types properly.

</details>

<details>
<summary><strong>Mixtral is called 8x7B but has 46.7B parameters and 12.9B active. Explain, and say what it costs to serve.</strong></summary>

Only the MLPs are replicated. Each of the 32 layers has 8 expert MLPs of about 176M parameters, while attention, norms and embeddings are shared, so the total is about 46.7B, not 8 × 7.2B = 58B. Each token goes through 2 experts, so per token it runs attention plus 2 experts per layer plus embeddings, about 12.9B. Compute per token therefore resembles a 13B dense model. Memory is the catch: every expert must be resident, about 93 GB in BF16, so it needs two 80 GB GPUs, or FP8 or 4-bit weights to fit on one. At small batch it decodes quickly because few experts are read per token; at large batch most experts are touched each step, so it behaves more like a 47B model on memory bandwidth.

</details>
