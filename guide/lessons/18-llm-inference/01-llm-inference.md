---
title: LLM inference
summary: Explain where the time and memory go when an LLM generates text (prefill vs decode, the KV cache, memory bandwidth) and estimate the effect of GQA, quantization, FlashAttention, speculative decoding, and mixture-of-experts.
skill: llms
minutes: 45
prerequisites: [transformer-architecture, decoding]
related: [serving-llms, self-attention, reliability-cost-and-observability, adapting-llms]
adapted_from: [phases/07-transformers-deep-dive/12-kv-cache-flash-attention, phases/07-transformers-deep-dive/16-speculative-decoding, phases/10-llms-from-scratch/11-quantization, phases/17-infrastructure-and-production/09-production-quantization]
---

# LLM inference

> **Mental model.** Generating text is two jobs. **Prefill** reads the whole prompt at once, like a chef doing all the prep in one go: lots of work, done in parallel, limited by how fast the chef can chop (compute). **Decode** then produces one token at a time, and for each token the chef has to walk to the pantry and carry every ingredient (all the weights and the saved notes on earlier tokens) back to the counter. Decode is limited by the walk (memory bandwidth), not the chopping.

**You will learn to**
- Distinguish prefill from decode and explain why decode is memory-bandwidth bound.
- Compute the KV-cache size for any model shape, context length, and batch size.
- Estimate an upper bound on decode speed from bandwidth and bytes read per step.
- Explain how MHA, MQA, and GQA, weight and KV quantization, FlashAttention, speculative decoding, and mixture-of-experts change memory, speed, or quality.
- Define TTFT, TPOT (inter-token latency), end-to-end latency, and throughput, and compute one from the others.

**Why it matters.** Every LLM product decision about context length, model size, batch size, hardware, and price comes down to a few lines of arithmetic. Engineers who can do that arithmetic can tell whether a model fits on a GPU, why a long-context feature halved throughput, and which optimization is worth trying first. Interviewers for ML and AI engineering roles ask for exactly this.

## 1. Intuition

**Two phases.** A request arrives with a prompt of, say, 2,000 tokens.

1. **[Prefill](../../glossary.md#prefill).** The model processes all 2,000 prompt tokens in one forward pass. Each weight matrix multiplies a $2000 \times d$ block of activations, so every weight loaded from memory is reused 2,000 times. The GPU's arithmetic units are the bottleneck. Prefill produces the first output token and fills the [KV cache](../../glossary.md#kv-cache). Its duration is most of the **time to first token**.
2. **Decode.** The model then generates one token per forward pass. Each pass multiplies a single row of activations by every weight matrix. Every weight is read from memory and used once. The arithmetic units mostly wait for data. This is why a GPU that can do hundreds of trillions of operations per second still produces only tens to hundreds of tokens per second for a single user.

**The KV cache.** In causal attention, a token's key and value vectors never change once computed ([self-attention](../14-transformers/01-self-attention.md)). So the model stores them: for every layer, every KV head, every past token. The next token computes only its own query, key, and value and attends over the cache. Without the cache, every new token would recompute the whole prefix. The price is memory: the cache grows linearly with context length and with the number of sequences in the batch, and at long contexts or large batches it is bigger than the model.

**Batching rescues decode.** If 16 users decode at once, each weight read from memory is used 16 times instead of once. The step takes only slightly longer, but produces 16 tokens. That is why servers batch, and why the KV cache (which grows with batch size) is what usually limits how many users fit on one GPU.

**The levers.** Every inference optimization attacks one of three costs:

| Lever | What it cuts | Cost |
|---|---|---|
| GQA / MQA (fewer KV heads) | KV-cache bytes per token | decided at training time; MQA can cost some quality |
| Weight quantization (INT8, FP8, INT4) | weight bytes read per step and stored | some accuracy loss, more at 4 bits |
| KV-cache quantization (FP8, 4-bit) | KV bytes stored and read | small accuracy loss, worse on long-context recall |
| FlashAttention | attention memory traffic (prefill especially) | none in quality: it is exact |
| Speculative decoding | number of sequential target-model passes | extra compute; gains depend on acceptance rate |
| Mixture-of-experts | compute per token | memory still holds every expert |

## 2. Visualization

<!-- lab:kv-cache -->
![Left: memory used by weights plus KV cache for a synthetic 8B-style model at batch 16 as context grows from 1k to 32k tokens. Multi-head attention with 32 KV heads crosses an 80 GB line before 8k tokens; grouped-query attention with 8 KV heads reaches about 85 GB at 32k; multi-query attention with 1 KV head stays near the 16 GB weights line. Right: at 8k context and 3.0 TB/s, the upper bound on aggregate decode throughput rises from 176 tokens per second at batch 1 to about 2,100 at batch 48, while per-sequence speed falls from 176 to 44 tokens per second.](../../figures/llm-inference.png)

*Synthetic model shape (32 layers, 32 query heads, 128-dimensional heads, 8 billion parameters, BF16) and a synthetic 80 GB, 3.0 TB/s accelerator. The left panel is why modern models use GQA. The right panel is the batching trade-off: more total tokens per second, fewer per user.*

*Interactive version: pick a model shape, context length, batch, precisions, and accelerator, and see whether it fits and how fast decode can go. [Open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/kv-cache/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Start from the lesson example (8B-style, 8k tokens, batch 16). Predict the KV-cache size if you switch attention to MHA. Then switch and check the multiple.
2. Pick **70B on one GPU**. Predict whether INT4 weights alone make it fit on the 80 GB accelerator at batch 8 and 8k context. Then try it, and try FP8 KV as well.
3. Raise the batch from 1 to 64 at 8k context. Predict which number rises and which falls: aggregate tokens/s or per-sequence tokens/s. Where does memory run out?
4. Switch to the **Small MHA** preset. Compare its KV bytes per token with the 8B-style GQA model. Which model has the bigger cache per token, and why?

## 3. The math

### Symbols

| Symbol | Meaning | Example (8B-style) |
|---|---|---|
| $L$ | number of transformer layers | 32 |
| $n_h$ | number of query heads | 32 |
| $n_{kv}$ | number of key/value heads | 8 |
| $d_{\text{head}}$ | dimension of each head | 128 |
| $b$ | bytes per stored number (BF16 = 2, FP8 = 1, INT4 = 0.5) | 2 |
| $T$ | tokens in context (prompt plus generated so far) | 8,192 |
| $B$ | sequences in the batch | 16 |
| $P$ | parameter count | $8 \times 10^9$ |
| $\text{BW}$ | memory bandwidth, bytes per second | $3.0 \times 10^{12}$ |
| $F$ | peak compute, FLOPs per second | $1000 \times 10^{12}$ |

At each layer, the cache holds a key tensor and a value tensor of shape $[B, n_{kv}, T, d_{\text{head}}]$.

### KV-cache size

```math
\text{KV bytes} = 2 \cdot L \cdot n_{kv} \cdot d_{\text{head}} \cdot b \cdot T \cdot B
```

The leading 2 counts K and V. Note what is *not* in the formula: $n_h$ and the parameter count. Only the number of KV heads matters.

**Worked example** (synthetic 8B-style GQA model, BF16 cache).

Per token: $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2$. Step by step: $2 \cdot 32 = 64$; $64 \cdot 8 = 512$; $512 \cdot 128 = 65{,}536$; $65{,}536 \cdot 2 = 131{,}072$ bytes = 128 KiB per token.

One sequence of $T = 8192$: $131{,}072 \cdot 8192 = 1{,}073{,}741{,}824$ bytes = 1.074 GB (GB means $10^9$ bytes).

Batch of 16: $1{,}073{,}741{,}824 \cdot 16 = 17{,}179{,}869{,}184$ bytes = 17.18 GB.

Weights in BF16: $8 \times 10^9 \cdot 2 = 16$ GB. Total: $16 + 17.18 = 33.18$ GB, which fits on an 80 GB accelerator. The largest batch that fits at 8k context (ignoring activations and overhead) is $\lfloor (80 - 16) / 1.074 \rfloor = \lfloor 59.6 \rfloor = 59$.

### MHA, MQA, GQA

- **Multi-head attention (MHA):** $n_{kv} = n_h = 32$. Cache per token $= 4 \times 131{,}072 = 524{,}288$ bytes. At $T = 8192$, $B = 16$ that is 68.72 GB, more than four times the weights.
- **Multi-query attention (MQA):** $n_{kv} = 1$, all query heads share one K and one V. $16{,}384$ bytes per token, 2.15 GB for the same batch. Smallest cache, but sharing a single KV head can cost quality.
- **[Grouped-query attention](../../glossary.md#grouped-query-attention) (GQA):** $n_{kv} = 8$, each group of $32 / 8 = 4$ query heads shares one KV head. 17.18 GB. Close to MHA quality at a quarter of the cache, which is why most current open-weight models use it.

Grouping does not change the parameter count much (the K and V projections shrink a little); it changes the cache.

### Why decode is memory-bandwidth bound

A forward pass for one token does about $2P$ floating-point operations (one multiply and one add per parameter). At batch 1, it reads every weight once: $P \cdot b$ bytes. **Arithmetic intensity** is FLOPs per byte read:

```math
I_{\text{decode},\,B=1} = \frac{2P}{P \cdot b} = \frac{2}{2} = 1 \text{ FLOP/byte (BF16)}
```

An accelerator can only keep its arithmetic units busy if intensity exceeds its **ridge point**, $F / \text{BW}$. For the synthetic accelerator: $1000 \times 10^{12} / 3.0 \times 10^{12} = 333.3$ FLOP/byte. Decode at batch 1 sits at 1, so it uses roughly $1/333 \approx 0.3\%$ of peak compute. That is the roofline picture: below the ridge point, time is set by bytes moved, not FLOPs done.

With batch $B$, the weights are read once and used $B$ times, so intensity is about $B$ FLOP/byte for the weight reads. Prefill of $N$ prompt tokens has intensity about $N$, which is why a 2,000-token prefill is compute-bound.

### Upper bound on decode speed

Each decode step reads all weights plus every sequence's KV cache, and produces one token per sequence:

```math
\text{steps/s} \le \frac{\text{BW}}{P \cdot b_w + \text{KV bytes}}, \qquad \text{aggregate tokens/s} \le B \cdot \text{steps/s}
```

Worked example at $\text{BW} = 3.0 \times 10^{12}$ bytes/s:

- Batch 1, short context (ignore KV): $3.0 \times 10^{12} / 16 \times 10^9 = 187.5$ tokens/s, or $1000 / 187.5 = 5.33$ ms per token.
- Batch 1, $T = 8192$: $3.0 \times 10^{12} / (16 + 1.074) \times 10^9 = 3000 / 17.074 = 175.7$ tokens/s.
- Batch 16, $T = 8192$: $3000 / (16 + 17.18) = 3000 / 33.18 = 90.4$ steps/s. Each user sees 90.4 tokens/s; the server produces $90.4 \times 16 = 1{,}447$ tokens/s.

Real systems reach a fraction of these bounds (kernel overheads, attention compute, communication), but the ratios hold: batching multiplies total throughput while each user slows down, and long contexts slow everyone because the KV cache must be read every step.

**Prefill lower bound.** A 2,000-token prompt needs about $2 \cdot 8 \times 10^9 \cdot 2000 = 3.2 \times 10^{13}$ FLOPs. At $10^{15}$ FLOP/s that is 32 ms if the GPU ran at peak. In practice utilization is lower, and attention adds a term that grows with $T^2$.

### Quantization

Weight memory is $P \cdot b_w$: 16 GB in BF16, 8 GB in FP8 or INT8, 4 GB in INT4 (plus a few percent for per-group scale factors). Because decode reads every weight each step, halving weight bytes nearly doubles the batch-1 speed bound. Common forms:

- **Weight-only INT8/INT4:** weights stored in low precision with scales per channel or per group of, say, 128 values, dequantized on the fly; activations stay 16-bit. Best for memory-bound decode. INT4 usually needs a calibration-based method to keep accuracy.
- **FP8 (weights and activations):** an 8-bit floating-point format supported natively by recent accelerators. It speeds up compute-bound prefill too.
- **KV-cache quantization (FP8 or 4-bit):** halves or quarters the cache. FP8 cache at our example: $17.18 / 2 = 8.59$ GB.

What you lose: small, task-dependent accuracy drops, usually largest on math, code, and long-context retrieval, and larger at 4 bits than at 8. Always re-run your own evals after quantizing ([evaluation](../17-llm-evaluation/01-llm-evaluation.md)).

### FlashAttention

Naive attention writes the full score matrix $QK^\top$ of shape $[T, T]$ to GPU main memory (HBM), reads it back for softmax, and again to multiply by $V$. At $T = 8192$ in 16-bit, one head's score matrix is $8192^2 \cdot 2 = 134{,}217{,}728$ bytes (134 MB); 32 heads make 4.29 GB for a single layer. **[FlashAttention](../../glossary.md#flashattention)** splits $Q$, $K$, and $V$ into tiles that fit in fast on-chip SRAM, computes softmax incrementally with a running max and running sum (online softmax), and never writes the $T \times T$ matrix to HBM. The result is **exact**, not an approximation. Attention memory drops from $O(T^2)$ to $O(T)$, and memory traffic falls enough that long-context prefill runs several times faster. It does not shrink the KV cache.

### Speculative decoding

In [speculative decoding](../../glossary.md#speculative-decoding), a small **draft model** proposes $\gamma$ tokens cheaply. The large **target model** checks all $\gamma$ in one forward pass (a short prefill, so it costs about the same as one decode step because decode is memory-bound). Tokens are accepted left to right until the first rejection; at the rejection the target supplies its own token. With a rejection-sampling acceptance rule, the output distribution is exactly the target model's. If each draft token is accepted with probability $\alpha$ independently, the expected tokens per target pass is

```math
E[\text{tokens}] = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}
```

**Worked example:** $\alpha = 0.8$, $\gamma = 4$. $0.8^5 = 0.32768$. $E = (1 - 0.32768) / (1 - 0.8) = 0.67232 / 0.2 = 3.3616$ tokens per target pass. If one draft pass costs $c = 0.05$ of a target pass, a round costs $1 + 4 \cdot 0.05 = 1.2$ target passes, so the speedup is $3.3616 / 1.2 = 2.80\times$. At $\alpha = 0.5$ it falls to $1.61\times$. Acceptance depends on how predictable the text is: code and boilerplate accept well, creative text at high temperature less. Variants use n-gram lookups from the prompt or extra prediction heads instead of a separate draft model.

### Mixture-of-experts

A **[mixture-of-experts](../../glossary.md#mixture-of-experts)** (MoE) layer replaces one feed-forward block with $E$ expert blocks and a router that sends each token to the top $k$. Synthetic example: 4B shared parameters (attention, embeddings, router) plus 16 experts of 2B each, top-2 routing.

- Total parameters: $4 + 16 \cdot 2 = 36$B. Memory in BF16: 72 GB. All of it must be resident.
- Active parameters per token: $4 + 2 \cdot 2 = 8$B. Compute per token: $2 \cdot 8 \times 10^9 = 1.6 \times 10^{10}$ FLOPs, the same as a dense 8B model.

So MoE buys the quality of a larger model at the compute of a smaller one, but pays in memory. At batch 1 decode reads only the active experts; at large batch, different tokens route to different experts, so nearly every expert is read each step.

### Latency metrics

| Metric | Definition |
|---|---|
| [TTFT (time to first token)](../../glossary.md#time-to-first-token) | request arrival to first output token: queueing plus prefill |
| TPOT / ITL (time per output token, inter-token latency) | average gap between output tokens during decode |
| End-to-end latency | TTFT $+$ TPOT $\cdot (n_{\text{out}} - 1)$ |
| Throughput | output tokens per second across all requests (sometimes requests per second) |

Worked example: TTFT 200 ms, TPOT 25 ms, 300 output tokens. $0.200 + 0.025 \cdot 299 = 0.200 + 7.475 = 7.675$ s. The user reads at $1 / 0.025 = 40$ tokens/s. Long prompts hurt TTFT; long outputs and big batches hurt end-to-end latency through TPOT.

## 4. Implementation

The arithmetic fits in a few lines, and it is worth keeping as a script next to any deployment plan.

```python
def kv_cache_bytes(layers, n_kv, d_head, bytes_per, tokens, batch):
    return 2 * layers * n_kv * d_head * bytes_per * tokens * batch   # K and V

def decode_tokens_per_s(weight_bytes, kv_bytes, bandwidth, batch):
    steps = bandwidth / (weight_bytes + kv_bytes)    # every step reads all weights + all KV
    return steps, steps * batch                      # per sequence, aggregate (upper bounds)

kv = kv_cache_bytes(32, 8, 128, 2, 8192, 16)        # 17,179,869,184 bytes = 17.18 GB
print(decode_tokens_per_s(16e9, kv, 3.0e12, 16))    # (90.4, 1446.7)
```

### Build it from scratch

The arithmetic above predicts; now build the three mechanisms and check them. The full script is [`code/18-llm-inference/kv_spec_quant_scratch.py`](../../code/18-llm-inference/kv_spec_quant_scratch.py) (NumPy only, synthetic weights).

**The KV cache.** Attention for the newest token needs the keys and values of every earlier token. Without a cache, each step projects the whole prefix again. Both versions share one attention function over a query vector `q` of shape `[D]` and cached `K`, `V` of shape `[t, D]`:

```python
def attend(q, K, V):
    s = K @ q / np.sqrt(D)  # [t] one score per cached position
    w = np.exp(s - s.max())
    return (w / w.sum()) @ V  # [D]

def decode_uncached(X):
    outs, proj = [], 0
    for t in range(1, len(X) + 1):
        K, V = X[:t] @ Wk, X[:t] @ Wv  # recompute K and V for the whole prefix
        proj += 2 * t
        outs.append(attend(X[t - 1] @ Wq, K, V))
    return np.array(outs), proj
```

*Predict:* for a 12-token sequence, how many K/V row projections does this do in total?

The cached version keeps `K` and `V` and appends one row each per step. Past rows never change, because in a causal model a token cannot see the future.

```python
def decode_cached(X):
    outs, K, V, proj = [], np.empty((0, D)), np.empty((0, D)), 0
    for x in X:
        K, V = np.vstack([K, x @ Wk]), np.vstack([V, x @ Wv])  # append one row each
        proj += 2
        outs.append(attend(x @ Wq, K, V))
    return np.array(outs), proj
```

*Predict:* the outputs match the uncached version to about 1e-16. What would break that equality if the model were bidirectional instead of causal?

**int8 weight quantization.** Store each weight as an 8-bit integer plus a floating-point scale. Absmax scaling maps the largest magnitude to 127. The only design choice is how many weights share one scale: the whole tensor, or one row (output channel).

```python
def quant_int8(W, axis=None):
    scale = np.abs(W).max(axis=axis, keepdims=axis is not None) / 127  # absmax maps to 127
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale

def rel_error(W, W_hat):
    return np.linalg.norm(W - W_hat) / np.linalg.norm(W)
```

*Predict:* the script scales three rows of a `[64, 256]` matrix by 40×. Which choice of `axis` suffers, and why? (One shared scale is set by the outliers, so every small weight is rounded on a coarse grid.)

**Speculative sampling.** The draft proposes a token from its distribution $p$; the target's distribution is $q$. Accept with probability $\min(1, q/p)$. On rejection, sample from the leftover mass $(q - p)_+$, renormalized. That pair of rules makes the output distribution exactly $q$, so the speed-up is free of quality loss.

```python
def spec_one_token(q, p, rng):
    d = rng.choice(len(p), p=p)  # draft proposes from p
    if rng.random() < min(1.0, q[d] / p[d]):
        return d, True  # target agrees often enough: keep the draft token
    resid = np.maximum(q - p, 0)  # where the target wants more mass than the draft gave
    return rng.choice(len(q), p=resid / resid.sum()), False

def expected_tokens(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)
```

*Predict:* the exact acceptance rate is $\alpha = \sum_i \min(p_i, q_i)$. If the draft equals the target ($p = q$), what are $\alpha$ and the expected tokens per target pass for $\gamma = 4$?

At temperature 0 both distributions are one-hot, so the rule reduces to greedy matching: accept draft tokens while each equals the target's argmax, then take the target's token at the first mismatch.

### Use the library

With a real model, read the shape from its config instead of guessing: `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, and `head_dim` (or `hidden_size / num_attention_heads`).

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
d_head = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
print(kv_cache_bytes(cfg.num_hidden_layers, cfg.num_key_value_heads, d_head, 2, 8192, 16) / 1e9, "GB")

# Assisted (speculative) generation: a small draft from the same tokenizer family
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto")
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")
inputs = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")("Explain KV caches.", return_tensors="pt").to(target.device)
out = target.generate(**inputs, assistant_model=draft, max_new_tokens=256)
```

A serving engine does all three for you. In vLLM, `LLM(model=..., quantization="fp8", kv_cache_dtype="fp8")` quantizes weights and the KV cache, and `speculative_config={"model": draft_name, "num_speculative_tokens": 4}` turns on draft-model speculation. The KV cache is always on. What the scratch version made explicit: the cache is just appended rows, a quantized weight is just an integer times a scale, and speculation is just one accept test plus a residual. Engines add fused GPU kernels, paged cache memory, and calibrated scales (AWQ, GPTQ, FP8) that protect salient weights better than plain absmax.

Then measure: run your serving engine with a fixed prompt and output length, record TTFT and TPOT at batch 1, 8, and 32, and compare with the bounds.

### Run it

From `guide/`:

```bash
uv run code/18-llm-inference/kv_spec_quant_scratch.py
```

```text
[kv] max |uncached - cached| = 4.4e-16
[kv] K/V projections: uncached 156, cached 24 (6.5x fewer with cache)
[quant] int8 per-tensor : relative error 0.0325
[quant] int8 per-channel: relative error 0.0067
[spec] acceptance: theory 0.840, simulated 0.841
[spec] max |empirical - target| = 0.0016  (output follows q, not p)
[spec] expected tokens per target pass, gamma=4: 3.64
```

Uncached work grows as $2(1 + 2 + \dots + 12) = 156$ projections; cached work is $2 \cdot 12 = 24$. Per-channel scales cut the error about 5× because each row gets its own grid. The empirical token frequencies match the target $q$ to within 0.002 across 50,000 samples, while 84% of draft tokens are kept.

The arithmetic script (every number in this lesson, plus the figure) is [`code/18-llm-inference/llm_inference.py`](../../code/18-llm-inference/llm_inference.py): `uv run code/18-llm-inference/llm_inference.py`.

## 5. Engineering

**Size before you buy.** Weights + KV cache at your p95 context length and target batch + headroom for activations and fragmentation (often 10 to 20%). If the weights alone exceed one GPU, you need quantization or tensor parallelism ([next lesson](02-serving-llms.md)).

**Know which phase hurts.** Long prompts and short answers (RAG, classification, summarization) are prefill-heavy: TTFT and compute matter, and prefix caching and FP8 compute help. Short prompts and long answers (chat, code generation, reasoning traces) are decode-heavy: bandwidth, batch size, KV size, and speculative decoding matter.

**Order of optimizations for decode.** Use a model with GQA; quantize weights to FP8 or INT8 (low risk); batch continuously; quantize the KV cache if memory caps your batch; try INT4 weights and speculative decoding last, after measuring quality and acceptance rates on your own traffic.

**Reasoning models change the budget.** Models that think for thousands of tokens before answering are decode-dominated and hold large KV caches for a long time. Plan capacity on output tokens, not request counts.

> [!WARNING]
> **Failure modes.** Out-of-memory errors that appear only when several long requests coincide; a quantized model that passes chat evals but regresses on math or long-document recall; speculative decoding that slows things down at high batch (the GPU is no longer idle, so the extra draft and verification work is not free) or on low-acceptance traffic; MoE models that fit the compute budget but not the memory budget.

### Common mistakes

- Using $n_h$ instead of $n_{kv}$ in the KV-cache formula, which overestimates a GQA model's cache by $n_h / n_{kv}$.
- Forgetting the factor of 2 for keys and values, or forgetting the batch dimension.
- Quoting FLOPs to predict decode speed. Decode speed follows bandwidth at small batch.
- Treating "tokens/s" as one number: per-user speed and aggregate throughput move in opposite directions as batch grows.
- Assuming an MoE model needs only its active parameters in memory.
- Thinking FlashAttention approximates attention or shrinks the KV cache. It is exact and it reduces attention's memory traffic, not the cache.

## 6. Knowledge check

<!-- quiz:llm-inference -->
**[Take the LLM inference quiz](../../quizzes/llm-inference.md)**
<!-- /quiz -->

**Practice exercise.** A synthetic 70B-style model has 80 layers, 64 query heads, 8 KV heads, and $d_{\text{head}} = 128$. Compute its BF16 KV cache per token, then for one 32k-token ($T = 32{,}768$) sequence, and for a batch of 4. With BF16 weights, how many 80 GB GPUs do you need at minimum? And with FP8 weights?

<details>
<summary>Solution</summary>

Per token: $2 \cdot 80 \cdot 8 \cdot 128 \cdot 2 = 327{,}680$ bytes (320 KiB). One sequence: $327{,}680 \cdot 32{,}768 = 10{,}737{,}418{,}240$ bytes = 10.74 GB. Batch of 4: 42.95 GB. BF16 weights: $70 \times 10^9 \cdot 2 = 140$ GB. Total 182.95 GB, so three 80 GB GPUs (240 GB) at minimum, before activations. FP8 weights: 70 GB, total 112.95 GB, which fits on two (160 GB). Query heads never enter the calculation.
</details>

**Implementation challenge.** Write `max_context(model, gpu_gb, batch, weight_bytes, kv_bytes, headroom=0.1)` that returns the longest context that fits, and `speculative_speedup(alpha, gamma, c)`. Find the $\gamma$ that maximizes speedup for $\alpha \in \{0.6, 0.8, 0.9\}$ with $c = 0.05$.

<details>
<summary>Solution sketch</summary>

```python
def max_context(L, n_kv, d_head, params, gpu_gb, batch, wb, kvb, headroom=0.1):
    free = gpu_gb * 1e9 * (1 - headroom) - params * wb
    return max(0, int(free // (2 * L * n_kv * d_head * kvb * batch)))

def speculative_speedup(a, g, c):
    return (1 - a ** (g + 1)) / (1 - a) / (1 + g * c)

for a in (0.6, 0.8, 0.9):
    best = max(range(1, 16), key=lambda g: speculative_speedup(a, g, 0.05))
    print(a, best, round(speculative_speedup(a, best, 0.05), 2))
```

Higher acceptance rates justify longer drafts. With $c = 0.05$ the best $\gamma$ is 4 at $\alpha = 0.6$ (1.92×), 8 at $\alpha = 0.8$ (3.09×), and 13 at $\alpha = 0.9$ (4.67×). The curves are flat near the optimum, and past it extra draft tokens are rarely accepted but still cost draft passes. Real acceptance is not independent across positions, so tune $\gamma$ on measured traffic.
</details>

## Summary

- Prefill processes the prompt in parallel and is compute-bound; decode generates one token per pass and is memory-bandwidth bound.
- KV cache $= 2 \cdot L \cdot n_{kv} \cdot d_{\text{head}} \cdot b \cdot T \cdot B$; GQA and MQA shrink it by cutting $n_{kv}$, and KV quantization by cutting $b$.
- Decode speed is bounded by bandwidth divided by bytes read per step (weights plus all KV). Batching raises total throughput and lowers per-user speed.
- Quantization trades a little accuracy for fewer bytes; FlashAttention is exact and cuts attention memory traffic; speculative decoding trades extra compute for fewer sequential passes; MoE trades memory for compute.
- Report TTFT, TPOT, end-to-end latency, and throughput separately; they respond to different levers.

**Next:** [Serving LLMs](02-serving-llms.md)

**Related:** [Self-attention](../14-transformers/01-self-attention.md) · [The transformer architecture](../14-transformers/02-transformer-architecture.md) · [Decoding](../15-llms/02-decoding.md) · [Reliability, cost, and observability](../20-production-ai/02-reliability-cost-and-observability.md) · [Cheat sheet: LLM inference](../../cheatsheets/llm-inference.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Inference Optimization](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/12-inference-optimization) · [Speculative Decoding and EAGLE-3](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/15-speculative-decoding-eagle3) · [Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson](https://aiengineeringfromscratch.com/lesson?path=phases/17-infrastructure-and-production/12-edge-inference)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [KV Cache, Flash Attention & Inference Optimization](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/12-kv-cache-flash-attention), [Speculative Decoding — Draft, Verify, Repeat](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/16-speculative-decoding), [Quantization: Making Models Fit](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/11-quantization), [Production Quantization — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4](https://aiengineeringfromscratch.com/lesson?path=phases/17-infrastructure-and-production/09-production-quantization).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Why is LLM decoding memory-bandwidth bound when prefill is compute-bound?</strong></summary>

It comes down to arithmetic intensity. In decode, each forward pass handles one new token per sequence, so every weight matrix multiplies a single activation row. Each weight is read from HBM and used for about two FLOPs, which is roughly 1 FLOP per byte in BF16. A modern accelerator needs a few hundred FLOPs per byte to keep its arithmetic units busy, so at batch 1 they sit idle waiting on memory, and time per token is about (weight bytes + KV bytes) ÷ bandwidth. Prefill pushes the whole prompt through at once. With $N$ prompt tokens, each weight is reused $N$ times, intensity is about $N$, and the bottleneck becomes FLOPs. That's why batching helps decode a lot (it reuses each weight read across $B$ sequences) and does little for prefill.

</details>

<details>
<summary><strong>You need to double the context length you serve on the same GPUs. Do you quantize the weights or the KV cache?</strong></summary>

At long context I'd quantize the KV cache first, because that's the part that grows with context. Weights are a fixed cost. For an 8B-style GQA model, BF16 weights are 16 GB, while a batch of 16 at 8k context already holds 17 GB of KV, and doubling the context doubles that to 34 GB. FP8 KV halves the cache and usually costs little accuracy, though I'd check long-context retrieval evals specifically, because that's where cache precision shows up. Weight quantization frees a one-time 8 GB (BF16 to FP8) and speeds up decode at small batch, so it's a good next step. If the model has full multi-head attention, no quantization helps as much as moving to a GQA variant, which cuts the cache by $n_h / n_{kv}$.

</details>

<details>
<summary><strong>After launching a 32k-context feature, throughput per GPU halved and you see occasional out-of-memory errors. What do you check?</strong></summary>

First the KV-cache budget. Per-sequence cache scales with $T$, so going from 8k to 32k quadruples it, and the scheduler can fit far fewer concurrent sequences. Fewer sequences means a smaller effective batch, which lowers throughput. I'd check the engine's reported KV-block usage, how often requests are preempted or swapped, and the p95 prompt length, not just the mean. The OOMs suggest memory is being reserved for the configured maximum length, or activation headroom is too thin when several long prefills land together. Fixes in order: cap or chunk prefill, enable FP8 KV, make sure prefix caching shares common system prompts, set the max-concurrency limit from the real KV budget, and route long-context requests to a separate pool so they don't starve short ones.

</details>

<details>
<summary><strong>Quick estimate: 70B-style model, 80 layers, 8 KV heads, head dimension 128, BF16, 32k context, batch 4. How much memory?</strong></summary>

KV per token is $2 \cdot 80 \cdot 8 \cdot 128 \cdot 2 = 327{,}680$ bytes, about 320 KiB. One 32,768-token sequence is $327{,}680 \times 32{,}768 \approx 10.7$ GB, so batch 4 is about 43 GB. BF16 weights are $70\text{B} \times 2 = 140$ GB, giving about 183 GB before activations and overhead. That needs at least three 80 GB GPUs, and in practice four with tensor parallelism, since TP degrees are usually powers of two. With FP8 weights it's 70 + 43 = 113 GB, which fits on two 80 GB GPUs. The decode bound at 2 × 3 TB/s is roughly 6e12 ÷ 113e9 ≈ 53 steps/s, so about 53 tokens/s per user and 212 tokens/s in aggregate.

</details>
