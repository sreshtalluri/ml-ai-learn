---
title: LLM inference
summary: KV-cache and decode-speed formulas, inference optimizations, serving techniques, latency metrics, and cost per token on one page.
---

# LLM inference

**KV cache:** $2 \cdot L \cdot n_{kv} \cdot d_{\text{head}} \cdot b \cdot T \cdot B$ bytes (K and V, every layer, every KV head, every token, every sequence). Uses $n_{kv}$, never $n_h$.

**Weights:** $P \cdot b_w$. 8B parameters: 16 GB BF16, 8 GB FP8/INT8, 4 GB INT4 (plus scales).

**Decode speed limit:** steps/s $\le \text{BW} / (\text{weight bytes} + \text{all KV bytes})$; aggregate tokens/s $= B \times$ steps/s.

**Arithmetic intensity:** decode at batch $B$ ≈ $B$ FLOP/byte (BF16 weights); prefill of $N$ tokens ≈ $N$. Compute-bound only above the ridge point $F / \text{BW}$ (a few hundred on current accelerators).

**Worked numbers** (synthetic 8B-style GQA: $L = 32$, $n_{kv} = 8$, $d_{\text{head}} = 128$, BF16; 80 GB, 3.0 TB/s):

| Quantity | Value |
|---|---|
| KV per token | $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072$ B (128 KiB) |
| KV, 8k context, batch 16 | 17.18 GB (MHA: 68.72; MQA: 2.15; FP8 KV: 8.59) |
| Max batch at 8k | $\lfloor (80 - 16)/1.074 \rfloor = 59$ |
| Decode bound, batch 1 | $3000 / 16 = 187.5$ tokens/s |
| Decode bound, batch 16 at 8k | 90.4 tokens/s per user, 1,447 total |

| Technique | Saves | Costs or caveat |
|---|---|---|
| GQA / MQA | KV memory by $n_h / n_{kv}$ | set at training; MQA can lose quality |
| Weight quantization (FP8, INT8, INT4) | memory and decode time | accuracy loss, larger at 4 bits; re-run evals |
| KV-cache quantization (FP8, 4-bit) | KV memory, KV reads | long-context recall most sensitive |
| FlashAttention | attention memory traffic, $O(T^2) \to O(T)$ memory | exact; does not shrink the KV cache |
| Speculative decoding | sequential target passes | extra compute; helps at low batch and high acceptance |
| Mixture-of-experts | compute per token (active params) | memory holds all experts |
| Continuous batching | idle slots | scheduler complexity |
| Paged KV (block tables) | reserved-but-unused KV memory | under one block of waste per sequence |
| Prefix caching | prefill for shared prefixes | stable content must come first in the prompt |
| Chunked prefill / disaggregation | decode stalls behind long prompts | slightly higher TTFT for the long prompt; KV transfer |

**Speculative decoding:** $E[\text{tokens per target pass}] = (1 - \alpha^{\gamma+1})/(1 - \alpha)$; speedup $= E / (1 + \gamma c)$. $\alpha = 0.8$, $\gamma = 4$, $c = 0.05$: $3.36 / 1.2 = 2.80\times$.

**Latency:** TTFT = queue + prefill. TPOT (ITL) = mean gap between tokens. End-to-end = TTFT + TPOT $\cdot (n_{\text{out}} - 1)$. Report p50 and p95, separately.

**Little's law:** requests in flight = arrival rate × time in system. Use it to size batch and KV memory.

**Parallelism:** replicas for throughput · tensor parallel when it doesn't fit or latency is too high (fast intra-node links) · pipeline parallel across nodes · expert parallel for MoE.

**Cost:** cost per million tokens = hourly GPU price ÷ (tokens/s × 3600 ÷ 1,000,000) ÷ utilization. \$4/h at 1,500 tokens/s: \$0.741 at 100%, \$2.47 at 30%.

Lessons: [LLM inference](../lessons/18-llm-inference/01-llm-inference.md) · [Serving LLMs](../lessons/18-llm-inference/02-serving-llms.md)
