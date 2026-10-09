<!-- GENERATED from llm-inference.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: LLM inference

Covers the lesson [LLM inference](../lessons/18-llm-inference/01-llm-inference.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/llm-inference/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

A synthetic model has 40 layers, 40 query heads, 8 KV heads, and $d_{\text{head}} = 128$. The KV cache is stored in BF16 (2 bytes). How many bytes of KV cache does **one token** of **one sequence** need?

<details>
<summary>Answer</summary>

**163840**

$2 \cdot L \cdot n_{kv} \cdot d_{\text{head}} \cdot b = 2 \cdot 40 \cdot 8 \cdot 128 \cdot 2$. Step by step: $2 \cdot 40 = 80$; $80 \cdot 8 = 640$; $640 \cdot 128 = 81{,}920$; $81{,}920 \cdot 2 = 163{,}840$ bytes (160 KiB). The 40 query heads do not enter the formula.

</details>

## 2. Calculation (medium)

The lesson's synthetic 8B-style GQA model stores 131,072 bytes of KV per token in BF16. How many GB ($10^9$ bytes) of KV cache does a batch of 8 sequences at 16,384 tokens each need? (2 decimals)

<details>
<summary>Answer</summary>

**17.18** (within ±0.01)

$131{,}072 \cdot 16{,}384 = 2{,}147{,}483{,}648$ bytes per sequence (2.147 GB). Times 8: $17{,}179{,}869{,}184$ bytes $= 17.18$ GB. Same as batch 16 at 8k: the cache depends on total tokens held, $T \cdot B$.

</details>

## 3. Calculation (medium)

A synthetic 13-billion-parameter model with FP8 weights (1 byte per parameter) runs at batch 1 with a short context on an accelerator with 2.6 TB/s memory bandwidth. Ignoring the KV cache, what is the upper bound on decode speed in tokens per second?

<details>
<summary>Answer</summary>

**200** (within ±1)

Weights: $13 \times 10^9 \cdot 1 = 13$ GB read per token. $2.6 \times 10^{12} / 13 \times 10^9 = 200$ tokens/s, or 5 ms per token. In BF16 it would be 26 GB and 100 tokens/s.

</details>

## 4. Calculation (hard)

Speculative decoding with draft length $\gamma = 3$ and an independent per-token acceptance rate $\alpha = 0.6$. What is the expected number of tokens produced per target-model forward pass? (3 decimals)

<details>
<summary>Answer</summary>

**2.176** (within ±0.002)

$E = (1 - \alpha^{\gamma+1})/(1 - \alpha) = (1 - 0.6^4)/0.4$. $0.6^4 = 0.1296$. $(1 - 0.1296)/0.4 = 0.8704/0.4 = 2.176$. Equivalently $1 + 0.6 + 0.36 + 0.216 = 2.176$.

</details>

## 5. Multiple choice (easy)

Why is single-user decode limited by memory bandwidth rather than by compute?

- **A.** Each step generates one token per sequence, so every weight is read from memory and used for only about 2 FLOPs, far below the accelerator's FLOPs-per-byte ridge point.
- **B.** Attention over the context has quadratic cost in decode.
- **C.** The softmax over the vocabulary is the slowest operation.
- **D.** Decode runs on the CPU while prefill runs on the GPU.

<details>
<summary>Answer</summary>

**A.** Each step generates one token per sequence, so every weight is read from memory and used for only about 2 FLOPs, far below the accelerator's FLOPs-per-byte ridge point.

Arithmetic intensity at batch 1 is about 1 FLOP per byte in BF16, while accelerators need hundreds of FLOPs per byte to be compute-bound. So time per token is about bytes read divided by bandwidth.

- **B:** With a KV cache, each decode step's attention is linear in context length, not quadratic. The quadratic term belongs to prefill.
- **C:** The vocabulary projection is one matrix among many; it is not what makes decode slow.
- **D:** Both phases run on the accelerator.

</details>

## 6. Multiple choice (hard)

A team quantizes a model's weights to INT4 and its KV cache to 4 bits. Chat quality scores barely move, but a contract-review feature that asks questions about 60-page documents starts missing clauses. What is the most likely cause and the best first step?

- **A.** The INT4 weights broke the tokenizer; retrain the tokenizer.
- **B.** Low-precision KV cache degrades long-context recall more than short chat; move the KV cache back to FP8 or BF16 and re-run a long-context retrieval eval.
- **C.** The model needs a higher temperature for long documents.
- **D.** Quantization lowered the context window; raise the maximum context length setting.

<details>
<summary>Answer</summary>

**B.** Low-precision KV cache degrades long-context recall more than short chat; move the KV cache back to FP8 or BF16 and re-run a long-context retrieval eval.

Errors in cached keys and values add up across thousands of positions, and retrieving one clause from a long document depends on precise attention scores. Short chat hides it. Evaluate quantization on the workloads you actually serve, separately for weights and KV.

- **A:** Quantization does not touch the tokenizer.
- **C:** Temperature changes randomness, not what the model can recall from context.
- **D:** Quantization does not change the configured context window.

</details>

## 7. Match (medium)

Match each attention variant to its KV heads and trade-off (synthetic model with 32 query heads).

| Concept | Options |
|---|---|
| Multi-head attention (MHA) | 1 KV head shared by all: smallest cache, can lose some quality |
| Grouped-query attention (GQA) | 8 KV heads shared by groups of 4 query heads: a quarter of the cache, close to MHA quality |
| Multi-query attention (MQA) | 32 KV heads: largest cache, the original design |

<details>
<summary>Answer</summary>

- Multi-head attention (MHA) → 32 KV heads: largest cache, the original design
- Grouped-query attention (GQA) → 8 KV heads shared by groups of 4 query heads: a quarter of the cache, close to MHA quality
- Multi-query attention (MQA) → 1 KV head shared by all: smallest cache, can lose some quality

KV cache scales with $n_{kv}$. GQA is the common compromise in current open-weight models.

</details>

## 8. Select all that apply (medium)

Which statements are correct?

- **A.** FlashAttention computes exact attention; it avoids writing the full $T \times T$ score matrix to GPU main memory.
- **B.** A mixture-of-experts model with 36B total and 8B active parameters needs about as much memory as a dense 8B model.
- **C.** Raising the batch size raises aggregate tokens/s but lowers each user's tokens/s.
- **D.** With rejection-sampling verification, speculative decoding produces the same output distribution as the target model alone.
- **E.** FlashAttention halves the size of the KV cache.

<details>
<summary>Answer</summary>

**A, C, D**

FlashAttention is exact and IO-aware. Batching trades per-user speed for total throughput. Speculative decoding with proper verification is lossless. MoE needs memory for all 36B parameters, and FlashAttention does not change the KV cache.

- **B:** Every expert must be resident, so memory follows total parameters (72 GB in BF16); compute follows active parameters.
- **E:** FlashAttention reduces memory traffic and the $O(T^2)$ score matrix, not the stored keys and values.

</details>

## 9. Calculation (easy)

A request has TTFT 400 ms and TPOT 30 ms and produces 101 output tokens. What is its end-to-end latency in seconds?

<details>
<summary>Answer</summary>

**3.4** (within ±0.01)

$0.400 + 0.030 \cdot (101 - 1) = 0.400 + 3.000 = 3.400$ s. The first token arrives at TTFT; each of the remaining 100 tokens takes one TPOT.

</details>

## 10. Reflection (hard)

Your team must pick between a dense 8B model and an MoE model with 36B total and 8B active parameters, for a high-traffic service on 80 GB GPUs. Explain the trade-off in memory, compute, and throughput.

<details>
<summary>Answer</summary>

**Model answer.** Per token, the MoE does the same compute as the dense 8B model, and it usually gives better quality because it has more total parameters. But all 36B parameters must be resident: 72 GB in BF16, which nearly fills an 80 GB GPU and leaves almost nothing for the KV cache. So it needs FP8 weights, more GPUs, or expert parallelism. At batch 1, decode reads only the active experts, so it is about as fast as the dense model. At high batch, tokens route to different experts, so nearly all experts are read every step and each step moves more bytes than the dense model's. The MoE wins on quality per FLOP; the dense model wins on memory and simplicity.

MoE trades memory (total parameters) for compute (active parameters).

</details>
