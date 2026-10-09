<!-- GENERATED from serving-llms.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Serving LLMs

Covers the lesson [Serving LLMs](../lessons/18-llm-inference/02-serving-llms.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/serving-llms/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

Two batch slots. Four requests arrive at step 0 with output lengths $[6, 2, 2, 2]$. Each busy slot emits one token per step. Under **static** batching (a new batch starts only when every slot is free, taking requests in order), what is the throughput in tokens per step?

<details>
<summary>Answer</summary>

**1.5** (within ±0.01)

Batch 1 = requests 0 and 1: finish at 6 and 2; slot 1 idles from 2 to 6. Batch 2 starts at 6: requests 2 and 3 finish at 8. Makespan 8, total tokens $6 + 2 + 2 + 2 = 12$, throughput $12 / 8 = 1.5$.

</details>

## 2. Calculation (medium)

Same four requests ($[6, 2, 2, 2]$, two slots, all arriving at step 0) under **continuous** batching. What is the mean latency in steps?

<details>
<summary>Answer</summary>

**4.5** (within ±0.01)

Request 0 runs 0 to 6. Request 1 runs 0 to 2, then request 2 takes that slot from 2 to 4, then request 3 from 4 to 6. Latencies $[6, 2, 4, 6]$, mean $18 / 4 = 4.5$ (static: $[6, 2, 8, 8]$, mean 6). Makespan drops from 8 to 6, so throughput rises from 1.5 to 2.0.

</details>

## 3. Calculation (hard)

A KV budget of 20 GB ($20 \times 10^9$ bytes), 131,072 bytes per token, 16-token blocks. Every request holds 1,000 tokens of context. How many sequences fit with paged allocation?

<details>
<summary>Answer</summary>

**151**

$\lceil 1000 / 16 \rceil = 63$ blocks $= 1008$ tokens. $1008 \cdot 131{,}072 = 132{,}120{,}576$ bytes per sequence. $20 \times 10^9 / 132{,}120{,}576 = 151.4$, so 151. Reserving a 4,096-token maximum instead would fit only $\lfloor 20 \times 10^9 / 536{,}870{,}912 \rfloor = 37$.

</details>

## 4. Calculation (medium)

A synthetic GPU costs USD 2.50 per hour and sustains 1,000 output tokens per second when busy. Average utilization is 50%. What is the cost in USD per million output tokens? (2 decimals)

<details>
<summary>Answer</summary>

**1.39** (within ±0.01)

$1000 \times 3600 = 3.6$ million tokens per busy hour. $2.50 / 3.6 = 0.694$ per million at 100%. At 50% utilization: $0.694 / 0.5 = 1.39$.

</details>

## 5. Calculation (easy)

A service receives 5 requests per second and each request spends 4 seconds in the system on average (queue plus generation). On average, how many requests are in flight?

<details>
<summary>Answer</summary>

**20**

Little's law: $5 \times 4 = 20$. Size the batch slots and KV memory for at least this many concurrent sequences, plus headroom for peaks.

</details>

## 6. Multiple choice (hard)

Users of a chat service report that streaming output "freezes" for a second or two at random moments. Traces show TPOT spikes that line up with the arrival of requests carrying 40k-token documents. TTFT for normal requests is fine. What is the most likely cause and fix?

- **A.** The KV cache is too small; buy GPUs with more memory.
- **B.** Long prompts are prefilled in one go, stalling every running sequence's next decode step; enable chunked prefill (or separate prefill and decode pools).
- **C.** Temperature is too high for long documents.
- **D.** Continuous batching is admitting too few requests; switch to static batching.

<details>
<summary>Answer</summary>

**B.** Long prompts are prefilled in one go, stalling every running sequence's next decode step; enable chunked prefill (or separate prefill and decode pools).

A single 40k-token prefill takes far longer than a decode step. If the scheduler runs it whole, everyone else's next token waits. Chunked prefill splits it into pieces interleaved with decode steps; disaggregation moves it to separate GPUs.

- **A:** A full cache would show up as queueing, preemptions, and rising TTFT, not as decode stalls timed with long arrivals.
- **C:** Temperature affects which token is chosen, not how long a step takes.
- **D:** Static batching would make waiting worse, not better.

</details>

## 7. Multiple choice (medium)

A model's BF16 weights need 140 GB. You have one node with four 80 GB GPUs on fast interconnect, and you want the lowest per-token latency. Which strategy fits best?

- **A.** Data parallelism with four full replicas.
- **B.** Tensor parallelism across the four GPUs in the node.
- **C.** Pipeline parallelism across four separate nodes.
- **D.** CPU offloading of half the layers.

<details>
<summary>Answer</summary>

**B.** Tensor parallelism across the four GPUs in the node.

Tensor parallelism splits every weight matrix, so each GPU stores and reads a quarter of the weights per step. That makes the model fit and adds aggregate bandwidth. The per-layer all-reduces are cheap on fast intra-node links.

- **A:** Each replica needs the full 140 GB, which doesn't fit on one 80 GB GPU.
- **C:** Pipeline stages across nodes add latency and bubbles; it is for models too big for one node.
- **D:** Streaming weights over PCIe every step would make decode far slower.

</details>

## 8. Select all that apply (medium)

Which practices increase prefix-cache hit rates?

- **A.** Put the fixed system prompt and tool definitions at the start of the prompt.
- **B.** Put a per-request timestamp on the first line of the system prompt.
- **C.** Route requests that share a long document or prefix to the same replica.
- **D.** Keep few-shot examples in a fixed order.

<details>
<summary>Answer</summary>

**A, C, D**

Prefix caches match exact token prefixes. Stable content first, consistent ordering, and prefix-aware routing all raise hits. Any change near the start invalidates everything after it.

- **B:** A changing token at the start makes every prompt's prefix unique, so nothing after it can be reused.

</details>

## 9. Arrange in order (easy)

Put a request's life on a continuous-batching server in order.

- Finish on an end token or length limit; free its KV blocks for the next request
- Decode one token per scheduler step alongside other sequences
- Prefill the prompt (possibly in chunks) and emit the first token
- Admitted when a slot and enough KV blocks are free
- Wait in the queue

<details>
<summary>Answer</summary>

1. Wait in the queue
2. Admitted when a slot and enough KV blocks are free
3. Prefill the prompt (possibly in chunks) and emit the first token
4. Decode one token per scheduler step alongside other sequences
5. Finish on an end token or length limit; free its KV blocks for the next request

TTFT covers the first three steps; TPOT is set during the fourth. Freed blocks are reused immediately, which is what keeps memory full of useful work.

</details>

## 10. Multiple choice (medium)

Which signal is the best primary trigger for autoscaling LLM decode replicas?

- **A.** Host CPU utilization
- **B.** GPU utilization percentage
- **C.** Queue depth and KV-cache utilization, checked against measured TTFT and TPOT
- **D.** Number of HTTP connections

<details>
<summary>Answer</summary>

**C.** Queue depth and KV-cache utilization, checked against measured TTFT and TPOT

Queueing and KV pressure show up before users feel the latency SLO breaking. GPU utilization looks high for a decode-bound server at almost any load.

- **A:** The CPU mostly schedules and tokenizes; it says little about GPU capacity.
- **B:** Decode-bound kernels keep the GPU "busy" whether the batch is 2 or 64.
- **D:** Streaming connections can stay open while idle, and connection count ignores prompt and output length.

</details>

## 11. Reflection (hard)

Your company spends a growing amount on a hosted LLM API. A colleague proposes self-hosting an open-weight model on rented GPUs. What numbers and factors would you gather before deciding?

<details>
<summary>Answer</summary>

**Model answer.** Monthly input and output token volume and its peak-to-average ratio; the quality gap between the open model and the API on our own evals; a load-tested throughput per GPU at our TTFT and TPOT SLOs with our real prompt and output lengths; GPU price and the utilization we would realistically hit (cost per million tokens = hourly price divided by millions of tokens per hour at that utilization); engineering and on-call cost; data-residency and privacy requirements; and whether we need fine-tuning. Self-hosting pays when volume is high and steady enough to keep GPUs busy (in the lesson's synthetic example, above about 74% utilization against a USD 1 per million API) or when control or privacy requires it. A hybrid with a router is common.

Utilization and measured throughput at SLO decide the cost comparison; quality and control decide the rest.

</details>
