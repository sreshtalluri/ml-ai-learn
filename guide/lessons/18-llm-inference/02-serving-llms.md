---
title: Serving LLMs
summary: Design an LLM serving stack (continuous batching, paged KV cache, prefix caching, chunked prefill, multi-GPU parallelism), set latency SLOs, and turn GPU hours into cost per million tokens.
skill: llms
minutes: 45
prerequisites: [llm-inference]
related: [production-architecture, reliability-cost-and-observability, rag-pipeline]
---

# Serving LLMs

> **Mental model.** An LLM server is a restaurant with a fixed number of tables (batch slots) and a limited amount of counter space (KV-cache memory). Static batching seats a whole party and clears no table until the slowest diner finishes. Continuous batching clears each table the moment its diner leaves and seats the next person in line. Paged memory is counter space handed out in small trays instead of reserving a whole banquet table for every guest.

**You will learn to**
- Simulate static and continuous batching by hand and compute throughput, latency, and idle capacity.
- Explain paged KV-cache memory, prefix caching, and chunked prefill, and what each one fixes.
- Choose between tensor, pipeline, data, and expert parallelism for a model that doesn't fit or doesn't keep up.
- Set and monitor SLOs for TTFT and TPOT at p50 and p95, and reason about the throughput-latency trade-off.
- Convert GPU price and throughput into cost per million tokens and compare self-hosting with an API.

**Why it matters.** The same model on the same GPU can serve several times more users depending on the scheduler. Most of the jump in open-source serving throughput over the last few years came from scheduling and memory management, not from faster chips. If you run models yourself, these decisions set your cost per token; if you buy an API, they explain its pricing and its latency under load.

## 1. Intuition

A request's life on a server: it waits in a **queue**, gets admitted, runs **prefill** on its prompt (producing the first token), then joins the running batch for **decode** steps until it emits an end token or hits its length limit. The previous lesson showed that one decode step costs about the same whether the batch holds 1 sequence or 32, so the goal is to keep the batch full of useful work.

**Static batching** collects up to $K$ requests, runs them together, and admits nobody new until all of them finish. Output lengths vary a lot (a "yes" next to a 900-token essay), so most slots go idle while the batch waits for its longest member, and new arrivals wait too.

**[Continuous batching](../../glossary.md#continuous-batching)** (also called in-flight or iteration-level batching) makes the admission decision every decode step. When a sequence finishes, its slot is refilled on the next step. Slots go idle only when nobody is waiting.

**[Paged attention](../../glossary.md#paged-attention)** stores each sequence's KV cache in fixed-size blocks (for example 16 tokens) allocated on demand, with a per-sequence block table mapping logical positions to physical blocks, like virtual memory pages. Without it, servers reserved a contiguous region for each request's *maximum* length, and most of that memory sat empty. Paging also lets sequences share blocks.

**[Prefix caching](../../glossary.md#prefix-caching)** reuses those shared blocks: if many requests start with the same system prompt, few-shot examples, or document, the KV for that prefix is computed once and reused. Hits skip prefill for the shared part. API providers expose the same idea as discounted "cached input tokens".

**Chunked prefill** splits a long prompt's prefill into chunks (say 512 tokens) and interleaves them with other sequences' decode steps. Without it, one 30k-token prompt stalls every user's next token while it prefills. **Prefill-decode disaggregation** goes further: separate GPU pools run prefill and decode, and the KV cache is transferred between them, so each pool can be sized and tuned for its own bottleneck.

**When one GPU isn't enough:**

| Parallelism | What is split | Communication | Use when |
|---|---|---|---|
| Data parallel (replicas) | requests; each GPU group has a full copy | none between replicas | the model fits; you need more throughput |
| Tensor parallel (TP) | each weight matrix, across GPUs | all-reduce inside every layer | the model doesn't fit on one GPU, or you need lower per-token latency; needs fast links (within a node) |
| Pipeline parallel (PP) | layers, in stages across GPUs | activations between stages | model spans nodes; adds bubbles and latency |
| Expert parallel (EP) | MoE experts, across GPUs | all-to-all token routing | large MoE models |

Tensor parallelism also adds bandwidth: two GPUs read half the weights each, so per-token latency can drop, minus communication time.

**Under load.** When KV memory runs out, the scheduler must stop admitting, or **preempt** a running sequence: drop its cache and recompute it later, or swap it to CPU memory over PCIe. Long queues show up first as rising TTFT; oversized batches show up as rising TPOT.

## 2. Visualization

<!-- lab:batching -->
![Two slot timelines for the same 14 synthetic requests on 4 slots. Static batching leaves large gaps and finishes at step 225 with 1.63 tokens per step, p95 latency 139 steps, 59 percent of slot time idle. Continuous batching packs requests back to back and finishes at step 168 with 2.18 tokens per step, p95 latency 82 steps, 45 percent idle. A third panel shows p95 latency against arrival rate: static rises from about 110 to 1,000 steps as load grows, continuous stays near 60 to 95 steps up to the capacity line at 0.167 requests per step.](../../figures/serving-llms.png)

*Synthetic workload: Poisson-like arrivals, log-normal output lengths with mean 24 tokens, 4 slots, one decode step per time unit. Prefill is not modelled. Idle time in the timelines includes moments when no request was waiting.*

*Interactive version: change arrival rate, slots, and length spread, and switch policies on the same workload. [Open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/batching/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Load the **Lesson example**. Before switching policies, predict the finish step of each request under static batching. Then check your answers against the worked example below.
2. Choose **Same lengths** (σ = 0). Predict whether continuous batching still wins, and by how much compared with σ = 0.8.
3. Choose **Near capacity** and then raise the arrival rate past the capacity hint. Predict what happens to p95 latency under each policy.
4. Hold the arrival rate fixed and double the slots from 4 to 8. Which metric improves most: throughput, mean latency, or idle %?

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $K$ | batch slots (maximum concurrent sequences) |
| $n_i$ | output tokens of request $i$ |
| $a_i, f_i$ | arrival step and finish step of request $i$ |
| $\ell_i = f_i - a_i$ | latency of request $i$, in steps |
| $M$ | makespan, the step when the last request finishes |
| $\lambda$ | arrival rate, requests per step (or per second) |
| $\bar{n}$ | mean output length |

The simulator's rules: one time step is one decode iteration, every occupied slot emits one token per step, and a request admitted at step $s$ finishes at $f = s + n$.

```math
\text{throughput} = \frac{\sum_i n_i}{M}, \qquad
\text{idle} = 1 - \frac{\sum_i n_i}{K \cdot M}, \qquad
p95 = \ell_{(\lceil 0.95 N \rceil)}
```

where $\ell_{(j)}$ is the $j$-th smallest latency (nearest-rank percentile).

### Worked example: static versus continuous

$K = 2$ slots; four requests all arrive at step 0 with $n = [2, 8, 3, 3]$. Total tokens $2 + 8 + 3 + 3 = 16$.

**Static.** Batch 1 is requests 0 and 1, starting at 0. Request 0 finishes at $0 + 2 = 2$; request 1 at $0 + 8 = 8$. Slot 0 sits idle from 2 to 8 because the batch isn't done. Batch 2 starts at 8: request 2 finishes at $8 + 3 = 11$, request 3 at $8 + 3 = 11$.

- Finish steps $[2, 8, 11, 11]$; latencies equal finish steps since all arrived at 0.
- $M = 11$. Throughput $= 16 / 11 = 1.4545$ tokens/step.
- Slot capacity $K \cdot M = 2 \cdot 11 = 22$ slot-steps; idle $= 1 - 16/22 = 6/22 = 27.3\%$.
- Mean latency $= (2 + 8 + 11 + 11)/4 = 32/4 = 8.0$ steps. p95: $\lceil 0.95 \cdot 4 \rceil = 4$, the 4th smallest, $= 11$.

**Continuous.** Requests 0 and 1 start at 0. Request 0 finishes at 2; request 2 takes slot 0 at step 2 and finishes at $2 + 3 = 5$; request 3 takes slot 0 at 5 and finishes at $5 + 3 = 8$. Request 1 finishes at 8 in slot 1.

- Finish steps $[2, 8, 5, 8]$. $M = 8$. Throughput $= 16/8 = 2.0$ tokens/step, a $2.0 / 1.4545 = 1.375\times$ gain.
- Idle $= 1 - 16/(2 \cdot 8) = 1 - 16/16 = 0\%$.
- Mean latency $= (2 + 8 + 5 + 8)/4 = 23/4 = 5.75$ steps. p95 $= 8$.

Same hardware, same requests: 37.5% more throughput and 28% lower mean latency, just from the admission rule. The gain grows with the spread of output lengths.

### Capacity and Little's law

A server with $K$ slots and mean output length $\bar{n}$ can finish at most $K / \bar{n}$ requests per step. In the figure, $4 / 24 = 0.167$. As $\lambda$ approaches capacity, queues grow without bound and latency explodes; static batching wastes slots, so its effective capacity is lower and it blows up earlier.

**Little's law** links the three quantities in any stable queue: average requests in the system $=$ arrival rate $\times$ average time in the system. With 2 requests/s and the previous lesson's 7.675 s end-to-end latency, the server holds $2 \times 7.675 = 15.35$ requests on average, so it needs a batch of about 16 and the KV memory for 16 sequences.

### Paged KV memory

KV budget 40 GB, 131,072 bytes per token (the 8B-style GQA model in BF16), requests average 600 tokens of context, configured maximum 4,096.

- **Contiguous reservation:** each sequence reserves $4096 \cdot 131{,}072 = 536{,}870{,}912$ bytes. Sequences that fit: $\lfloor 40 \times 10^9 / 536{,}870{,}912 \rfloor = \lfloor 74.5 \rfloor = 74$. Useful fraction: $600 / 4096 = 14.6\%$.
- **Paged, 16-token blocks:** $\lceil 600/16 \rceil = 38$ blocks $= 608$ tokens $= 79{,}691{,}776$ bytes. Sequences that fit: $\lfloor 40 \times 10^9 / 79{,}691{,}776 \rfloor = \lfloor 501.9 \rfloor = 501$. Useful fraction: $600/608 = 98.7\%$.

About $6.8\times$ more concurrent sequences from the same memory. Waste is at most one partly filled block per sequence. (Real sequences grow while decoding, so the scheduler must also handle running out of blocks mid-generation, by preempting.)

### Prefix caching

A 2,000-token shared system prompt plus 200 user tokens: prefill drops from 2,200 to 200 tokens on a cache hit, $1 - 200/2200 = 90.9\%$ less prefill compute, and TTFT falls by a similar fraction. The 2,000 tokens of KV are stored once and shared by every request that hits.

### Cost per million tokens

Synthetic numbers: one GPU at \$4.00 per hour sustaining 1,500 output tokens/s at your latency SLO.

- Tokens per hour: $1500 \times 3600 = 5{,}400{,}000$.
- Cost per million at 100% busy: $4.00 / 5.4 = 0.741$, so \$0.741.
- Real traffic has peaks and valleys. At 60% average utilization: $0.741 / 0.6 = 1.23$, so \$1.23; at 30%: $0.741 / 0.3 = 2.47$, so \$2.47.
- One GPU for a month (730 hours) costs $4.00 \times 730 = 2{,}920$, so \$2,920, and can produce at most $5.4\text{M} \times 730 = 3.942$ billion tokens. Against a synthetic API price of \$1.00 per million output tokens, break-even is $2{,}920$ million tokens per month, which is $2{,}920 / 3{,}942 = 74\%$ utilization, before engineering and on-call time.

Utilization, not the hourly price, usually decides whether self-hosting pays.

## 4. Implementation

The core of the simulator (the same rules as the lab):

```python
def simulate(requests, slots, policy):            # requests: list of (arrival, n_tokens)
    queue, nxt, t = [], 0, 0
    busy_until, finish = [0] * slots, [None] * len(requests)
    order = sorted(range(len(requests)), key=lambda i: requests[i][0])
    while None in finish:
        while nxt < len(order) and requests[order[nxt]][0] <= t:
            queue.append(order[nxt]); nxt += 1
        free = [s for s in range(slots) if busy_until[s] <= t]
        if policy == "continuous" or len(free) == slots:   # static waits for the whole batch
            for s in free[:len(queue)]:
                i = queue.pop(0)
                busy_until[s] = finish[i] = t + requests[i][1]
        t += 1
    return finish
```

Measuring a real server: most open-source serving engines expose an OpenAI-compatible HTTP API, so you can measure TTFT and TPOT with any streaming client.

```python
import time, httpx, json

def measure(url, model, prompt, max_tokens=256):
    t0 = time.perf_counter(); times = []
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "stream": True}
    with httpx.stream("POST", f"{url}/v1/completions", json=body, timeout=120) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                times.append(time.perf_counter())            # one chunk is usually one token
    ttft = times[0] - t0
    tpot = (times[-1] - times[0]) / max(1, len(times) - 1)
    return ttft, tpot
```

Run it from many concurrent clients at increasing request rates, and plot p50 and p95 TTFT and TPOT against throughput. That curve, not a single benchmark number, tells you how much traffic a replica can take within your SLO.

Runnable script (worked example, paged memory, prefix caching, cost, load sweep, and the figure): [`code/18-llm-inference/batching_sim.py`](../../code/18-llm-inference/batching_sim.py).

## 5. Engineering

**SLOs.** Write them per use case, as percentiles: for a chat product, for example, p95 TTFT under 1 s and p95 TPOT under 50 ms (20 tokens/s, faster than people read). Batch jobs (offline summarization, evals, synthetic data) have no TTFT SLO at all, so run them at the largest batch that fits and on spare capacity.

**The throughput-latency knob.** Bigger batches raise tokens/s per GPU and cost less per token, but every step reads more KV and does more work, so TPOT rises. Most engines expose it as a maximum number of concurrent sequences and a maximum number of tokens per scheduler step. Tune both against the SLO curve from your load test.

**Admission and fairness.** Reject or queue early with a clear error rather than accepting work you cannot finish within SLO. Cap per-tenant concurrency so one customer's batch job can't starve interactive traffic. Route very long prompts to a separate pool.

**Autoscaling signals.** CPU and even GPU utilization are poor signals: a decode-bound GPU looks "busy" at any batch size. Scale on queue depth, KV-cache utilization, and measured TTFT and TPOT. Model weights take minutes to load, so keep warm capacity for peaks.

**Prefix caching in practice.** Put stable content (system prompt, tool definitions, few-shot examples, long documents) at the *start* of the prompt and the variable part at the end. A timestamp in the first line kills every cache hit. Route requests that share a prefix to the same replica.

**Self-host or API.** Self-hosting wins on steady high volume, data-residency or privacy requirements, custom fine-tuned models, and control over latency. An API wins for spiky or low volume, the newest frontier models, and teams without inference engineers. Many teams use both, with a router ([production architecture](../20-production-ai/01-production-architecture.md)).

> [!WARNING]
> **Failure modes.** Preemption storms when long requests exhaust KV blocks (throughput collapses while GPUs look busy); a long-prompt request stalling every user's decode (fix with chunked prefill); prefix caches that never hit because of a per-request token at the start of the prompt; cross-tenant leakage if cached prefixes aren't isolated where required; tensor parallelism across slow links where communication eats the gain; load tests with fixed prompt and output lengths that hide the real length distribution.

### Common mistakes

- Reporting average latency. Users feel p95 and p99, and queueing makes the tail grow much faster than the mean.
- Benchmarking with one prompt length and one output length. Batching gains depend on the spread.
- Sizing GPUs from request counts instead of prompt and output tokens.
- Pricing self-hosting at 100% utilization.
- Treating TTFT and TPOT as one latency. Queueing and prefill set TTFT; batch size and KV reads set TPOT.

## 6. Knowledge check

<!-- quiz:serving-llms -->
**[Take the serving LLMs quiz](../../quizzes/serving-llms.md)**
<!-- /quiz -->

**Practice exercise.** Two slots, four requests all arriving at step 0 with output lengths $[4, 1, 1, 4]$. Compute the finish steps, throughput, idle %, and mean latency under static and continuous batching.

<details>
<summary>Solution</summary>

Total tokens 10.

**Static:** batch 1 = requests 0 and 1. Request 1 finishes at 1, request 0 at 4. Batch 2 starts at 4: request 2 finishes at 5, request 3 at 8. Finish $[4, 1, 5, 8]$, $M = 8$, throughput $10/8 = 1.25$ tokens/step, idle $1 - 10/16 = 37.5\%$, mean latency $(4 + 1 + 5 + 8)/4 = 4.5$.

**Continuous:** request 1 finishes at 1; request 2 takes its slot at 1 and finishes at 2; request 3 takes it at 2 and finishes at 6; request 0 finishes at 4. Finish $[4, 1, 2, 6]$, $M = 6$, throughput $10/6 = 1.667$, idle $1 - 10/12 = 16.7\%$ (slot 0 is empty from 4 to 6 because nobody is waiting), mean latency $(4 + 1 + 2 + 6)/4 = 3.25$.
</details>

**Implementation challenge.** Extend the simulator with prefill. Give each request a prompt length $p_i$; prefill takes $\lceil p_i / C \rceil$ steps in its slot (chunk size $C$ tokens per step) and emits the first token at the end. Report mean and p95 TTFT as well as end-to-end latency. Then compare $C = 256$ with unchunked prefill, modelled as every other slot's decode pausing for $\lceil p_i / 256 \rceil$ steps while the whole prompt is processed.

<details>
<summary>Solution sketch</summary>

Track two phases per slot: `prefill_left` and `decode_left`. Each step, a slot in prefill decrements `prefill_left` (and records TTFT when it reaches 0); a slot in decode emits a token. For the unchunked case, a step that contains a new prefill advances the clock by $\lceil p_i/256 \rceil$ steps for *every* slot, so decoding requests see a TPOT spike. You should find that chunking barely changes the long request's own TTFT but sharply cuts p95 TPOT for everyone else, which is why chunked prefill is on by default in most engines.
</details>

## Summary

- Continuous batching refills slots every step, so idle capacity falls and throughput and latency both improve; the gain grows with output-length spread.
- Paged KV memory allocates small blocks on demand, packing several times more sequences into the same memory, and makes prefix sharing possible.
- Prefix caching skips prefill for shared prompt prefixes; chunked prefill and disaggregation stop long prompts from stalling decode.
- Use data parallel replicas for throughput, tensor parallel when the model doesn't fit or latency is too high, pipeline and expert parallel for the largest models.
- Set SLOs on p95 TTFT and TPOT, load-test with realistic length distributions, and price capacity at real utilization.

**Next:** [Tool use and agents](../19-agents/01-tool-use-and-agents.md)

**Related:** [LLM inference](01-llm-inference.md) · [Reliability, cost, and observability](../20-production-ai/02-reliability-cost-and-observability.md) · [The RAG pipeline](../16-rag/01-rag-pipeline.md) · [Cheat sheet: LLM inference](../../cheatsheets/llm-inference.md)

## Interview angle

<details>
<summary><strong>Explain continuous batching and paged attention. Why did they raise serving throughput so much?</strong></summary>

They fix two different kinds of waste. Static batching admits a fixed group and holds every slot until the longest request finishes. Output lengths vary by orders of magnitude, so most slots sit idle and new requests wait. Continuous batching decides admission at every decode step, so a finished sequence's slot is refilled on the next step. Paged attention fixes memory waste. Older servers reserved a contiguous KV region for each request's maximum length, and most of it stayed empty. Paging allocates fixed-size blocks, around 16 tokens each, as the sequence grows, and tracks them in a block table, so waste is under one block per sequence and blocks can be shared across requests with a common prefix. Decode is memory-bound, so throughput scales almost linearly with concurrent sequences, and fitting several times more sequences gives several times the throughput.

</details>

<details>
<summary><strong>When would you self-host an open-weight model instead of calling an API?</strong></summary>

I'd self-host when volume is high and steady, when data can't leave our environment, when I need a fine-tuned or custom model, or when I need latency guarantees the provider won't give. Otherwise I'd use the API. The deciding number is utilization. Take a GPU at \$4 an hour that sustains 1,500 tokens/s: fully busy, that's about \$0.74 per million tokens. At a realistic 30% average utilization it's about \$2.50. Running 24/7 it costs about \$2,900 a month whether we use it or not, so against a \$1-per-million API it only breaks even above about 74% utilization. That's before engineers, on-call, upgrades, and evals. Spiky or low traffic and frontier-quality needs favor the API. A common middle ground is to route bulk, predictable traffic to self-hosted models and keep the API for the hard requests.

</details>

<details>
<summary><strong>During a traffic peak, p95 TTFT jumped from 0.8 s to 6 s, but TPOT stayed flat and GPU utilization looks normal. What do you check?</strong></summary>

Flat TPOT with exploding TTFT means requests are waiting before prefill, not decoding slowly. So I'd look at the queue first: queue depth and wait time, the arrival rate against measured capacity, and whether the scheduler is refusing admission because KV blocks are full. KV-cache utilization near 100%, or rising preemption and swap counts, means memory, not compute, is limiting concurrency. Next I'd check the prompt-length distribution, because a burst of very long prompts (one tenant uploading big documents, say) can monopolize prefill, especially without chunked prefill. I'd also check the prefix-cache hit rate, since a prompt template change that moved variable text to the front kills it, and per-tenant concurrency. GPU utilization isn't a useful signal here because a decode-bound GPU looks busy at any load. Fixes: autoscale on queue depth, isolate long prompts, enforce tenant limits.

</details>

<details>
<summary><strong>Size a deployment: 50 requests/s, 1,000-token prompts, 250-token outputs, p95 TPOT under 50 ms, 8B-style model.</strong></summary>

Output demand is $50 \times 250 = 12{,}500$ tokens/s. Suppose a load test shows one replica sustains 1,500 tokens/s while staying under 50 ms p95 TPOT (that number has to be measured, not assumed). Then I need $\lceil 12{,}500/1{,}500 \rceil = 9$ replicas, and about 12 with headroom for peaks and failures. Check memory with Little's law. End-to-end latency is roughly 0.3 s TTFT + 249 × 0.04 s ≈ 10.3 s, so about $50 \times 10.3 \approx 513$ requests are in flight, or 57 per replica. At 1,250 tokens each and 128 KiB per token, that's about 9.3 GB of KV per replica, which is comfortable. Prefill demand is 50,000 prompt tokens/s, roughly $8 \times 10^{14}$ FLOP/s in total, spread over nine GPUs. I'd add prefix caching if prompts share a system prompt, and chunked prefill to protect TPOT.

</details>
