---
title: Reliability, cost, and observability
summary: Make AI services dependable with timeouts, retries, circuit breakers, and idempotency; cut cost and latency with caching, batching, KV caching, quantization, and distillation; and monitor drift in data, prompts, and quality.
skill: engineering
minutes: 40
prerequisites: [production-architecture]
related: [production-architecture, llm-evaluation, ml-workflow]
---

# Reliability, cost, and observability

> **Mental model.** Every dependency will eventually be slow or down, every model will drift, and every token costs money. Reliability patterns contain failures, efficiency techniques shrink cost and latency, and observability tells you which of these is happening before your users do.

**You will learn to**
- Apply timeouts, bounded retries with backoff, circuit breakers, idempotency, and fallbacks.
- Compute how retries and chained dependencies affect failure rates.
- Reduce cost and latency with caching, batching, KV caching, quantization, distillation, and routing.
- Instrument traces and metrics for ML and LLM systems.
- Detect data drift, prompt drift, training-serving skew, and quality regressions.

**Why it matters.** Users judge an AI feature by whether it works every time and responds quickly. Finance judges it by cost per request. Both degrade silently without the right patterns and dashboards.

## 1. Intuition

**Reliability patterns:**

- **Timeouts:** never wait forever on a model, tool, or index.
- **Retries with exponential backoff and jitter:** retry transient errors a bounded number of times, waiting longer each time, so a struggling service isn't hammered.
- **Circuit breakers:** after repeated failures, stop calling a dependency for a while and use a fallback, instead of piling up slow requests.
- **Idempotency:** make retried operations safe to repeat (an idempotency key ensures a payment is charged once).
- **Fallbacks:** a smaller model, a cached answer, a search-only response, or "please try again" beats a hang.

**Efficiency:**

- **Caching:** exact-match caches for repeated prompts; semantic caches for near-duplicates (with care); provider-side prompt caching for shared long prefixes.
- **Batching:** process many requests together on the GPU for throughput, at some cost to individual latency.
- **KV caching:** reuse attention keys and values of previous tokens during generation.
- **Quantization:** store weights in fewer bits (int8, int4) to cut memory and often raise speed, with possible small quality loss.
- **Distillation:** train a small "student" model to imitate a large "teacher."
- **Routing:** send easy requests to cheaper models.

**Observability:** structured logs, metrics, and end-to-end traces with prompt, model, and index versions, token counts, latency per step, and quality signals.

## 2. Visualization

![Three panels. A right-skewed latency histogram with p50 at about 1.2 seconds, p95 near 2.7, and p99 higher still. A line falling from 12 to 1.2 dollars per thousand requests as cache hit rate rises from 0 to 90 percent. Bars for 7-billion-parameter model weight memory: 28 GB in fp32, 14 in fp16, 7 in int8, 3.5 in int4, with a 24 GB GPU line.](../../figures/reliability-cost-and-observability.png)

*Synthetic latencies and illustrative prices. Averages hide the tail that users feel; caching scales cost down linearly with hit rate; quantization decides which GPUs can hold the model at all.*

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $p$ | probability a single attempt fails (assumed independent) |
| $r$ | number of retries |
| $a_i$ | availability of dependency $i$ |
| $h$ | cache hit rate |
| $b$ | bits per weight |

### Retries and dependency chains

```math
P(\text{request fails}) = p^{\,r + 1} \qquad A_{\text{chain}} = \prod_i a_i
```

Retries help only for *independent, transient* failures. If the dependency is down, every retry fails and just adds load and latency.

### Cost with caching, memory with quantization

```math
\text{cost} = (1 - h)\,c_{\text{miss}} + h\,c_{\text{hit}} \qquad \text{weight memory} = \frac{\text{parameters} \times b}{8} \text{ bytes}
```

### Worked example 1: retries

A model API fails 2% of requests transiently. With no retries, 2% of user requests fail. With one retry: $0.02^2 = 0.0004$ (0.04%). With two: $0.02^3 = 0.000008$. Each retry adds latency only on failure, but during an outage retries triple traffic to an already failing service, which is why you add a circuit breaker.

### Worked example 2: chained availability

A request depends on gateway, retrieval, model, a tool, and a database, each 99% available. End-to-end availability is $0.99^5 = 0.951$: about 1 in 20 requests fail, far worse than any single component. Fallbacks and graceful degradation (answer without the tool, with a notice) raise effective availability.

### Worked example 3: caching

Requests cost 1.2 cents on a cache miss and almost nothing on a hit. With a 40% hit rate, average cost is $0.6 \times 1.2 = 0.72$ cents, a 40% saving, and cached answers return in tens of milliseconds instead of seconds.

### Worked example 4: quantization

A 7-billion-parameter model needs $7 \times 10^9 \times 16 / 8 = 14$ GB in 16-bit precision, $7$ GB in int8, and $3.5$ GB in int4 (weights only; the KV cache and activations need more). In 16-bit it doesn't fit next to a long-context KV cache on a 16 GB GPU; in int4 it fits comfortably.

## 4. Implementation

```python
import random, time

class CircuitBreaker:
    def __init__(self, threshold=5, cooldown=30):
        self.failures, self.threshold, self.cooldown, self.opened_at = 0, threshold, cooldown, None

    def call(self, fn, *args, fallback=None):
        if self.opened_at and time.time() - self.opened_at < self.cooldown:
            return fallback() if fallback else None              # open: skip the dependency
        try:
            result = fn(*args)
            self.failures, self.opened_at = 0, None               # success closes the breaker
            return result
        except TransientError:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.time()
            raise

def with_retries(fn, *args, attempts=3, base=0.5, cap=8.0):
    for k in range(attempts):
        try:
            return fn(*args)
        except TransientError:
            if k == attempts - 1:
                raise
            time.sleep(min(cap, base * 2 ** k) * random.uniform(0.5, 1.5))   # backoff + jitter
```

A trace record per request (store in your logging or tracing system):

```python
trace = {
    "request_id": rid, "user_tier": tier, "route": "small-model",
    "prompt_version": "triage-v7", "model": "model-2026-06", "index_version": "docs-2026-10-01",
    "retrieved_ids": [12, 87, 3], "input_tokens": 1834, "output_tokens": 212,
    "latency_ms": {"retrieval": 85, "model": 1240, "total": 1402}, "cache_hit": False,
    "validation_passed": True, "cost_usd": 0.0089, "user_feedback": None,
}
```

Runnable script (retry, chain, caching, and quantization arithmetic and figures): [`code/18-production-ai/production.py`](../../code/18-production-ai/production.py).

## 5. Engineering

**What to monitor.**

| Classical ML | LLM applications |
|---|---|
| feature drift | prompt and input drift |
| prediction distribution | token and tool-call distribution |
| ground-truth performance | judge, human, and task-success signals |
| training-serving skew | prompt, template, and index version skew |
| data-quality anomalies | retrieval gaps and grounding failures |
| latency and throughput | latency, tokens/sec, cost, cache hit rate |

**Drift.** Data drift is a change in input distributions (new user populations, new product lines); concept drift is a change in the input-to-label relationship; prompt drift is users asking different kinds of questions over time. Compare live distributions with training or baseline windows and alert on large shifts. Ground truth often arrives late, so pair drift metrics with delayed quality checks.

**Training-serving skew** means features or prompts at inference differ from those used in training or evaluation. Share feature code between training and serving, and log versions.

**Latency budget.** Measure time to first token and tokens per second separately. Stream tokens to the user. Batching raises throughput but can raise individual latency; tune batch size against your p95 target.

**Cost levers, in order of effort:** shorter prompts and fewer retrieved chunks; caching; routing to smaller models; output length limits; quantized self-hosted models; distillation for high-volume narrow tasks.

**Privacy in logs.** Redact secrets and personal data before logging prompts and outputs; restrict access; set retention limits.

> [!WARNING]
> **Failure modes.** Retry storms during outages; retries duplicating side effects; caches serving one user's answer to another; dashboards showing averages while the p99 burns; silent quality regressions after a provider's model update; logs full of personal data.

### Common mistakes

- Retrying non-idempotent operations without idempotency keys.
- Monitoring uptime but not answer quality.
- Quantizing without re-running the evaluation set.

## 6. Knowledge check

<!-- quiz:reliability-cost-and-observability -->
**[Take the reliability, cost, and observability quiz](../../quizzes/reliability-cost-and-observability.md)**
<!-- /quiz -->

**Practice exercise.** A pipeline has four dependencies at 99.9%, 99.5%, 99%, and 99.9% availability. What is end-to-end availability, and how many failures per 100,000 requests does that imply?

<details>
<summary>Solution</summary>

$0.999 \times 0.995 \times 0.99 \times 0.999 = 0.9831$. About 1,690 failures per 100,000 requests. The 99% dependency is the first one to fix or wrap with a fallback.
</details>

**Implementation challenge.** Wrap a mock model client that fails randomly 10% of the time and has occasional 10-second hangs with timeouts, retries with jittered backoff, a circuit breaker, and a cached fallback. Simulate 10,000 requests and report the failure rate and p50/p95/p99 latency with and without each pattern.

## Summary

- Timeouts, bounded retries with backoff, circuit breakers, idempotency, and fallbacks contain failures; chained dependencies multiply unavailability.
- Caching, batching, KV caching, quantization, distillation, and routing cut cost and latency.
- Report latency percentiles, cost per request, and cache hit rate, not just averages.
- Monitor data, prompt, and quality drift and version skew; trace every request with its prompt, model, and index versions.

**Next:** [Securing AI systems](../19-safety-security/01-ai-security.md)

**Related:** [Production architecture](01-production-architecture.md) · [LLM evaluation](../17-llm-evaluation/01-llm-evaluation.md) · [The ML workflow](../02-ml-workflow/01-ml-workflow.md)
