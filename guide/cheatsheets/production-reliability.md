---
title: Production reliability
summary: Patterns, formulas, and monitoring for dependable, affordable AI services.
---

# Production reliability

| Pattern | Use it to |
|---|---|
| Timeouts | never wait forever on a model, tool, or index |
| Bounded retries with exponential backoff + jitter | recover from transient failures without retry storms |
| Circuit breaker | stop calling a failing dependency; use a fallback |
| Idempotency keys | make retried side effects happen once |
| Fallbacks | smaller model, cached answer, degraded feature |
| Structured outputs + schema validation | never act on malformed model output |
| Rate limits and quotas | protect capacity and budget |
| Model / prompt / index versioning | reproduce and roll back |

| Formula | Meaning |
|---|---|
| $P(\text{fail}) = p^{r+1}$ | independent failures with $r$ retries |
| $A = \prod_i a_i$ | availability of dependencies in series |
| $L = \lambda W$ | Little's law: requests in flight |
| $\text{cost} = n_{\text{in}}c_{\text{in}} + n_{\text{out}}c_{\text{out}}$ | per-request token cost |
| $(1 - h)\,c_{\text{miss}}$ | cost with cache hit rate $h$ (hits ≈ free) |
| params × bits / 8 | weight memory (7B: 14 GB fp16, 3.5 GB int4) |

**Cost and latency levers:** shorter prompts, fewer chunks, caching, routing to small models, output limits, streaming, batching, KV caching, quantization, distillation.

**Monitor:** latency percentiles (not averages), errors, cost, cache hit rate, input/prompt drift, retrieval gaps, quality signals, version skew.

Lessons: [Production architecture](../lessons/20-production-ai/01-production-architecture.md) · [Reliability, cost, and observability](../lessons/20-production-ai/02-reliability-cost-and-observability.md)
