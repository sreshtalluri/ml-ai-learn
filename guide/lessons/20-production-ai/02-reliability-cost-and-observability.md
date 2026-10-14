---
title: Reliability, cost, and observability
summary: Make AI services dependable with timeouts, retries, circuit breakers, and idempotency; cut cost and latency with caching, batching, KV caching, quantization, and distillation; and monitor drift in data, prompts, and quality.
skill: engineering
minutes: 40
prerequisites: [production-architecture]
related: [production-architecture, llm-evaluation, ml-workflow]
adapted_from: [phases/11-llm-engineering/11-caching-cost, phases/11-llm-engineering/15-prompt-caching, phases/13-tools-and-protocols/20-opentelemetry-genai, phases/17-infrastructure-and-production/14-prompt-semantic-caching, phases/17-infrastructure-and-production/16-model-routing]
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

### Build it from scratch

The full script is [`code/20-production-ai/cache_router_scratch.py`](../../code/20-production-ai/cache_router_scratch.py). It builds the four cost levers and the trace record that tells you whether they work. Prices are synthetic.

**Exact cache keys and a toy similarity.** An exact cache is only correct if the key includes everything that changes the answer: model, prompt version, temperature, and the normalised prompt. The semantic cache needs a similarity; bag-of-words cosine stands in for an embedding model.

```python
# --- 1. Exact cache: key on everything that changes the answer -----------------------------
def cache_key(model, prompt_version, prompt, temperature):
    raw = f"{model}|{prompt_version}|{temperature}|{prompt.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

# --- 2. Semantic cache: bag-of-words cosine against stored questions ---------------------------
def bow(text):
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))

def cosine(a, b):
    dot = sum(a[w] * b[w] for w in a)
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0
```

*Predict:* you ship prompt version `faq-v4` but forget to add `prompt_version` to the key. What do users see?

**Semantic cache.** Return a stored answer when a new question is "close enough" to an old one. The threshold is the whole design decision.

```python
class SemanticCache:
    def __init__(self, threshold):
        self.threshold, self.items = threshold, []      # ponytail: linear scan, use a vector index past ~10k entries

    def get(self, q):
        best = max(self.items, key=lambda it: cosine(bow(q), it[0]), default=None)
        if best and cosine(bow(q), best[0]) >= self.threshold:
            return best[1]
        return None

    def put(self, q, answer):
        self.items.append((bow(q), answer))
```

*Predict:* "how can i reset my password" and "how do i reset my username" both score 0.83 against the stored "how do i reset my password". Which threshold serves the first without wrongly serving the second?

**Prompt-cache cost.** Providers cache a long, stable prefix (system prompt, tool definitions, documents). Writing the cache costs a little more than normal input; reading it costs much less. Put stable content first and variable content last, or the prefix never matches.

```python
# --- 3. Prompt (prefix) caching cost -------------------------------------------------------
PRICE_IN, PRICE_OUT = 3.00, 15.00             # SYNTHETIC USD per 1M tokens
WRITE_MULT, READ_MULT = 1.25, 0.10            # cache write premium, cache read discount

def request_cost(prefix_tok, suffix_tok, out_tok, cache_state):
    """cache_state: 'none' (no caching), 'write' (first call stores prefix), 'read' (prefix hit)."""
    mult = {"none": 1.0, "write": WRITE_MULT, "read": READ_MULT}[cache_state]
    return (prefix_tok * mult * PRICE_IN + suffix_tok * PRICE_IN + out_tok * PRICE_OUT) / 1e6
```

*Predict:* the first request writes the cache. Does a workload of exactly one request get cheaper or more expensive?

**Router.** Send easy requests to a small model and hard ones to a large one. Real routers use a trained classifier or the small model's own confidence; this rule-based score shows the shape.

```python
# --- 4. Router: cheap model unless the request looks hard ----------------------------------
HARD_WORDS = {"prove", "derive", "refactor", "debug", "legal", "contract", "multi-step", "why"}

def route(prompt):
    words = set(re.findall(r"[a-z\-]+", prompt.lower()))
    score = len(words & HARD_WORDS) + (1 if len(prompt) > 400 else 0)
    return ("large-model" if score >= 1 else "small-model"), score
```

*Predict:* what does this router do with "Why are your opening hours different on Sunday?"

**Trace span.** One span per model call, with the OpenTelemetry GenAI semantic-convention attribute names so any backend can chart tokens and cost per model. App-specific fields go in your own namespace.

```python
# --- 5. One trace span per LLM call, GenAI semantic-convention attribute names -------------
def chat_span(model, provider, in_tok, out_tok, latency_ms, cache_read_tok=0):
    return {
        "name": f"chat {model}",
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": in_tok,
            "gen_ai.usage.output_tokens": out_tok,
            "app.cache_read_tokens": cache_read_tok,      # app-specific attrs: own namespace, not gen_ai.*
            "app.latency_ms": latency_ms,
        },
    }
```

### Use the library

Retries, breakers, and the trace record are small enough to own. Keep them in one shared module so every call site gets the same behaviour:

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

**Provider prompt caching.** With the Anthropic SDK you mark the end of the stable prefix; usage reports cache writes and reads separately. OpenAI caches long prompt prefixes automatically and reports `usage.prompt_tokens_details.cached_tokens`.

```python
resp = client.messages.create(
    model=MODEL, max_tokens=300,
    system=[{"type": "text", "text": POLICY_DOC, "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": question}],
)
print(resp.usage.cache_creation_input_tokens, resp.usage.cache_read_input_tokens)
```

**Tracing.** The OpenTelemetry SDK records the span; instrumentation packages for the major provider SDKs can set the `gen_ai.*` attributes for you. The GenAI conventions are still marked experimental and names have moved (`gen_ai.system` became `gen_ai.provider.name`), so pin the semconv version you emit.

```python
from opentelemetry import trace

tracer = trace.get_tracer("support-bot")
with tracer.start_as_current_span(f"chat {MODEL}") as span:
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.request.model", MODEL)
    resp = call_model(prompt)
    span.set_attribute("gen_ai.usage.input_tokens", resp.usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", resp.usage.output_tokens)
```

### Run it

From `guide/`:

```bash
uv run code/20-production-ai/cache_router_scratch.py
```

Key output:

```text
Semantic cache (stored: 'how do i reset my password'):
  threshold 0.8: [('how can i reset my password', 0.83, True), ('how do i reset my username', 0.83, True), ...]
  threshold 0.9: [('how can i reset my password', 0.83, False), ('how do i reset my username', 0.83, False), ...]
Prompt caching, 4000-token shared prefix, 200 new tokens in, 300 out, 100 calls:
  no cache USD 1.7100   cached USD 0.6438   saving 62%
Router:
  small-model  score=0  What are your opening hours?
  large-model  score=2  Why does this recursive function overflow? Debug it.
all checks passed
```

No threshold separates the paraphrase from the different question: word overlap is identical. That is why semantic caches need a real embedding model, a high threshold, and an offline check of false-hit rate on labelled pairs. Prompt caching cuts this workload's bill by 62% because the 4,000-token prefix is read at 0.1x on 99 of 100 calls: one call costs USD 0.0171 uncached, USD 0.0201 when writing, and USD 0.0063 when reading.

Retry, chain, caching, and quantization arithmetic and figures: [`code/20-production-ai/production.py`](../../code/20-production-ai/production.py).

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

**Next:** [Securing AI systems](../21-safety-security/01-ai-security.md)

**Related:** [Production architecture](01-production-architecture.md) · [LLM evaluation](../17-llm-evaluation/01-llm-evaluation.md) · [The ML workflow](../02-ml-workflow/01-ml-workflow.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Batch APIs — the 50% Discount as Industry Standard](https://aiengineeringfromscratch.com/lesson?path=phases/17-infrastructure-and-production/15-batch-apis) · [FinOps for LLMs — Unit Economics and Multi-Tenant Attribution](https://aiengineeringfromscratch.com/lesson?path=phases/17-infrastructure-and-production/27-finops-llms)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Caching, Rate Limiting & Cost Optimization](https://aiengineeringfromscratch.com/lesson?path=phases/11-llm-engineering/11-caching-cost), [Prompt Caching and Context Caching](https://aiengineeringfromscratch.com/lesson?path=phases/11-llm-engineering/15-prompt-caching), [OpenTelemetry GenAI — Tracing Tool Calls End-to-End](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/20-opentelemetry-genai), [Prompt Caching and Semantic Caching Economics](https://aiengineeringfromscratch.com/lesson?path=phases/17-infrastructure-and-production/14-prompt-semantic-caching), [Model Routing as a Cost-Reduction Primitive](https://aiengineeringfromscratch.com/lesson?path=phases/17-infrastructure-and-production/16-model-routing).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Why do you need both retries and a circuit breaker when calling a model API?</strong></summary>

Retries fix independent transient failures; circuit breakers stop retries from making a real outage worse. If each attempt fails independently with probability $p$, $r$ retries make a request fail with probability $p^{r+1}$: with a 2% transient error rate, one retry gives $0.02^2 = 0.04\%$ and two give $0.0008\%$. Retries cost latency only on failure. But during an outage, failures aren't independent: every retry also fails, and three attempts per request triple the load on an already failing service, a retry storm that slows recovery and exhausts your own connection pools. A circuit breaker counts recent failures and, past a threshold, stops calling the dependency for a cooldown period and returns a fallback immediately (a secondary model, cached answer, or degraded mode), then lets a trial request through. Retries also need exponential backoff with jitter so clients don't retry in sync, and idempotency keys for side-effecting calls.

</details>

<details>
<summary><strong>Will a 13-billion-parameter model fit on a 24 GB GPU for serving?</strong></summary>

Not in 16-bit precision, comfortably in int4, and only at some cost in int8. Weight memory is parameters times bytes per weight: $13 \times 10^9 \times 2 = 26$ GB in fp16 or bf16, already over 24 GB before anything else. Int8 needs 13 GB, leaving about 11 GB for the KV cache, activations, and framework overhead. Int4 needs about 6.5 GB, leaving around 17 GB. That leftover matters because the KV cache is what lets you serve long contexts and many concurrent sequences: a model this size can easily need close to 1 MB of cache per token in 16-bit, so 11 GB covers roughly 10 to 15 thousand tokens across all active requests. Quantization usually costs a small amount of quality, more at int4 and on some tasks. Re-run the evaluation set on the quantized model before deploying it. FP8 is another option on hardware that supports it.

</details>

<details>
<summary><strong>Answer quality dropped noticeably last week, but there are no errors and latency looks normal. How do you investigate?</strong></summary>

Look for what changed, using the versions you record in every trace. First, compare quality metrics (judge scores, thumbs-down rate, escalations, validation failures) by prompt version, model version, and index version, and find the date the drop started. Common causes: the provider updated the model behind an unpinned alias; someone shipped a prompt-template change without evaluation; the index was rebuilt with a different embedding model, or a re-index job failed and content went stale; or a new feature or user segment changed what people ask (prompt drift). Second, re-run the offline evaluation set against current production. If it still passes, the inputs changed rather than the system, so sample recent traffic, label it, and add it to the set. If it fails, diff the versions and bisect. Long-term fixes: pin model versions, gate every change on the evaluation set, and alert on quality signals, not just uptime.

</details>

<details>
<summary><strong>Your LLM feature's serving cost has to drop by half. What levers do you pull?</strong></summary>

Start with the levers that don't change the model, measure each on the evaluation set, and stack them. Caching: exact-match caching for repeated questions, and prefix caching for the shared system prompt and documents, which avoids recomputing those tokens. A 40% hit rate on 1.2-cent requests cuts the average to $0.6 \times 1.2 = 0.72$ cents. Prompt diet: trim boilerplate instructions and retrieve fewer, better chunks, since input tokens are often most of the bill. Routing: send easy requests to a smaller model. Output limits: shorter answers cut the more expensive output tokens and latency. If you self-host, continuous batching and paged attention raise GPU throughput substantially, int8 or int4 quantization fits more batch per GPU, and speculative decoding cuts latency. Distilling a narrow task into a small fine-tuned model is the bigger project with the biggest savings. Watch p95 latency and quality per segment, not just averages.

</details>
