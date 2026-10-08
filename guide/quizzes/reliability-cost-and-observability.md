<!-- GENERATED from reliability-cost-and-observability.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Reliability, cost, and observability

Covers the lesson [Reliability, cost, and observability](../lessons/18-production-ai/02-reliability-cost-and-observability.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/reliability-cost-and-observability/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

Independent transient failures occur 5% of the time. With two retries (three attempts total), what fraction of requests fail?

<details>
<summary>Answer</summary>

**0.000125** (within ±0.000001)

$0.05^3 = 0.000125$.

</details>

## 2. Calculation (medium)

Three dependencies in series each have 98% availability. What is end-to-end availability? (4 decimals)

<details>
<summary>Answer</summary>

**0.9412** (within ±0.0002)

$0.98^3 = 0.941192$.

</details>

## 3. Calculation (medium)

How many GB do the weights of a 13-billion-parameter model need in int8? (1 GB = $10^9$ bytes)

<details>
<summary>Answer</summary>

**13** (within ±0.01)

8 bits = 1 byte per parameter, so 13 GB. In fp16 it would be 26 GB.

</details>

## 4. Calculation (easy)

Misses cost 2 cents, hits cost nothing, and the hit rate is 25%. What is the average cost per request in cents?

<details>
<summary>Answer</summary>

**1.5** (within ±0.0001)

$0.75 \times 2 = 1.5$.

</details>

## 5. Match (medium)

Match each pattern to the problem it solves.

| Concept | Options |
|---|---|
| Circuit breaker | Keeps the feature usable when a dependency is down |
| Idempotency key | Spreads retries out so they don't synchronize |
| Exponential backoff with jitter | Makes retried side effects happen only once |
| Fallback response | Stops hammering a dependency that keeps failing |

<details>
<summary>Answer</summary>

- Circuit breaker → Stops hammering a dependency that keeps failing
- Idempotency key → Makes retried side effects happen only once
- Exponential backoff with jitter → Spreads retries out so they don't synchronize
- Fallback response → Keeps the feature usable when a dependency is down

Each pattern addresses a different failure mode.

</details>

## 6. Reflection (hard)

A month after launch, your RAG assistant's thumbs-down rate doubles, though uptime and latency look normal. What would you investigate, and in what order?

<details>
<summary>Answer</summary>

**Model answer.** First, what changed: deployments of prompts, model version (including silent provider updates), embedding model, or index; and whether the index is stale (new documents not ingested). Then input drift: are users asking new kinds of questions (cluster recent queries and compare with launch)? Then retrieval quality on recent failing queries (is the evidence retrieved?) versus generation (is it grounded?). Use traces with versions and retrieved IDs, add the failures to the eval set, and fix the root cause.

Quality monitoring needs version tracking, drift analysis, and per-layer diagnosis.

</details>
