<!-- GENERATED from production-architecture.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Production architecture

Covers the lesson [Production architecture](../lessons/20-production-ai/01-production-architecture.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/production-architecture/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

Requests arrive at 30 per second and spend 2 seconds in the system. How many are in flight on average?

<details>
<summary>Answer</summary>

**60**

Little's law, $L = \lambda W = 30 \times 2 = 60$.

</details>

## 2. Calculation (medium)

70% of traffic goes to a model costing 0.2 cents per request and 30% to one costing 2 cents. What is the blended cost per request in cents?

<details>
<summary>Answer</summary>

**0.74** (within ±0.0001)

$0.7 \times 0.2 + 0.3 \times 2 = 0.14 + 0.6 = 0.74$.

</details>

## 3. Match (medium)

Match each concern to the component that should own it.

| Concept | Options |
|---|---|
| Rate limits and authentication | Governance |
| Choosing which model handles a request | Retrieval |
| Filtering documents by the user's permissions | Orchestration |
| Retention and audit policies | API gateway |

<details>
<summary>Answer</summary>

- Rate limits and authentication → API gateway
- Choosing which model handles a request → Orchestration
- Filtering documents by the user's permissions → Retrieval
- Retention and audit policies → Governance

Each concern has a natural home in the architecture.

</details>

## 4. Multiple choice (hard)

The model emits a tool call to `refund_order(order_id=991, amount=500)`. What should the application do?

- **A.** Execute it; the model decided.
- **B.** Validate the arguments with a schema, check the tool is allowlisted and that this user may refund this order (and the amount), require approval if above a threshold, then execute with an idempotency key.
- **C.** Ask the model whether it is sure.
- **D.** Ignore all tool calls.

<details>
<summary>Answer</summary>

**B.** Validate the arguments with a schema, check the tool is allowlisted and that this user may refund this order (and the amount), require approval if above a threshold, then execute with an idempotency key.

Tool calls are requests. Authorization, validation, approvals, and idempotency are enforced in code.

</details>

## 5. Multiple choice (easy)

You need to classify 2 million archived support tickets once. Which inference mode fits best?

- **A.** Online, one request per ticket as fast as possible
- **B.** Batch inference, optimized for throughput and cost
- **C.** Streaming chat
- **D.** Manual review

<details>
<summary>Answer</summary>

**B.** Batch inference, optimized for throughput and cost

No user is waiting, so batch for throughput and lower cost.

</details>

## 6. Reflection (medium)

Why use structured outputs with schema validation instead of parsing the model's free-form text?

<details>
<summary>Answer</summary>

**Model answer.** Free-form text varies in wording and format, so regex parsing breaks silently. A schema defines exactly which fields, types, and ranges are allowed; validation rejects anything else, so downstream code never acts on malformed or unexpected values. Failures become explicit (retry or fall back safely), and the contract is testable.

Validation turns probabilistic text into a checked interface.

</details>
