---
title: Building reliable agents
summary: Quantify how errors compound over agent steps, design tools agents use well, contain actions with permissions and approvals, decide when multi-agent systems pay off, and evaluate, observe, and budget agents in production.
skill: engineering
minutes: 40
prerequisites: [tool-use-and-agents]
related: [ai-security, llm-evaluation, reliability-cost-and-observability]
---

# Building reliable agents

> **Mental model.** An agent is a chain of bets. Each step is probably right, but the task succeeds only if every step is right, so small per-step error rates multiply into large task failure rates. Reliable agents come from fewer steps, steps that are easier to get right, checks that catch mistakes before they spread, and walls that limit the damage when something still goes wrong.

**You will learn to**
- Compute task success from per-step reliability ($p^n$) and the gain from verification and retry.
- Design tools that are narrow, well named, idempotent, and return useful errors.
- Apply least privilege, human approval, and sandboxing to agent actions.
- Decide when a multi-agent design is worth its extra tokens.
- Evaluate agents with task success, trajectory checks, cost and latency per task, and pass^k versus pass@k.
- Diagnose the common failure modes: loops, tool hallucination, context overflow, and prompt injection through tool results.

**Why it matters.** Agent demos are easy and agent products are hard. The gap is almost entirely engineering: tool design, guardrails, evaluation, and tracing. These are also the topics interviewers probe once you say you have "built an agent".

## 1. Intuition

**Errors compound.** If each step of a 10-step task is right 95% of the time and steps fail independently, the task succeeds only about 60% of the time. An agent that looks great on any single step can still fail 4 times out of 10. Three levers fight this: make each step more reliable (better tools, clearer instructions), use fewer steps (a better tool that does in one call what took three), and catch errors before they propagate (validate tool arguments, check results, retry).

**Tools are the agent's interface, so design them like an API for a new colleague.** A good agent tool does one thing, has a name that says what, takes few arguments with clear types, and returns a compact result. It is **idempotent** where possible: calling `hold_booking(request_id=...)` twice with the same request ID makes one hold, not two, so a retry after a timeout is safe. Its errors say what went wrong and what to do next ("flight_id AS33 not found; flight IDs look like AS330, get them from search_flights"). A vague tool list is the single most common cause of agents choosing the wrong action.

**Guardrails limit what a wrong step can do.** The model will sometimes be wrong and can be manipulated by text it reads, so the runtime, not the prompt, enforces safety:
- **Least privilege:** the agent gets only the tools and scopes this task needs, with the *user's* permissions, never an admin key.
- **Human approval for irreversible or high-impact actions:** payments, deletions, sending email, merging code. The model proposes; a person confirms.
- **Sandboxing:** code execution and file access happen in an isolated container with no network or a strict allowlist, and resource limits.
- **Validation:** arguments are schema-checked and business-rule-checked (amount within limit, recipient on an allowlist) before execution.

**Multi-agent systems.** One model call can act as an *orchestrator* that splits work and hands pieces to *worker* agents, each with its own fresh context and tools, then combines their results. This helps when subtasks are genuinely independent and parallel (research several sources at once), or when a single context would overflow. It costs more tokens, because each worker re-reads instructions and some work gets duplicated, and it adds coordination failures: workers misunderstand their brief, duplicate each other, or return results that don't fit together. For tightly coupled tasks, where every step depends on the last, one agent with good tools usually wins.

**Evaluate agents on outcomes and on the path.** Final-state checks ask: was the right flight held, did the tests pass? **Trajectory evals** look at the steps: did it call a forbidden tool, ask for approval, take 30 steps for a 5-step task? Because agents are stochastic, run each task several times. **pass@k** (succeeds at least once in $k$ tries) measures capability; **[pass^k](../../glossary.md#passk)** (succeeds in all $k$ tries) measures the reliability users actually experience.

## 2. Visualization

![Left: task success versus number of steps for per-step success 0.99, 0.95, and 0.90; the 0.95 curve passes 0.60 at 10 steps, and a dashed curve for 0.95 with verification and one retry (effective 0.988) stays near 0.89 at 10 steps. Right: bar chart comparing a single agent (120 s, 105k tokens) with an orchestrator and three parallel workers (60 s, 131k tokens).](../../figures/reliable-agents.png)

*Synthetic numbers. Left: independent per-step errors multiply, and a verifier plus one retry raises effective per-step reliability. Right: a parallel multi-agent design halves wall-clock time here but spends about 25% more tokens.*

**Try it** (predict first, then check in the [agent loop lab](01-tool-use-and-agents.md#2-visualization)):
1. Turn on **Ambiguous tool descriptions**. Predict at which step the run goes wrong and whether anything in the runtime catches it. What single tool-design change prevents it?
2. Turn on **Search tool returns an error**. The error is marked `retryable`. Predict what would happen if the error were just the string `"failed"`.
3. Turn on **Require approval for side effects** and reject the hold. Which part of the system guarantees the hold did not execute: the model or the runtime?
4. Combine the ambiguous tools with approval on. Would a human approver reading only the proposed call `hold_booking(DL1180)` catch the mistake? What would they need to see?

## 3. The math

### Symbols

| Symbol | Meaning | Range |
|---|---|---|
| $p$ | probability one step is correct | $[0, 1]$ |
| $n$ | number of steps that must all be correct | count |
| $S$ | task success probability | $[0, 1]$ |
| $\rho$ | verifier recall: share of wrong steps the check catches | $[0, 1]$ |
| $r$ | number of retries after a caught error | count |
| $s$ | per-trial task success rate | $[0, 1]$ |
| $k$ | number of independent trials | count |

### Compounding error

Assuming independent steps:

```math
S = p^n
```

**Worked example.** $p = 0.95$, $n = 10$.

1. $0.95^2 = 0.9025$
2. $0.95^4 = 0.9025^2 = 0.81450625$
3. $0.95^5 = 0.81450625 \times 0.95 = 0.7737809375$
4. $0.95^{10} = 0.7737809375^2 = 0.5987$ (to 4 decimals)

So $S \approx 0.60$. Cutting the task to 5 steps gives $0.95^5 \approx 0.774$. Turning it around: for $S = 0.9$ over 10 steps you need $p = 0.9^{1/10} \approx 0.9895$ per step, which is a much harder target than "95% looks fine".

> [!NOTE]
> Independence is a simplification. Real errors are correlated (one wrong assumption early can doom every later step), and agents can sometimes recover from a mistake later. The formula is a planning estimate and an argument for short paths and early checks, not a precise prediction.

### Verification and retry

Suppose a check after each step catches a wrong step with probability $\rho$, and a caught step is retried $r$ times. A step ends up wrong if it errs and either the check misses it or every retry also errs:

```math
p_{\text{eff}} = 1 - (1-p)\big[(1-\rho) + \rho\,(1-p)^r\big]
```

**Worked example.** $p = 0.95$, $\rho = 0.8$, $r = 1$.

1. $1 - p = 0.05$
2. Bracket: $(1 - 0.8) + 0.8 \times 0.05 = 0.2 + 0.04 = 0.24$
3. $p_{\text{eff}} = 1 - 0.05 \times 0.24 = 1 - 0.012 = 0.988$
4. Over 10 steps: $0.988^{10}$. $0.988^2 = 0.976144$; $0.988^4 = 0.976144^2 = 0.952857$; $0.988^5 = 0.952857 \times 0.988 = 0.941423$; $0.988^{10} = 0.941423^2 = 0.8863$.

Task success rises from 0.60 to 0.89. A checker that catches most errors is worth more than a modestly better model, which is why tests, schema checks, and "verify the claim against the observation" steps pay off.

### pass@k versus pass^k

For a task the agent solves with probability $s$ per independent trial:

```math
\text{pass@}k = 1 - (1-s)^k, \qquad \text{pass}^k = s^k
```

**Worked example.** $s = 0.7$, $k = 3$.

1. pass@3 $= 1 - 0.3^3 = 1 - 0.027 = 0.973$
2. pass^3 $= 0.7^3 = 0.343$

The same agent "solves" 97% of attempts-with-retries but behaves correctly on all three runs only 34% of the time. If users run the agent once and expect it to work, pass^k (or the plain per-trial rate) is the number to report.

### Multi-agent cost (synthetic)

A research task has three independent subtasks of about 40 s and 25k tokens each. A single agent does them in sequence and later subtasks re-read earlier findings (+15k tokens twice): 120 s and $3 \times 25{,}000 + 2 \times 15{,}000 = 105{,}000$ tokens. An orchestrator (8 s, 6k tokens) runs three workers in parallel (40 s, 35k tokens each, because each reads its brief and some searches overlap), then a synthesis step (12 s, 20k tokens): latency $8 + 40 + 12 = 60$ s, tokens $6{,}000 + 3 \times 35{,}000 + 20{,}000 = 131{,}000$, about 1.25 times the single agent. You trade tokens for wall-clock time, which is worth it only if the subtasks are truly independent and latency matters.

## 4. Implementation

The core ideas are small pieces of runtime code around the loop from the [previous lesson](01-tool-use-and-agents.md).

```python
POLICY = {                       # least privilege, per tool
    "search_flights": {"side_effect": False},
    "get_fare_rules": {"side_effect": False},
    "hold_booking":   {"side_effect": True, "approval": True, "max_per_task": 1},
}

def guarded_execute(call, ctx):
    rule = POLICY.get(call.name)
    if rule is None:
        return {"error": f"unknown tool {call.name}; available: {list(POLICY)}"}
    if (err := validate(call)):                             # schema + business rules
        return {"error": err, "retryable": False}
    if ctx.count(call.name) >= rule.get("max_per_task", 99):
        return {"error": "limit reached for this tool", "retryable": False}
    if rule.get("approval") and not ctx.ask_human(call):
        return {"status": "declined_by_user"}
    key = idempotency_key(ctx.task_id, call)                # same call -> same effect
    with ctx.trace.span(call.name, args=call.args):         # observability
        return with_retries(lambda: TOOLS[call.name](**call.args, idempotency_key=key),
                            retries=2, timeout_s=10)
```

Runnable script (compounding error with a Monte Carlo check, verify-and-retry, pass@k vs pass^k, the multi-agent comparison, and the figure): [`code/19-agents/agent_reliability.py`](../../code/19-agents/agent_reliability.py).

## 5. Engineering

**Designing tools for agents.**
- *Narrow and well named:* `get_fare_rules(flight_id)` rather than `airline_api(endpoint, params)`. Merge tools that are always called together; split tools that do unrelated things.
- *Descriptions as instructions:* when to use it, when not to, what it returns, example arguments.
- *Idempotent writes:* accept a request or idempotency key so retries can't double-book or double-charge.
- *Good errors:* say what failed, whether retrying helps, and what to try instead.
- *Compact, typed results:* return the fields the model needs, paginate the rest.
- *Test tools with the model:* run your eval suite after every tool or description change; small wording changes move tool-selection accuracy.

**Guardrails and permissions.** Scope credentials to the user and the task. Separate read tools from write tools, and gate writes. Require human approval for irreversible actions, and show the approver the evidence (price, recipient, diff), not just the call. Run generated code in a sandbox with no secrets and limited network. Enforce limits in code: max steps, max spend, max calls per tool, allowlisted domains. See [Securing AI systems](../21-safety-security/01-ai-security.md) for the threat model.

**Multi-agent: when it's worth it.** Worth it for broad, parallelizable work (research across many sources, reviewing many files) where a single context would overflow or wall-clock time matters. Not worth it for sequential tasks with shared state, such as most coding edits, where handoffs lose context. If you do it: give each worker a complete, specific brief and an output format; have the orchestrator check results; and measure the token multiplier.

**Evaluating agents.** Build a fixed suite of realistic tasks in a reproducible environment (mocked or sandboxed tools, seeded data) with automatic final-state checks. Add trajectory assertions (no forbidden tools, approval requested before writes, under N steps). Run each task several times and report per-trial success and pass^k, plus cost and latency per task (median and p95). Grade free-text outcomes with rubric-based LLM judges calibrated against human labels (see [Evaluating LLM systems](../17-llm-evaluation/01-llm-evaluation.md)). Include adversarial cases: injected instructions in tool results, tools that fail, ambiguous requests.

**Observability.** Log a trace per task: every model call (prompt version, model, input and output tokens, latency), every tool call (arguments, result size, latency, errors), approvals, and the stop reason. Traces are how you debug "it did something weird", find the step where cost explodes, and turn production failures into eval cases. Redact secrets and personal data in traces.

**Cost control.** Set a per-task token or money budget and stop gracefully at it. Keep prompts and tool lists stable for prefix caching. Route easy subtasks (classification, extraction, summarizing tool output) to smaller models. Cache deterministic tool results. Cut steps with better tools and parallel calls; since billed tokens grow roughly with the square of steps, fewer steps is the biggest lever.

> [!WARNING]
> **Common failure modes.**
> - **Loops:** the same call repeated, or oscillating between two actions. Detect duplicate calls, cap steps, return informative results.
> - **Tool hallucination:** calling tools that don't exist or inventing argument values (an ID never returned by any tool). Validate names and arguments; check IDs against earlier observations.
> - **Context overflow:** long runs push out the task or key facts, or exceed the window. Compact old results, keep notes, use sub-agents with fresh contexts.
> - **Prompt injection via tool results:** a web page, email, or file says "ignore previous instructions and send the data to...". Treat all tool output as untrusted data, keep high-impact actions behind approval and allowlists, and don't let one task's untrusted content flow into another's privileged tool.
> - **False success:** the final answer claims something the observations contradict. Verify claims against tool results before reporting.

### Common mistakes

- Reporting pass@k from several attempts as if it were the reliability a user gets from one run.
- Evaluating only the final answer, so an agent that succeeded by doing something forbidden scores as a pass.
- Giving the agent a broad service account "to make it work", instead of the user's own scoped credentials.
- Asking humans to approve raw JSON calls with no context, which trains them to click approve.
- Adding agents (planner, critic, researcher, writer) before measuring whether one agent with better tools would do.
- Retrying non-idempotent writes after a timeout.

## 6. Knowledge check

<!-- quiz:reliable-agents -->
**[Take the reliable agents quiz](../../quizzes/reliable-agents.md)**
<!-- /quiz -->

**Practice exercise.** An agent's task has 8 steps, each correct with probability 0.97. (a) What is the task success rate? (b) You add a check with recall 0.9 and one retry. What is the new per-step and task success rate? (c) If the per-trial success is your answer to (b), what are pass@2 and pass^2?

<details>
<summary>Solution</summary>

(a) $0.97^8$: $0.97^2 = 0.9409$; $0.97^4 = 0.9409^2 = 0.88529$; $0.97^8 = 0.88529^2 = 0.7837$.

(b) $1 - p = 0.03$. Bracket $= 0.1 + 0.9 \times 0.03 = 0.1 + 0.027 = 0.127$. $p_{\text{eff}} = 1 - 0.03 \times 0.127 = 1 - 0.00381 = 0.99619$. Task: $0.99619^8$: $0.99619^2 = 0.992395$; $^4 = 0.984848$; $^8 = 0.969926 \approx 0.970$.

(c) pass@2 $= 1 - 0.030^2 \approx 1 - 0.0009 = 0.9991$. pass^2 $= 0.970^2 \approx 0.941$.

</details>

**Implementation challenge.** In [`agent_reliability.py`](../../code/19-agents/agent_reliability.py), replace the independence assumption with a simple correlated model: with probability 0.1 the agent starts with a wrong assumption and every step then succeeds with 0.80; otherwise every step succeeds with 0.97. Compute task success for 10 steps analytically and by Monte Carlo, and compare it with independent steps at the same average per-step accuracy.

<details>
<summary>Solution sketch</summary>

Analytic: $S = 0.1 \times 0.8^{10} + 0.9 \times 0.97^{10} = 0.1 \times 0.10737 + 0.9 \times 0.73742 = 0.01074 + 0.66368 = 0.6744$. Average per-step accuracy $= 0.1 \times 0.8 + 0.9 \times 0.97 = 0.953$, and $0.953^{10} \approx 0.618$. For Monte Carlo, draw the "bad start" flag per trial, then draw 10 steps with the matching probability. Correlated errors concentrate failures in fewer runs, so the success rate is higher than the independent estimate at the same average, but the failing runs fail badly; this is why trajectory-level evaluation matters.

</details>

## Summary

- Task success is roughly $p^n$: 0.95 per step over 10 steps is about 0.60. Fewer steps and per-step checks with retry are the main levers.
- Good agent tools are narrow, clearly named and described, idempotent for writes, compact in output, and explicit in their errors.
- Safety comes from the runtime: least privilege, validation, human approval for irreversible actions, sandboxing, and hard limits.
- Multi-agent designs trade extra tokens and coordination risk for parallelism and fresh contexts; use them for broad, independent subtasks.
- Evaluate on a fixed task suite with final-state and trajectory checks, report pass^k alongside pass@k, track cost and latency per task, and trace every step.

**Next:** [Production architecture](../20-production-ai/01-production-architecture.md)

**Related:** [Tool use and agents](01-tool-use-and-agents.md) · [Evaluating LLM systems](../17-llm-evaluation/01-llm-evaluation.md) · [Reliability, cost, and observability](../20-production-ai/02-reliability-cost-and-observability.md)

## Interview angle

<details>
<summary><strong>Why do agents that look good on individual steps still fail so often on whole tasks?</strong></summary>

Because errors compound. If a task needs $n$ steps and each is right with probability $p$, the task succeeds with roughly $p^n$ when errors are independent: 95% per step over 10 steps is about 60%. Real errors are correlated, but the direction holds: long chains amplify small error rates, and one wrong early assumption, such as a hallucinated ID, poisons every step after it. So I attack three things. Make steps easier with better tools and clearer instructions. Make paths shorter, for example one tool that does what took three calls. And catch errors early with schema validation, checks of results against the goal, and retries. With a check that catches 80% of wrong steps and one retry, effective per-step accuracy goes from 0.95 to 0.988 and 10-step success from 0.60 to about 0.89.

</details>

<details>
<summary><strong>When is a multi-agent architecture worth it over a single agent with more tools?</strong></summary>

When the work splits into independent pieces that benefit from parallelism or from separate contexts. A breadth-first research task, such as surveying ten sources, works well with an orchestrator fanning out to workers: each worker gets a fresh context, they run concurrently, and the orchestrator synthesizes. The costs are real. Every worker re-reads its brief, work gets duplicated, and in practice the token bill can be noticeably higher than a single agent's. Handoffs also lose context, so workers can misunderstand their task or return pieces that don't fit together. For tightly coupled, sequential work like editing a codebase, where each step depends on the last, one agent with good tools is usually more reliable and cheaper. I'd decide by measuring task success, tokens, and latency for both designs on the same eval suite.

</details>

<details>
<summary><strong>Your support agent issued a refund nobody asked for. How do you investigate and prevent a repeat?</strong></summary>

Start with the trace for that task: the user's request, every model call, every tool call and result, and whether an approval step ran. I'm looking for where the decision came from. Typical causes: the model misread an ambiguous request; a tool result, such as an email or ticket body, contained injected text telling it to refund; the refund tool was described too broadly; or no approval gate existed. Prevention is in the runtime, not the prompt. Scope the agent to the customer's own orders, enforce business rules in code (amount caps, a refund only on an eligible order), require human approval for refunds with the evidence shown, make the tool idempotent with a request key, and treat ticket and email text as untrusted data. Then add this case, and injection variants of it, to the eval suite with a trajectory assertion that refunds never execute without approval.

</details>

<details>
<summary><strong>Your agent scores 0.8 per-trial success on the eval suite. A PM asks how often a customer running it three times in a week will see it work every time. What do you tell them?</strong></summary>

That's pass^3, the chance all three runs succeed: $0.8^3 = 0.512$, so about half of those customers would hit at least one failure, assuming independent runs. The tempting number is pass@3, $1 - 0.2^3 = 0.992$, but that only says the agent can solve the task if you let it try three times and keep the best result, which is not how a customer experiences it. Two caveats. Failures are usually concentrated on certain task types, so I'd break the 0.8 down by category; some customers will see near-perfect behavior and others repeated failures. And the suite has to reflect real traffic. To raise pass^k, I'd look for nondeterministic failures in the traces, add verification and retries inside a run, and lower sampling temperature for tool selection.

</details>
