---
title: Agents and tool use
summary: Tool calling, the agent loop, token growth, MCP, workflow patterns, reliability math, and guardrails on one page.
---

# Agents and tool use

**Tool call lifecycle:** tool schemas in the prompt → model emits a structured call (constrained decoding keeps it schema-valid) → runtime validates, authorizes, executes → result appended as a tool message → model called again.

**Agent loop (ReAct):** observe the context → reason → act (tool call or final answer) → observe the result. Stops on a final answer or a runtime limit (max steps, token or cost budget, timeout, repeated call).

| Formula | Meaning |
|---|---|
| $c_t = B + \sum_{s<t}(o_s + r_s)$ | input tokens read by call $t$ |
| $C_{\text{in}} = TB + d\,T(T-1)/2$ | total billed input for $T$ calls, $d$ tokens added per step |
| $S = p^n$ | task success, $n$ independent steps at per-step success $p$ (0.95 over 10 steps ≈ 0.60) |
| $p_{\text{eff}} = 1 - (1-p)[(1-\rho) + \rho(1-p)^r]$ | per-step success with a check of recall $\rho$ and $r$ retries |
| $\text{pass@}k = 1-(1-s)^k$ | at least one of $k$ trials succeeds (capability) |
| $\text{pass}^k = s^k$ | all $k$ trials succeed (reliability) |

| Pattern | Use when |
|---|---|
| Prompt chaining | the task splits into known stages |
| Routing | inputs fall into distinct types |
| Parallelization | subtasks are independent, or you want votes |
| Orchestrator-workers | subtasks are decided at run time |
| Evaluator-optimizer | there are clear quality criteria to iterate against |
| Autonomous agent | the steps can't be written down in advance |

**MCP:** the host app runs MCP clients; each connects to an MCP server (stdio locally, HTTP remotely, JSON-RPC messages) that exposes **tools**, **resources**, and **prompts**. Write an integration once, use it from any compliant client. Server output is still untrusted.

**Memory:** short-term = the context (exact, bounded, re-billed every step). Long-term = a store the agent writes to and retrieves from (RAG's failure modes apply).

**Tool design:** narrow, specific names; descriptions say when not to use the tool; few typed arguments; idempotent writes with request keys; compact results; errors that say what to do next.

**Guardrails:** least privilege with the user's credentials; validate arguments and business rules in code; human approval for irreversible actions (show the evidence); sandbox code execution; per-task step, spend, and per-tool limits.

**Evaluate:** fixed task suite in a reproducible environment; final-state checks plus trajectory assertions; several runs per task; report per-trial success and pass^k; cost and latency per task (median, p95); adversarial cases (injection in tool results, failing tools).

**Failure modes:** loops, wrong tool from vague descriptions, hallucinated tools or IDs, context overflow, prompt injection through tool results, false success claims.

**Cost levers:** fewer steps (billed input grows roughly with $T^2$), prefix caching (stable prompt and tools, append-only history), compact tool results, parallel tool calls, smaller models for subtasks.

Lessons: [Tool use and agents](../lessons/19-agents/01-tool-use-and-agents.md) · [Building reliable agents](../lessons/19-agents/02-reliable-agents.md) · [Securing AI systems](../lessons/21-safety-security/01-ai-security.md)
