---
title: Agents and tool use
summary: "Let models act: tool calling with JSON schemas, the agent loop, context and memory, MCP, workflows versus agents, and the engineering that makes agents reliable."
skill: engineering
---

# Module 19: Agents and tool use

An agent is a model in a loop: it reads the context, calls a tool, sees the result, and decides what to do next. The model only proposes actions; the runtime validates, authorizes, executes, and stops them. This module covers how that loop works and how to make it reliable, safe, and affordable.

| Piece | What it does | Failure mode |
|---|---|---|
| Tool schemas | describe each tool's name, purpose, and arguments | vague names or overlapping tools, so the wrong one is picked |
| Structured output | model emits a schema-valid tool call | valid JSON with wrong values |
| Runtime | validates, authorizes, executes, appends results | side effects without checks or approval |
| Context | short-term memory, re-read every step | token cost grows quadratically; overflow |
| Long-term memory | notes stored and retrieved across tasks | stale or injected memories |
| Stopping conditions | final answer, step and cost limits | infinite loops, runaway spend |
| Evaluation and tracing | task suites, trajectories, pass^k, traces | demo-driven confidence |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Tool use and agents](01-tool-use-and-agents.md) | trace a tool call end to end, compute context and token growth, explain MCP, and choose between workflow patterns and an agent |
| 2 | [Building reliable agents](02-reliable-agents.md) | quantify compounding error, design tools and guardrails, decide on multi-agent designs, and evaluate and observe agents |

**Interactive lab:** [Agent loop](https://sreshtalluri.github.io/ml-ai-learn/labs/agent-loop/)
