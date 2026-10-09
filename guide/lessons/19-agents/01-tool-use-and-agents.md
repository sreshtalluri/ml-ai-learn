---
title: Tool use and agents
summary: Explain how tool calling works from JSON schema to executed call, run the agent loop with stopping conditions and context budgets, use MCP to connect tools, and decide when a fixed workflow beats an autonomous agent.
skill: llms
minutes: 45
prerequisites: [decoding, rag-pipeline]
related: [reliable-agents, ai-security, production-architecture, llm-evaluation]
---

# Tool use and agents

> **Mental model.** A language model can only write text. Tool calling teaches it to write one special kind of text: a structured request ("call `search_flights` with these arguments"). Your code runs the request and pastes the result back into the conversation. An agent is that exchange in a loop: the model reads everything so far, picks the next action, sees what happened, and repeats until it decides it is done or your code stops it.

**You will learn to**
- Trace a tool call end to end: schema in the prompt, structured output from the model, validation and execution by the runtime, result appended to the context.
- Explain the agent loop (observe, reason, act) and the ReAct pattern, and choose stopping conditions.
- Compute how context and billed tokens grow with the number of steps, and how prefix caching changes the bill.
- Distinguish short-term memory (the context) from long-term memory (retrieval), and plan-and-execute from reflection.
- Describe the Model Context Protocol (clients, servers, tools, resources, prompts) and why a standard protocol helps.
- Pick between the common workflow patterns and an autonomous agent for a given task.

**Why it matters.** Most useful LLM products now act, not just answer: they search, read files, query databases, file tickets, and write code. The model is the same next-token predictor you studied in [decoding](../15-llms/02-decoding.md). Everything that makes it an agent lives in the runtime around it, and that is the part you design, debug, and get asked about in interviews.

## 1. Intuition

**Tool calling is a protocol, not a capability the model runs.** The application sends the model a list of tools. Each tool has a name, a description in plain English, and a JSON schema for its arguments. When the model decides a tool would help, it outputs a structured call instead of prose, for example `{"name": "get_fare_rules", "arguments": {"flight_id": "AS330"}}`. The model never executes anything. Your runtime parses the call, checks it, runs the real function, and appends the result to the conversation as a new message. Then it calls the model again.

Models get good at this through fine-tuning on examples of tool use, and the runtime makes the output reliable with **structured outputs**: [constrained decoding](../../glossary.md#constrained-decoding) masks every next token that would break the JSON schema, so the call always parses. (Constrained decoding guarantees the *shape*, not that the arguments are *right*.)

**The agent loop.** An [agent](../../glossary.md#agent) wraps tool calling in a loop:

1. **Observe:** the model reads the context: system prompt, tool schemas, the user's task, and every earlier step.
2. **Reason:** it writes a short plan or thought ("UA412 is cheapest; check whether it is refundable").
3. **Act:** it emits a tool call, or a final answer.
4. The runtime executes the call and appends the **observation**. Back to 1.

Interleaving reasoning text with actions and observations is called [ReAct](../../glossary.md#react) (reason + act). Writing the thought before the action helps the model keep track of what it has learned and what is left.

**Stopping conditions.** The loop ends when the model produces a final answer with no tool call. Because the model can also loop forever, the runtime always adds hard limits: a maximum number of steps, a token or cost budget, a wall-clock timeout, and stops on repeated identical calls. A step limit is not a failure mode; running without one is.

**Context is the agent's working memory, and it only grows.** Each step appends the model's output and the tool result. The next model call re-reads all of it. So a long task costs more per step as it goes, and eventually the context fills up. That is why agents need *context management*: truncating or summarizing old observations, returning compact tool results, and moving facts that should outlive the task into **long-term memory**: a store the agent writes to and later retrieves from, exactly like [RAG](../16-rag/01-rag-pipeline.md) over its own notes.

**Planning patterns.**
- **Plan-and-execute:** one model call writes a step-by-step plan; then steps run one at a time (often by a cheaper model), with replanning only when something fails. Fewer expensive reasoning calls; less adaptive.
- **Reflection:** after producing a draft or finishing a step, the model (or a second prompt) critiques the result against the goal and revises. Useful when there is a checkable criterion (tests pass, answer cites sources); wasteful when there is not.

**MCP.** Every application used to wire up every tool by hand. The [Model Context Protocol](../../glossary.md#model-context-protocol-mcp) (MCP) is an open standard for that connection. An **MCP server** wraps a system (a file system, a database, an issue tracker) and exposes three kinds of things: **tools** (functions the model can call), **resources** (data the application can read into context, such as a file), and **prompts** (reusable templates). An **MCP client** lives inside the host application (an IDE, a chat app, your agent) and connects to servers over standard input/output locally or over HTTP remotely, using JSON-RPC messages. The host lists the server's tools, converts them into the tool schemas it sends to the model, and forwards the model's calls to the server. The benefit is the usual one for protocols: write a server once and every compliant client can use it, instead of N applications times M integrations. The cost is the usual one too: a server you install is code and text you are trusting, so its tool descriptions and outputs are untrusted input (see [Securing AI systems](../21-safety-security/01-ai-security.md)).

**Workflows versus agents.** Not every LLM system with tools should be an autonomous agent. In a **workflow**, your code fixes the control flow and calls the model at chosen points. In an **agent**, the model chooses the control flow. Five workflow patterns cover most products:

| Pattern | Control flow | Good for |
|---|---|---|
| Prompt chaining | fixed sequence; each call's output feeds the next, with checks between | tasks that split cleanly into stages (outline → draft → format) |
| Routing | a classifier call picks one of several specialized paths | distinct input types (refund vs technical vs sales) |
| Parallelization | independent calls run at once, then results are combined (sectioning) or compared (voting) | independent subtasks; higher confidence via several samples |
| Orchestrator-workers | a model splits the task into subtasks at run time and delegates them | subtasks you can't list in advance (which files to change) |
| Evaluator-optimizer | one call generates, another critiques against criteria, loop | clear quality criteria, iterative refinement |
| Autonomous agent | model picks every next action in a loop | open-ended tasks where the number and order of steps is unknown |

> [!IMPORTANT]
> **A fixed workflow beats an agent when you can write the steps down.** Workflows are cheaper, faster, easier to test, and fail in predictable places. Reach for an agent when the path genuinely depends on what the model discovers along the way, and when you can tolerate higher cost and latency and have a way to check the result.

## 2. Visualization

<!-- lab:agent-loop -->
![Three panels for a synthetic agent with a 1,200-token starting context that grows by 400 tokens per step. Left: bars of input tokens per call rising linearly from 1,200 to 4,800 over 10 steps. Middle: cumulative input tokens rising quadratically, 13,200 after 6 steps and 30,000 after 10. Right: cumulative cost without caching rises to about 0.10 USD by step 10, while with prefix caching (cached input billed at 10%) it rises to about 0.034 USD.](../../figures/tool-use-and-agents.png)

*Synthetic token counts and prices. Each model call re-reads the whole context, so the per-call input grows linearly and the total billed input grows quadratically. Prefix caching bills the re-read part at a discount.*

*Interactive version: [open the lab on the website](https://sreshtalluri.github.io/ml-ai-learn/labs/agent-loop/).*
<!-- /lab -->

The lab runs a fully scripted agent (no real model, no real bookings) on the task "find the cheapest refundable flight under \$300 and hold it" with four mocked tools.

**Try it** (predict first, then check):
1. Press **Step** through the happy path. Before step 3, predict how many input tokens that model call will read. Why is it more than step 2, even though step 2's tool result was small?
2. Turn on **Search tool returns an error**. Predict whether the agent fails, and how many extra model calls and tokens the error costs.
3. Turn on **Ambiguous tool descriptions**. Predict which tool the model picks first and whether its final answer is right. Compare the final answer with the hold observation just above it.
4. Set **max steps** to 3, then turn on **Require approval for side effects** with max steps back at 8. In each case, what does the runtime do that the model cannot?

## 3. The math

### Symbols

| Symbol | Meaning | Shape / unit |
|---|---|---|
| $B$ | base context: system prompt + tool schemas + user task | tokens |
| $o_t$ | tokens the model writes at step $t$ (thought + tool call) | tokens |
| $r_t$ | tokens of the tool result appended at step $t$ | tokens |
| $c_t$ | input tokens read by the model call at step $t$ | tokens |
| $T$ | number of model calls in the run | count |
| $\pi_{\text{in}}, \pi_{\text{out}}, \pi_{\text{cache}}$ | price per token for input, output, cached input | USD/token |

### Context growth

Call $t$ reads the base plus everything earlier steps appended:

```math
c_t = B + \sum_{s=1}^{t-1} (o_s + r_s)
```

If every step adds the same $d = o + r$ tokens, $c_t = B + (t-1)d$ grows linearly, and the total input tokens billed over $T$ calls is an arithmetic series:

```math
C_{\text{in}}(T) = \sum_{t=1}^{T} \big(B + (t-1)d\big) = T B + d\,\frac{T(T-1)}{2}
```

The $T^2$ term is why long agent runs get expensive faster than you expect.

### Worked example: a six-step run

Synthetic numbers: system prompt 500 tokens, four tool schemas 600 tokens, user task 100 tokens. Each step the model writes $o = 80$ tokens and the tool returns $r = 320$ tokens. The run takes $T = 6$ model calls.

1. Base context: $B = 500 + 600 + 100 = 1{,}200$ tokens.
2. Tokens added per step: $d = 80 + 320 = 400$.
3. Input read by each call: $c_1 = 1{,}200$, $c_2 = 1{,}600$, $c_3 = 2{,}000$, $c_4 = 2{,}400$, $c_5 = 2{,}800$, $c_6 = 3{,}200$.
4. Sum: $1{,}200 + 1{,}600 + 2{,}000 + 2{,}400 + 2{,}800 + 3{,}200 = 13{,}200$.
5. Check with the formula: $6 \times 1{,}200 + 400 \times \frac{6 \times 5}{2} = 7{,}200 + 400 \times 15 = 7{,}200 + 6{,}000 = 13{,}200$. ✓
6. Output tokens: $6 \times 80 = 480$.

The final context is only 3,200 tokens, but the run billed 13,200 input tokens, about 4 times as many.

**Cost without caching** (synthetic prices: \$3 per million input tokens, \$15 per million output tokens):

```math
13{,}200 \times 3\times10^{-6} + 480 \times 15\times10^{-6} = 0.0396 + 0.0072 = 0.0468 \text{ USD}
```

**Cost with prefix caching.** Each call's context starts with exactly the previous call's context, so an inference server with [prefix caching](../../glossary.md#kv-cache) can reuse the already-computed prefix, and providers bill those cached tokens at a discount (here, synthetic: 10% of the input price, \$0.30 per million). Over the run, each token is read fresh once and from cache afterward:

1. Fresh input tokens = the final context = $c_6 = 3{,}200$.
2. Cached input tokens = $13{,}200 - 3{,}200 = 10{,}000$.
3. Cost $= 3{,}200 \times 3\times10^{-6} + 10{,}000 \times 0.3\times10^{-6} + 480 \times 15\times10^{-6}$
4. $= 0.0096 + 0.0030 + 0.0072 = 0.0198$ USD, a 58% cut.

(Real pricing can add a premium for writing to the cache and expire cached prefixes after minutes; the shape of the saving is the same.) Caching only works if the prefix is byte-identical, which is one reason to keep the system prompt and tool list stable and append rather than edit history.

### Constrained decoding in one line

At each decoding step the model produces a distribution $p(y_t \mid y_{<t})$ over the vocabulary. A grammar compiled from the JSON schema gives the set $A_t$ of tokens that keep the output valid. Constrained decoding samples from

```math
\tilde p(y_t) = \frac{p(y_t)\,\mathbb{1}[y_t \in A_t]}{\sum_{y \in A_t} p(y)}
```

so the output always parses. It does not make the model choose the right tool or the right values; those still come from $p$.

## 4. Implementation

The whole runtime fits in a loop. The model here is a placeholder for any chat API that accepts tool schemas and returns either text or a tool call.

```python
TOOLS = [{"name": "get_fare_rules",
          "description": "Return refundability and change fee for one flight.",
          "input_schema": {"type": "object",
                           "properties": {"flight_id": {"type": "string"}},
                           "required": ["flight_id"]}}]          # ... plus the other tools

def run_agent(task, max_steps=8, approve=ask_human):
    messages = [{"role": "user", "content": task}]
    for step in range(max_steps):                                # hard stop
        reply = model(system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": reply})
        if not reply.tool_calls:                                 # final answer
            return reply.text
        for call in reply.tool_calls:
            if (err := validate(call, TOOLS)):                   # schema check
                result = {"error": err}
            elif call.name in SIDE_EFFECTS and not approve(call):
                result = {"status": "declined_by_user"}          # human gate
            else:
                result = execute(call)                           # your code runs it
            messages.append({"role": "tool", "id": call.id, "content": result})
    return "Stopped: step limit reached."
```

Note what the model never touches: validation, permissions, execution, and the step limit are all runtime code.

Runnable script (scripted mock model, four mocked tools, schema validation, error retry, approval gate, the token arithmetic above, and the figure; fully offline): [`code/19-agents/agent_loop.py`](../../code/19-agents/agent_loop.py).

## 5. Engineering

**Writing tool schemas.** The description is a prompt. Say what the tool does, what it returns, and when *not* to use it. Use specific names (`search_flights`, not `search`). Keep the tool list short: every schema costs tokens on every call and every extra similar-looking tool is a chance to pick the wrong one. Return compact, structured results, and make errors informative ("flight_id not found; use search_flights first") so the model can recover. The [next lesson](02-reliable-agents.md) covers tool design in depth.

**Parallel tool calls.** Most APIs let the model emit several independent calls in one turn (check fare rules for three flights at once). The runtime runs them concurrently and appends all results. This cuts model calls and wall-clock time.

**Context management.** Truncate or summarize old tool outputs once they have been used; keep the task, the plan, and key facts. Paginate large results and let the model ask for more. For long tasks, have the agent write notes to a file or memory store and reload only what it needs. Keep the system prompt and tool list stable at the front of the context so prefix caching keeps working.

**Memory.** Short-term memory is the context: exact, but bounded and expensive. Long-term memory is a store outside the model (key-value facts, a vector index of past interactions, files) that the agent writes to deliberately and retrieves from like RAG. It has RAG's failure modes: stale or wrong memories get retrieved and trusted, and memory written from untrusted content can carry injected instructions into later sessions.

**MCP in practice.** Use it to reuse existing integrations and to give one agent many tools without bespoke glue. Pin and review the servers you install, run local servers with least privilege, require authentication for remote ones, and treat tool descriptions and results from third-party servers as untrusted text.

**Choosing workflow or agent.** Start with the simplest thing: one model call with retrieval. Add a fixed chain or router when the task has known stages. Use an agent only for genuinely open-ended work, and give it a budget, a step limit, and a way to verify its result.

> [!WARNING]
> **Failure modes.** Infinite or repetitive loops (calling the same tool with the same arguments); picking the wrong tool when descriptions overlap; invented tool names or arguments; claiming success the observations don't support; context overflow on long tasks; and [prompt injection](../../glossary.md#prompt-injection) through tool results, where a web page or email the agent reads contains instructions.

### Common mistakes

- Believing the model "runs" the tool. It only proposes a call; your code decides whether to run it.
- Letting the model's output reach a side-effecting API without validation, authorization, and (for irreversible actions) human approval.
- No step or cost limit, so a confused agent loops until it hits the context limit or a large bill.
- Dumping raw tool output (full HTML pages, 10,000-row query results) into the context.
- Editing earlier messages or reordering tools mid-run, which breaks the prefix cache.
- Building an autonomous agent for a task a three-step chain would solve more cheaply and reliably.
- Treating constrained decoding as correctness. Valid JSON with the wrong `flight_id` is still wrong.

## 6. Knowledge check

<!-- quiz:tool-use-and-agents -->
**[Take the tool use and agents quiz](../../quizzes/tool-use-and-agents.md)**
<!-- /quiz -->

**Practice exercise.** An agent starts with a 2,000-token context (system prompt, eight tool schemas, task). Each step adds 150 output tokens and a 650-token tool result. It finishes in 10 model calls. How many input tokens are billed in total, and what is the size of the final call's context?

<details>
<summary>Solution</summary>

$B = 2{,}000$, $d = 150 + 650 = 800$, $T = 10$.

Total input: $10 \times 2{,}000 + 800 \times \frac{10 \times 9}{2} = 20{,}000 + 800 \times 45 = 20{,}000 + 36{,}000 = 56{,}000$ tokens.

Final call reads $c_{10} = 2{,}000 + 9 \times 800 = 2{,}000 + 7{,}200 = 9{,}200$ tokens. The run billed about 6 times its final context. Halving the tool result to 325 tokens would make $d = 475$ and the total $20{,}000 + 475 \times 45 = 41{,}375$.

</details>

**Implementation challenge.** Extend [`agent_loop.py`](../../code/19-agents/agent_loop.py): (1) stop early if the model emits the same tool call with the same arguments twice in a row, and (2) after step 4, replace every tool result older than two steps with a one-line summary. Print the billed input tokens before and after the change for the happy path, assuming the summary is 30 tokens.

<details>
<summary>Solution sketch</summary>

Keep `last_call = None`; before executing, `if call == last_call: return "Stopped: repeated call"`. For compaction, store per-message token counts, and when building the context for step $t > 4$ substitute 30 tokens for tool messages from steps $< t-2$. Recompute $c_t$ as the sum over the compacted messages. Only results older than two steps shrink, so the saving grows with run length, and the cost is that the model can no longer see old details unless you let it re-fetch them.

</details>

## Summary

- Tool calling is a protocol: the model emits a schema-shaped call, the runtime validates, authorizes, executes, and appends the result. Constrained decoding guarantees valid structure, not correct choices.
- An agent is that exchange in a loop (observe, reason, act), with hard stopping conditions owned by the runtime.
- Each call re-reads the whole context, so billed input tokens grow as $TB + d\,T(T-1)/2$; compact results, manage context, and keep a stable prefix for caching.
- Memory splits into the context (short-term) and retrieval over stored notes (long-term); planning patterns trade adaptivity for cost.
- MCP standardizes how hosts connect to tools, resources, and prompts; it reduces integration work but not the need to treat tool output as untrusted.
- Prefer a fixed workflow (chain, route, parallelize, orchestrate, evaluate-optimize) when you can write the steps down; use an agent when you can't.

**Next:** [Building reliable agents](02-reliable-agents.md)

**Related:** [Decoding](../15-llms/02-decoding.md) · [RAG pipeline](../16-rag/01-rag-pipeline.md) · [Securing AI systems](../21-safety-security/01-ai-security.md) · [Production architecture](../20-production-ai/01-production-architecture.md)

## Interview angle

<details>
<summary><strong>Walk me through exactly what happens when an LLM "calls a tool".</strong></summary>

The model never executes anything; it emits a structured request and the application does the rest. The application sends tool definitions (name, description, JSON schema for arguments) along with the conversation. The model, fine-tuned on tool-use examples, decides a tool helps and generates a tool-call block naming the tool and its arguments, often under constrained decoding so the JSON always matches the schema. The runtime parses it, validates the arguments, checks the user is authorized, runs the real function with timeouts, and appends the result as a tool message. Then it calls the model again with the longer context. The model reads the result and either calls another tool or answers. So correctness and safety live in the runtime: schema validity is guaranteed by decoding, but whether that was the right tool with the right values is not.

</details>

<details>
<summary><strong>When would you build a fixed workflow instead of an autonomous agent?</strong></summary>

Whenever I can write the steps down in advance. If a support request always goes classify, retrieve policy, draft, check, then a routing step plus a prompt chain is cheaper, faster, and much easier to test, because each stage has a known input and output and fails in a known place. An agent lets the model choose the control flow, which buys flexibility on open-ended tasks, such as fixing a bug whose location you don't know or research where the next query depends on what you found. The cost is more model calls, variable latency, a quadratic token bill as context grows, and failures that are harder to reproduce. I usually start with one call plus retrieval, add a chain or router, and only move to an agent, with a step budget and a verifier, when the workflow keeps needing branches it can't anticipate.

</details>

<details>
<summary><strong>Your agent keeps calling the same search tool with nearly identical queries until it hits the step limit. What do you check?</strong></summary>

First the trace: what each search returned and what the model thought after it. Usually the tool result isn't giving the model what it needs to make progress. Common causes: the result is empty or truncated without saying so; an error comes back as an unhelpful string, so the model retries blindly; the result is too large and the useful part gets lost; or the description doesn't say which tool answers this question, so it keeps using a general search instead of the specific one. Then I check the stopping logic, whether the model is told what "done" looks like, and whether old results were compacted away so it forgets what it already tried. Fixes: informative, compact results ("0 results; try a broader date range"), clearer tool descriptions, a duplicate-call detector in the runtime, and an eval case that reproduces the loop.

</details>

<details>
<summary><strong>An agent starts with 4,000 tokens and each step adds 1,000. Roughly what does a 20-step run cost in input tokens, and how would you cut it?</strong></summary>

Each call reads $4{,}000 + 1{,}000(t-1)$ tokens, so the total is $20 \times 4{,}000 + 1{,}000 \times \frac{20 \times 19}{2} = 80{,}000 + 190{,}000 = 270{,}000$ input tokens, even though the final context is only 23,000. Most of the bill is re-reading history, so the levers are: prefix caching, which bills the repeated prefix at a fraction of the input price if I keep the system prompt and tools stable and only append; smaller tool results, since $d$ multiplies the quadratic term; compacting or summarizing old observations; parallel tool calls to cut the number of steps; and routing easy subtasks to a smaller model. Cutting steps matters most, because the cost grows roughly with $T^2$.

</details>
