---
title: Multi-agent systems
summary: Recognize the single-agent ceiling, choose between pipeline, fan-out, orchestrator-worker, and peer designs, compute their token and latency cost against a single agent, and build an orchestrator that hands out briefs and merges summaries.
skill: llms
minutes: 40
prerequisites: [tool-use-and-agents, reliable-agents]
related: [agent-memory, model-context-protocol, llm-evaluation, reliability-cost-and-observability]
adapted_from: [phases/16-multi-agent-and-swarms/01-why-multi-agent]
---

# Multi-agent systems

> **Mental model.** A multi-agent system is a manager and a team. The manager splits the job, hands each person a written brief, and later reads their one-page reports, not their notebooks. Each person works with a clear desk. It finishes faster when the pieces are independent, costs more in total hours, and fails in the handoffs: a vague brief, two people doing the same work, or reports that don't fit together.

**You will learn to**
- Name the three single-agent limits that motivate splitting: context saturation, role confusion, and sequential execution.
- Choose between pipeline, fan-out/fan-in, orchestrator-worker, and peer (swarm) designs for a given task shape.
- Compute total input tokens, peak context, and critical-path length for a single agent and for an orchestrator with workers.
- Build an orchestrator that sends structured briefs to parallel workers and merges only their summaries.
- Decide when a single agent with good context management beats a multi-agent design.

**Why it matters.** "Should this be multi-agent?" is one of the most common agent design questions, and one of the most commonly answered wrong. Splitting helps broad, parallel work like research across many sources. It hurts tightly coupled work like most code edits, where every handoff loses context. You need numbers, not intuitions, to tell the two apart.

## 1. Intuition

**The single-agent ceiling.** One agent is one loop, one context window, and one system prompt. Three things eventually break:
1. **Context saturation.** Every tool result stays in the context. By step 30, the window holds pages of file contents and search results, the early details get lost, and every call re-reads all of it.
2. **Role confusion.** One prompt that says "research, then code, then review, then test" produces an agent that does each job half-well. Reviewing your own work in the same context also inherits your own blind spots.
3. **Sequential execution.** Reading three independent sources takes three rounds, one after another, even though nothing about the second depends on the first.

**Splitting fixes each one, at a price.** Give each agent one job, its own fresh context, and a prompt tuned for that job. Workers read a lot and return a little, so the coordinator's context stays small. Independent workers run at the same time. The price: every boundary is lossy (a worker's 20,000 tokens of reading become a 400-token summary), coordination is new code that can fail, and total tokens can go up because every worker re-reads its own instructions and may re-discover what another already found.

**A spectrum, not a switch.**

| Design | Shape | Good for | Main risk |
|---|---|---|---|
| Single agent | one loop | most tasks; tightly coupled steps | context overflow on long runs |
| Subagents | parent spawns a child for a side task, gets a summary back | keeping the parent's context clean (e.g. "search the codebase for X") | child lacks context the parent had |
| Pipeline | A → B → C, fixed stages | staged work: research → draft → review | one stage's error flows downstream |
| Fan-out / fan-in | split into parallel workers, merge | independent subtasks known in advance | merge step must reconcile conflicts |
| Orchestrator-worker | a model decides the subtasks at run time and delegates | breadth-first research, subtasks not known in advance | vague briefs, duplicated work |
| Peer / swarm | agents share a queue or message bus, no fixed leader | high-throughput queues of similar items | hard to debug; emergent loops |

Pipeline and fan-out are *workflows*: your code fixes the control flow. Orchestrator-worker and swarms are *agentic*: a model decides it. The rule from [tool use and agents](01-tool-use-and-agents.md) still applies: if you can write the steps down, write them down.

**The handoff is the product.** A worker only knows what its brief says. A good brief has the objective, what is already known, the boundaries (what not to do, which sources to skip), the tools to use, and the exact output format and length. A good report is short, structured, and cites what it found, so the orchestrator can merge and check it without re-doing the work.

> [!IMPORTANT]
> **Multi-agent pays when work is broad and independent.** It buys parallel speed and clean contexts. It does not make a model smarter at a single coupled problem, and it usually costs more tokens than a single agent that manages its context well.

## 2. Visualization

![Two panels for synthetic token sizes. Left: total input tokens versus number of independent subtasks k from 1 to 10. A single agent that keeps everything rises quadratically to about 712 thousand at k=10. A single agent that compacts each finished subtask rises to about 189 thousand. An orchestrator with workers doing 3 steps each rises linearly to about 99 thousand. Workers that need 6 steps each reach about 323 thousand. Right: sequential model calls on the critical path: a single agent needs 3k calls, rising to 30, while orchestrator plus workers stays at 5.](../../figures/multi-agent-systems.png)

*Synthetic sizes. Left: the big token saving is against a single agent that never compacts; against one that does, the gap is small, and if workers must re-discover context they cost more. Right: the robust win is latency, because parallel workers keep the critical path flat.*

**Try it** (run [`multi_agent.py`](../../code/19-agents/multi_agent.py), predict first, then check):
1. Before running, predict which design wins on total tokens at k = 1. Check the left panel.
2. Set `M` (summary length) to 4,000 tokens. Predict what happens to the orchestrator's synthesis call and its peak context.
3. Make workers need 6 steps (the dashed line). At what k does multi-agent become cheaper than the non-compacting single agent?
4. Suppose the subtasks are *not* independent: worker 2 needs worker 1's result. What happens to the critical path?

## 3. The math

### Symbols

| Symbol | Meaning | Unit |
|---|---|---|
| $k$ | independent subtasks | count |
| $s$ | tool steps per subtask | count |
| $B$ | single agent's base context (prompt, tools, task) | tokens |
| $D$ | tokens appended per step (model output + tool result) | tokens |
| $B_O$, $B_W$ | orchestrator's and each worker's base context | tokens |
| $P$, $M$ | plan length; each worker's summary length | tokens |
| $C$ | total input tokens billed | tokens |

One agent loop of $T$ calls bills $TB + D\,T(T-1)/2$ input tokens, because each call re-reads everything before it (derived in [tool use and agents](01-tool-use-and-agents.md)).

**Single agent** doing all $k$ subtasks in one loop, $T = ks$:

```math
C_{\text{single}} = ksB + D\,\frac{ks(ks-1)}{2}, \qquad \text{critical path} = ks \text{ calls}
```

**Orchestrator with workers**: one planning call, $k$ parallel worker loops, one synthesis call that reads the plan and the $k$ summaries:

```math
C_{\text{multi}} = B_O + k\left(sB_W + D\,\frac{s(s-1)}{2}\right) + (B_O + P + kM), \qquad \text{critical path} = s + 2 \text{ calls}
```

The single agent's quadratic term grows with $(ks)^2$; the workers' grows with $k s^2$. Splitting removes the cross-subtask re-reading.

### Worked example

Synthetic sizes: $k = 4$, $s = 3$, $B = 2{,}000$, $D = 1{,}500$, $B_O = 2{,}000$, $B_W = 1{,}500$, $P = 300$, $M = 400$.

**Single agent.**
1. $T = 4 \times 3 = 12$ calls.
2. $C = 12 \times 2{,}000 + 1{,}500 \times \frac{12 \times 11}{2} = 24{,}000 + 1{,}500 \times 66 = 24{,}000 + 99{,}000 = 123{,}000$.
3. Peak context (last call): $2{,}000 + 11 \times 1{,}500 = 18{,}500$ tokens.

**Orchestrator with workers.**
1. Plan: $2{,}000$.
2. One worker: $3 \times 1{,}500 + 1{,}500 \times \frac{3 \times 2}{2} = 4{,}500 + 4{,}500 = 9{,}000$. Four workers: $36{,}000$.
3. Synthesis: $2{,}000 + 300 + 4 \times 400 = 3{,}900$.
4. Total: $2{,}000 + 36{,}000 + 3{,}900 = 41{,}900$.
5. Peak context: a worker's last call, $1{,}500 + 2 \times 1{,}500 = 4{,}500$, versus synthesis $3{,}900$, so $4{,}500$.
6. Critical path: $1 + 3 + 1 = 5$ sequential calls instead of 12.

**The fair baseline.** A single agent that replaces each finished subtask's history with a 400-token summary pays, for subtask $j = 0, 1, 2, 3$, $s(B + jM) + D\,s(s-1)/2$:

1. Base reads: $3 \times 2{,}000 \times 4 = 24{,}000$.
2. Summary reads: $3 \times 400 \times (0 + 1 + 2 + 3) = 7{,}200$.
3. Within-subtask growth: $4 \times 1{,}500 \times 3 = 18{,}000$.
4. Total: $49{,}200$, close to the multi-agent $41{,}900$, with no coordination code.

**When workers duplicate work.** If each worker needs 6 steps because it must re-discover context the orchestrator already had, one worker costs $6 \times 1{,}500 + 1{,}500 \times 15 = 9{,}000 + 22{,}500 = 31{,}500$; four cost $126{,}000$; the total is $2{,}000 + 126{,}000 + 3{,}900 = 131{,}900$, more than the naive single agent.

Real systems that explore broadly spend far more. Anthropic [reported](https://www.anthropic.com/engineering/multi-agent-research-system) that its multi-agent research system used roughly 15 times the tokens of a chat interaction, and that this paid off only on tasks valuable enough and parallel enough to justify it. The model above shows why: the token bill depends on how much each worker explores, while the latency win depends only on independence.

## 4. Implementation

### Build it from scratch

The full script is [`code/19-agents/multi_agent.py`](../../code/19-agents/multi_agent.py). Agents are mocks (no API calls) so the run is offline and deterministic in tokens; "latency" is a short sleep per model call.

**The cost model.** One helper bills an agent loop; the two designs are built from it, so the arithmetic above is exactly what the code computes.

```python
def loop_tokens(base, d, steps):
    """Input tokens billed by one agent loop: each call re-reads base + everything appended so far."""
    return steps * base + d * steps * (steps - 1) // 2

def single_agent(k, s):
    t = k * s
    return {"input": loop_tokens(B, D, t), "critical_calls": t, "peak": B + (t - 1) * D}
```

*Predict:* doubling $k$ roughly multiplies the single agent's tokens by what factor, once $k$ is large?

**The orchestrator design.** Plan, $k$ independent worker loops, then one synthesis call that reads only the summaries.

```python
def orchestrator_workers(k, s):
    plan = B_O
    workers = k * loop_tokens(B_W, D, s)
    synth = B_O + P + k * M
    return {"input": plan + workers + synth, "critical_calls": 1 + s + 1,
            "peak": max(B_W + (s - 1) * D, synth), "parts": (plan, workers, synth)}
```

*Predict:* with $k = 50$ subtasks, which call has the largest context, a worker or the synthesis?

**A worker gets a brief, not the orchestrator's context.** It works in its own context and returns a short, addressed message. Nothing else crosses the boundary.

```python
def worker(brief, steps):
    """A mock worker: its own fresh context (the brief only), a few tool steps, then a short summary."""
    for _ in range(steps):
        time.sleep(CALL_SECONDS)
    return {"to": "orchestrator", "from": brief["id"], "summary": f"findings on {brief['topic']} ({M} tokens)",
            "tokens": loop_tokens(B_W, D, steps)}
```

*Predict:* the brief has no field saying which sources other workers cover. What failure does that invite?

**Fan out, fan in.** Briefs are structured (goal, format, length cap). Workers run concurrently in a thread pool; the orchestrator waits for all of them, then synthesizes.

```python
def run_fanout(topics, steps):
    briefs = [{"id": f"w{i + 1}", "topic": t, "goal": "3 key facts with sources", "format": "bullets, max 400 tokens"}
              for i, t in enumerate(topics)]
    t0 = time.perf_counter()
    time.sleep(CALL_SECONDS)                                          # orchestrator plans
    with ThreadPoolExecutor(max_workers=len(briefs)) as pool:         # workers run concurrently
        results = list(pool.map(lambda b: worker(b, steps), briefs))
    time.sleep(CALL_SECONDS)                                          # orchestrator synthesizes summaries only
    return results, time.perf_counter() - t0
```

*Predict:* one worker hangs. What does `pool.map` do, and what should a production orchestrator do instead?

### Use the library

Agent frameworks package these patterns: the OpenAI Agents SDK (handoffs and agents-as-tools), LangGraph (graphs of nodes with shared state), Microsoft's AutoGen (conversations between agents), CrewAI (role-based crews), and Anthropic's Claude Agent SDK (subagents that run with their own context and return a summary). What they add over the script: real model calls, tool routing, shared state or message passing, retries, timeouts, and tracing across agents. What they do not add is judgment about whether to split. Start with one agent, measure where it fails, and split only the part that is broad and parallel.

### Run it

```bash
uv run code/19-agents/multi_agent.py
```

Key lines of output (wall-clock times vary slightly by machine):

```
single agent: T = 4*3 = 12 calls, input = 12*2000 + 1500*12*11/2 = 123,000, peak context 18,500
orchestrator+workers: plan 2,000 + workers 4*9,000 = 36,000 + synthesis 3,900 = 41,900, peak context 4,500
critical path: 12 sequential calls vs 5
single agent that compacts each finished subtask to 400 tokens: 49,200
if each worker needs 6 steps: 131,900 input tokens (single agent: 123,000)
wall clock 0.28s vs 0.70s if the same calls ran one after another
```

## 5. Engineering

**Decide with three questions.** Are the subtasks independent (no worker needs another's result)? Does a single context overflow or slow down on this task? Is the task valuable enough to pay for parallel exploration? Three yeses: consider orchestrator-worker. Otherwise, one agent with good tools and [context management](03-agent-memory.md) usually wins.

**Write briefs like tickets.** Objective, known facts, scope boundaries, tools and sources, and an output contract (format, length cap, citations). Most multi-agent failures trace back to a brief that assumed context the worker didn't have.

**Keep reports small and checkable.** Workers return structured summaries, not transcripts. The orchestrator checks them (are claims cited, do they conflict) before merging, and can send a worker back with a sharper brief.

**Budget and stop.** Give every worker a step and token cap, and the orchestrator a cap on how many workers it may spawn. Scale effort to the task: a simple question does not need ten workers.

**Observe across agents.** Trace every agent with a shared run ID so you can follow a request from plan to briefs to worker steps to synthesis. Without it, debugging means reading several transcripts and guessing how they relate. Evaluate the whole system end to end, as in [LLM evaluation](../17-llm-evaluation/01-llm-evaluation.md), and also check each handoff.

**Prefer read-only parallelism.** Parallel workers that only read and report are safe to retry and easy to merge. Parallel workers that write (edit the same files, place orders) create conflicts and side effects that are hard to undo; keep writes in one agent, or serialize them.

> [!WARNING]
> **Failure modes.** Vague briefs; workers duplicating each other; lost context at handoffs; conflicting reports merged without reconciliation; a spawning loop where the orchestrator keeps adding workers; one slow worker holding up the whole run; and agents echoing each other's mistakes (peer designs can amplify a wrong claim rather than catch it).

### Common mistakes

- Splitting a tightly coupled task (most code edits) across agents, then losing the decisions each made.
- Comparing multi-agent to a single agent that never compacts its context, and calling the token saving a win.
- Passing workers the orchestrator's full transcript "for context", which removes the clean-context benefit.
- No output contract, so the orchestrator receives ten differently shaped essays.
- No per-worker budget or timeout.
- Measuring only final quality, not total tokens and latency, so the design's cost is invisible.

## 6. Knowledge check

<!-- quiz:multi-agent-systems -->
**[Take the multi-agent systems quiz](../../quizzes/multi-agent-systems.md)**
<!-- /quiz -->

**Practice exercise.** Use the lesson's sizes ($B = B_O = 2{,}000$, $B_W = 1{,}500$, $D = 1{,}500$, $P = 300$, $M = 400$) with $k = 6$ subtasks of $s = 2$ steps each. Compute total input tokens and critical-path length for the single agent and for orchestrator-workers.

<details>
<summary>Solution</summary>

**Single agent:** $T = 12$. $C = 12 \times 2{,}000 + 1{,}500 \times 66 = 24{,}000 + 99{,}000 = 123{,}000$. Critical path 12 calls. (Same $T$ as the worked example, so the same total.)

**Orchestrator-workers:** one worker $= 2 \times 1{,}500 + 1{,}500 \times \frac{2 \times 1}{2} = 3{,}000 + 1{,}500 = 4{,}500$; six workers $= 27{,}000$. Synthesis $= 2{,}000 + 300 + 6 \times 400 = 4{,}700$. Total $= 2{,}000 + 27{,}000 + 4{,}700 = 33{,}700$. Critical path $= 1 + 2 + 1 = 4$ calls.

Shorter subtasks favour splitting: each worker's quadratic term is tiny, and the synthesis call stays small because summaries are capped.

</details>

**Implementation challenge.** Add a timeout and a retry to [`run_fanout`](../../code/19-agents/multi_agent.py): if a worker takes longer than 3 model-call times, cancel waiting for it, re-issue its brief once with `steps=1`, and mark its report as partial. Show the wall clock when one worker is slow.

<details>
<summary>Solution sketch</summary>

Submit with `futures = {pool.submit(worker, b, steps): b for b in briefs}` and collect with `concurrent.futures.wait(futures, timeout=3 * CALL_SECONDS * steps)`. For each brief whose future is not done, submit `worker(b, 1)` and add `"partial": True` to its result. Threads can't be killed, so the slow one keeps running in the background; in production you'd pass a cancellation flag or use async tasks with cancellation. Wall clock becomes roughly plan + timeout + one retry step + synthesis, instead of waiting for the slowest worker.

</details>

## Summary

- One agent hits three limits: context saturation, role confusion, and sequential execution.
- Designs range from subagents and pipelines (fixed flow) to orchestrator-worker and swarms (model-chosen flow); choose by task shape.
- Splitting turns the $(ks)^2$ re-reading term into $k s^2$ and the critical path from $ks$ to $s + 2$ calls, but against a single agent that compacts its context the token saving is small, and duplicated exploration erases it.
- The reliable win is latency and clean contexts on broad, independent work; the reliable cost is lossy handoffs and coordination code.
- Briefs and reports are the interface: write briefs like tickets, cap and structure reports, budget every worker, and trace across agents.

**Next:** [Production architecture](../20-production-ai/01-production-architecture.md)

**Related:** [Building reliable agents](02-reliable-agents.md) · [Agent memory](03-agent-memory.md) · [Model Context Protocol](04-model-context-protocol.md) · [Reliability, cost, and observability](../20-production-ai/02-reliability-cost-and-observability.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [A2A — Agent-to-Agent Protocol](https://aiengineeringfromscratch.com/lesson?path=phases/13-tools-and-protocols/19-a2a-protocol) · [Multi-Agent Debate and Collaboration](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/25-multi-agent-debate) · [Orchestration Patterns: Supervisor, Swarm, Hierarchical](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/28-orchestration-patterns) · [Supervisor / Orchestrator-Worker Pattern](https://aiengineeringfromscratch.com/lesson?path=phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern) · [Role Specialization — Planner, Critic, Executor, Verifier](https://aiengineeringfromscratch.com/lesson?path=phases/16-multi-agent-and-swarms/08-role-specialization) · [Handoffs and Routines — Stateless Orchestration](https://aiengineeringfromscratch.com/lesson?path=phases/16-multi-agent-and-swarms/11-handoffs-and-routines)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Why Multi-Agent?](https://aiengineeringfromscratch.com/lesson?path=phases/16-multi-agent-and-swarms/01-why-multi-agent).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Why would you split an agent into several agents, and what does it cost?</strong></summary>

Three reasons. Context: a single agent accumulates every tool result, so on long, broad tasks the window fills, early details get lost, and every call re-reads all of it; workers each start clean and return a short summary. Specialization: a reviewer prompt in its own context catches things the author's context would rationalize. Parallelism: independent subtasks like researching five sources can run at once, so the critical path drops from many sequential calls to a few. The costs: every handoff compresses a worker's full context into a summary and loses detail; coordination is new code that fails in new ways, like vague briefs, duplicated work, or conflicting reports; and total tokens often go up because workers re-read instructions and re-explore. I split when work is broad and independent; I keep one agent when steps are tightly coupled.

</details>

<details>
<summary><strong>A team says their multi-agent coder beats their single agent. What would you check before believing it?</strong></summary>

First the baseline: was the single agent given the same tools, budget, and context management? A single agent that never compacts its history is a weak comparison, because most of the multi-agent token and quality gain can come from clean contexts that a compacting single agent also gets. Second, cost: total tokens and wall-clock time per task, not just success rate, since multi-agent designs often spend several times more tokens. Third, the task mix: coding edits are usually tightly coupled, so I'd look at which tasks improved; gains often come from the broad, read-heavy parts like searching the codebase, which subagents handle well, while the edits themselves are better kept in one agent. Fourth, variance: run each design several times and report pass^k, not one lucky run. Then I'd decide whether the gain is worth the extra cost.

</details>

<details>
<summary><strong>Your research orchestrator's answers often contain duplicated findings and miss obvious sources. How do you debug it?</strong></summary>

Read the briefs first, because that's where these failures usually start. Duplicates mean the workers' scopes overlap: briefs like "research the market" with no boundaries, so every worker searches the same top results. Missed sources mean nobody owned them. I'd check the traces with a shared run ID: what each worker was told, which queries it ran, and what it returned. Fixes: briefs that partition the space explicitly (this worker covers regulation, from these source types, and skips news coverage), list what's already known, and set an output contract with citations. Then have the orchestrator check coverage and overlap before synthesizing, and send a worker back with a sharper brief when a gap shows up. Finally, add eval cases for coverage and duplication so the fix holds as prompts change.

</details>

<details>
<summary><strong>Four independent subtasks, three tool steps each, 2,000-token base context, 1,500 tokens added per step. Single agent or orchestrator with workers?</strong></summary>

Single agent first: 12 calls, so input tokens are 12 times 2,000 plus 1,500 times 12 times 11 over 2, which is 24,000 plus 99,000, or 123,000, with a peak context of 18,500. With an orchestrator, each worker with a 1,500-token brief costs 4,500 plus 4,500, or 9,000 tokens; four is 36,000, plus a 2,000-token plan and a 3,900-token synthesis, about 42,000 in total, with a peak of 4,500 and 5 sequential calls instead of 12. But a single agent that compacts each finished subtask to a 400-token summary costs about 49,000, so the token gap is small. I'd choose workers here for latency, since the subtasks are independent, and watch that workers don't re-explore, because at 6 steps each the multi-agent bill rises to about 132,000.

</details>
