<!-- GENERATED from multi-agent-systems.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Multi-agent systems

Covers the lesson [Multi-agent systems](../lessons/19-agents/05-multi-agent-systems.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/multi-agent-systems/) grades these interactively and tracks a review queue.

## 1. Select all that apply (easy)

Which are limits of a single agent that motivate splitting work across agents? Select all that apply.

- **A.** The context fills with tool results and early details get lost
- **B.** One prompt covering several roles does each role less well
- **C.** Independent subtasks run one after another
- **D.** A single agent cannot call tools

<details>
<summary>Answer</summary>

**A, B, C**

Context saturation, role confusion, and sequential execution are the three limits. Single agents call tools fine.

- **A:** Correct.
- **B:** Correct.
- **C:** Correct.
- **D:** Tool calling is what makes it an agent.

</details>

## 2. Multiple choice (easy)

When is a pipeline (A then B then C) a better fit than parallel fan-out?

- **A.** When each stage needs the previous stage's output
- **B.** When the subtasks are fully independent
- **C.** When you want the lowest possible latency
- **D.** When there is no clear order of work

<details>
<summary>Answer</summary>

**A.** When each stage needs the previous stage's output

Pipelines suit staged work such as research, then draft, then review, where each step consumes the last.

- **A:** Correct.
- **B:** Independent subtasks should run in parallel.
- **C:** Pipelines are sequential, so they do not minimize latency.
- **D:** Without an order, a fixed pipeline does not apply.

</details>

## 3. Multiple choice (easy)

What distinguishes the orchestrator-worker pattern from a fixed fan-out?

- **A.** The workers share one context window
- **B.** A model decides at run time which subtasks to create and delegates them
- **C.** There is no synthesis step
- **D.** Workers talk to each other with no coordinator

<details>
<summary>Answer</summary>

**B.** A model decides at run time which subtasks to create and delegates them

In fan-out your code fixes the split; in orchestrator-worker an LLM plans the subtasks, writes briefs, and merges the results.

- **A:** Each worker has its own fresh context; that is the point.
- **B:** Correct.
- **C:** The orchestrator synthesizes the workers' reports.
- **D:** That describes a peer or swarm design.

</details>

## 4. Calculation (medium)

A single agent does $k = 5$ subtasks of $s = 2$ steps each in one loop. Base context $B = 1{,}000$ tokens, and each step appends $D = 500$ tokens. Using $C = TB + D\,T(T-1)/2$ with $T = ks$, how many input tokens are billed?

<details>
<summary>Answer</summary>

**32500**

$T = 10$. $C = 10 \times 1{,}000 + 500 \times \frac{10 \times 9}{2} = 10{,}000 + 500 \times 45 = 10{,}000 + 22{,}500 = 32{,}500$.

</details>

## 5. Calculation (easy)

An orchestrator plans (1 call), runs 8 workers in parallel with 4 steps each, then synthesizes (1 call). How many model calls are on the critical path?

<details>
<summary>Answer</summary>

**6**

Workers run concurrently, so the critical path is $1 + 4 + 1 = 6$ calls, versus $8 \times 4 = 32$ for one agent doing it all.

</details>

## 6. Multiple choice (medium)

A team reports their multi-agent design uses 65% fewer tokens than their single agent. What is the most important thing to check?

- **A.** Whether the single agent compacted or summarized finished subtasks, since most of that saving can come from clean contexts alone
- **B.** Whether the workers used the same model
- **C.** Whether the dashboard rounds numbers
- **D.** Whether the orchestrator's prompt is long

<details>
<summary>Answer</summary>

**A.** Whether the single agent compacted or summarized finished subtasks, since most of that saving can come from clean contexts alone

Against a single agent that never compacts, the multi-agent design looks cheap. A compacting single agent often closes most of the gap without coordination code.

- **A:** Correct.
- **B:** Model choice matters for cost, but it does not explain the comparison's fairness.
- **C:** Rounding cannot create a 65% difference.
- **D:** Orchestrator prompt length is a small, fixed cost.

</details>

## 7. Multiple choice (medium)

A research orchestrator's final reports repeat the same findings several times and miss obvious sources. Where do you look first?

- **A.** The temperature of the synthesis call
- **B.** The briefs. Overlapping, unbounded scopes cause duplicates; unowned areas cause gaps
- **C.** The embedding model
- **D.** The number of GPUs

<details>
<summary>Answer</summary>

**B.** The briefs. Overlapping, unbounded scopes cause duplicates; unowned areas cause gaps

Workers only know their brief. Partition the space explicitly, list what is known, and set an output contract with citations.

- **A:** Temperature affects wording, not which sources workers search.
- **B:** Correct.
- **C:** The duplication comes from overlapping assignments, not from retrieval vectors.
- **D:** Hardware does not decide what each worker searches.

</details>

## 8. Multiple choice (medium)

For a multi-file refactor where each edit depends on decisions made in earlier edits, which design is usually best?

- **A.** A swarm of peer agents editing files in parallel
- **B.** Fan-out with one worker per file and a merge step
- **C.** One agent with good tools and context management, optionally using subagents for read-only searches
- **D.** A pipeline with a separate agent per file, run in sequence with summaries in between

<details>
<summary>Answer</summary>

**C.** One agent with good tools and context management, optionally using subagents for read-only searches

Tightly coupled edits lose decisions at every handoff. Keep writes in one agent; use subagents for broad, read-only work like searching the codebase.

- **A:** Parallel writers create conflicts and inconsistent decisions.
- **B:** Per-file workers can't see each other's design choices.
- **C:** Correct.
- **D:** Each handoff compresses context, so later files lose earlier decisions.

</details>

## 9. Reflection (hard)

In two or three sentences, explain why an orchestrator-worker design reliably wins on latency for independent subtasks but does not reliably win on total tokens.

<details>
<summary>Answer</summary>

**Model answer.** Latency depends on the critical path, and parallel workers keep it at roughly plan + one worker's steps + synthesis regardless of how many subtasks there are. Total tokens depend on how much each worker reads and re-explores; every worker re-reads its own brief and tools and may re-discover context, so against a single agent that compacts its history the token saving is small, and duplicated exploration can make multi-agent more expensive.

The latency win needs only independence; the token win needs workers to explore efficiently and the baseline to be a non-compacting single agent.

</details>
