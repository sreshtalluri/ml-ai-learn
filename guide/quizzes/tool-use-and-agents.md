<!-- GENERATED from tool-use-and-agents.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Tool use and agents

Covers the lesson [Tool use and agents](../lessons/19-agents/01-tool-use-and-agents.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/tool-use-and-agents/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

In tool calling, what does the model itself produce?

- **A.** The result of running the function
- **B.** A structured request naming a tool and its arguments
- **C.** An HTTP request sent directly to the tool's API
- **D.** A Python function definition the runtime compiles

<details>
<summary>Answer</summary>

**B.** A structured request naming a tool and its arguments

The model only emits a schema-shaped call. The application validates, authorizes, executes it, and appends the result.

- **A:** The runtime runs the function; the model only sees the result when it is appended to the context.
- **B:** Correct.
- **C:** The model has no network access; it emits text, and your code decides whether to make any request.
- **D:** Tools are defined ahead of time as schemas; the model chooses among them and fills in arguments.

</details>

## 2. Arrange in order (easy)

Order the steps of one agent iteration.

- Result is appended to the context as an observation
- Runtime executes the tool
- Runtime validates and authorizes the call
- Model writes a thought and emits a tool call
- Model reads the context (system prompt, tool schemas, task, earlier steps)

<details>
<summary>Answer</summary>

1. Model reads the context (system prompt, tool schemas, task, earlier steps)
2. Model writes a thought and emits a tool call
3. Runtime validates and authorizes the call
4. Runtime executes the tool
5. Result is appended to the context as an observation

Observe, reason, act, then the runtime executes and appends the observation, and the next model call reads it.

</details>

## 3. Calculation (medium)

An agent's base context is 1,000 tokens and each step adds 500 tokens (model output plus tool result). How many input tokens are billed over 8 model calls?

<details>
<summary>Answer</summary>

**22000**

$C = TB + d\,T(T-1)/2 = 8 \times 1{,}000 + 500 \times \frac{8 \times 7}{2} = 8{,}000 + 500 \times 28 = 8{,}000 + 14{,}000 = 22{,}000$.

</details>

## 4. Calculation (hard)

Same run as the worked example: 13,200 input tokens billed over 6 calls and a final context of 3,200 tokens. With prefix caching, every token is read fresh once and from cache afterward. How many input tokens are billed at the cached rate?

<details>
<summary>Answer</summary>

**10000**

Fresh tokens equal the final context (3,200), since each token enters once. Cached = $13{,}200 - 3{,}200 = 10{,}000$.

</details>

## 5. Multiple choice (medium)

Constrained decoding against the tool's JSON schema guarantees which of these?

- **A.** The model picks the correct tool
- **B.** The argument values are correct
- **C.** The output parses and matches the schema
- **D.** The tool call is safe to execute

<details>
<summary>Answer</summary>

**C.** The output parses and matches the schema

Masking invalid tokens guarantees structure only. Choice of tool and values still come from the model's distribution, and safety is the runtime's job.

- **A:** Any tool name in the schema is valid, so a wrong but valid choice passes.
- **B:** `flight_id: "UA412"` is valid JSON even when the right flight was AS330.
- **C:** Correct.
- **D:** Schema-valid calls can still be harmful; validation of business rules, permissions, and approval are separate.

</details>

## 6. Multiple choice (hard)

Traces show the agent often answers flight-price questions using `web_search` and quotes stale prices, even though a `search_flights` tool with live data exists. What is the most likely fix?

- **A.** Raise the temperature so it explores other tools
- **B.** Rename and rewrite the tool descriptions so it is clear which tool has live fares and that web search is not a source of prices
- **C.** Add more tools so the model has more options
- **D.** Increase the context window

<details>
<summary>Answer</summary>

**B.** Rename and rewrite the tool descriptions so it is clear which tool has live fares and that web search is not a source of prices

Tool selection is driven by names and descriptions. Overlapping, vague descriptions are the usual cause of wrong-tool choices; fix them and re-run the eval suite.

- **A:** More randomness makes tool choice less consistent, not more correct.
- **B:** Correct.
- **C:** More similar tools make selection harder.
- **D:** The context is not the bottleneck here; the model has the tool list but can't tell the tools apart.

</details>

## 7. Select all that apply (medium)

Which statements about the Model Context Protocol (MCP) are true? Select all that apply.

- **A.** An MCP server can expose tools, resources, and prompts
- **B.** MCP clients run inside a host application such as an IDE or agent
- **C.** Using MCP makes tool outputs trustworthy, so injection is no longer a concern
- **D.** Local servers commonly talk over standard input/output, remote ones over HTTP

<details>
<summary>Answer</summary>

**A, B, D**

MCP standardizes the connection between hosts and integrations. It does not change the fact that tool descriptions and results are untrusted text.

- **A:** Correct.
- **B:** Correct.
- **C:** A protocol standardizes transport and message shapes, not the trustworthiness of the content; a malicious or compromised server can still inject instructions.
- **D:** Correct.

</details>

## 8. Multiple choice (medium)

Incoming support emails are either billing, technical, or sales, and each type has a fixed handling procedure. Which design fits best?

- **A.** An autonomous agent with every tool available
- **B.** Routing to one of three fixed prompt chains
- **C.** Evaluator-optimizer loop
- **D.** Orchestrator-workers

<details>
<summary>Answer</summary>

**B.** Routing to one of three fixed prompt chains

The categories and procedures are known in advance, so a router plus fixed chains is cheaper, faster, and easier to test than an agent.

- **A:** An agent adds cost, latency, and unpredictability for a task whose steps are already known.
- **B:** Correct.
- **C:** Iterative critique helps when quality criteria are the hard part, not when choosing a procedure.
- **D:** Orchestrator-workers fits when subtasks are discovered at run time; here they are fixed.

</details>

## 9. Select all that apply (easy)

Which stopping conditions should a production agent runtime enforce? Select all that apply.

- **A.** Maximum number of model calls
- **B.** Token or cost budget per task
- **C.** Detection of the same tool call repeated with identical arguments
- **D.** Trusting the model to stop when it is done, with no other limit

<details>
<summary>Answer</summary>

**A, B, C**

The model's final answer is the normal stop, but the runtime needs hard limits because a confused or manipulated model can loop indefinitely.

- **A:** Correct.
- **B:** Correct.
- **C:** Correct.
- **D:** Without hard limits, a looping agent runs until it hits the context limit or a large bill.

</details>

## 10. Reflection (medium)

Explain the difference between an agent's short-term and long-term memory, and one risk of long-term memory.

<details>
<summary>Answer</summary>

**Model answer.** Short-term memory is the context window: everything in the current run, exact but bounded and re-read (and billed) on every call. Long-term memory is an external store (notes, key-value facts, a vector index) the agent writes to and later retrieves from, like RAG. Risks: stale or wrong memories are retrieved and trusted, and content written from untrusted sources can carry injected instructions into future sessions.

Long-term memory inherits RAG's failure modes plus persistence of bad or malicious content.

</details>
