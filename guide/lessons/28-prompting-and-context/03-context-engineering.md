---
title: Context engineering
summary: Decide what goes into the context window and in what order (instructions, tools, retrieved documents, history, memory) under a token budget, and keep per-call cost bounded as conversations grow.
skill: llms
minutes: 40
prerequisites: [prompt-engineering, rag-pipeline]
related: [few-shot-cot, tool-use-and-agents, reliable-agents, llm-inference, production-architecture]
adapted_from: [phases/11-llm-engineering/05-context-engineering]
---

# Context engineering

**Mental model.** The context window is a small desk, not a filing cabinet. Everything the model will use for this answer has to be on the desk at once, and a cluttered desk makes the important page harder to find. Context engineering is deciding what goes on the desk, how much space each thing gets, and where it sits.

**You will learn to**

- List the components of a model call's context (system prompt, tool schemas, retrieved documents, history, memory, the query, output reserve) and budget tokens across them.
- Explain why a long window does not make long contexts free: cost, latency, and lower accuracy on facts buried in the middle.
- Order retrieved material so the most relevant items sit at the edges.
- Compact conversation history and select tools per query, and compute the savings.
- Choose between short-term context, long-term memory, and retrieval for a given piece of information.

**Why it matters.** Most failures of LLM applications that look like "the model is dumb" are context failures: the right document wasn't retrieved, it was there but buried, the instructions were pushed out by a long history, or 40 similar tool descriptions confused the choice. Prompt wording ([prompt engineering](01-prompt-engineering.md)) is one slice of the context. This lesson is about the whole window, and it ties together [RAG](../16-rag/01-rag-pipeline.md), [agents](../19-agents/01-tool-use-and-agents.md), and [inference cost](../18-llm-inference/01-llm-inference.md).

## 1. Intuition

**Everything the model knows at answer time is in the window.** A model call sees one sequence of tokens. It typically holds:

| Component | Typical role | Changes per call? |
|---|---|---|
| System prompt | role, rules, output format | rarely |
| Tool schemas | names, descriptions, JSON argument schemas | sometimes |
| Few-shot examples | demonstrations of the format | sometimes |
| Long-term memory | stored facts about the user or project | per session |
| Retrieved documents | chunks from search or a vector index | every call |
| Conversation history | earlier turns, tool results | grows every turn |
| User query | the current request | every call |
| Output reserve | room for the answer | fixed |

The [context window](../../glossary.md#context-window) limit covers input *and* output, so the output reserve comes out of the same budget.

**Big windows are not free.** Windows of hundreds of thousands of tokens, and some of a million or more, are common in 2026. But every input token costs money, adds prefill time before the first output token, and occupies [KV cache](../../glossary.md#kv-cache) memory on the server. And more text is not more understanding: measured accuracy on long-context tests generally drops as irrelevant text grows, and models often use information at the start and end of a long input better than information in the middle. That last effect is called [lost in the middle](../../glossary.md#lost-in-the-middle) (Liu et al., 2023, "Lost in the Middle: How Language Models Use Long Contexts"). Its size varies by model and task and has shrunk in newer models, but the safe assumption is still: less, better-placed context beats more context.

**Four moves cover most of context engineering.**

1. **Select.** Include only what this call needs: the top few relevant chunks, not the top twenty; the tools for this intent, not all of them.
2. **Order.** Stable material first (system prompt, tool schemas), so [prefix caching](../../glossary.md#prefix-caching) can reuse it. The most relevant retrieved items at the edges of the retrieved block, the least relevant in the middle. The question last, close to where generation starts.
3. **Compress.** Replace old conversation turns with a summary; trim long tool outputs to the fields that matter; paginate large results.
4. **Remember outside the window.** Facts that must survive across sessions go into a store (files, a database, a vector index) and are *retrieved* when relevant. Long-term memory is just retrieval over your own notes.

> [!IMPORTANT]
> **Context engineering versus prompt engineering.** Prompt engineering is how you phrase the instructions. Context engineering is everything that ends up in the window and where it sits: retrieval, tool selection, history management, memory, and ordering. In agents and RAG systems, the second matters more.

## 2. Visualization

![Left: a horizontal bar of a 128,000-token window. System prompt 500, tool schemas 4,500, retrieved documents 4,000, history 3,000, and query 200 together fill 12,200 tokens (9.5 percent); a 4,000-token output reserve sits at the right end; the rest is empty. Right: input tokens per call over 30 turns. The naive line rises linearly from 9,200 to about 16,450. The engineered line starts at 3,700, rises for five turns, then stays flat at 4,850.](../../figures/context-engineering.png)

*Synthetic component sizes for a coding assistant (token counts estimated at 1.3 tokens per word). Left: one call uses under a tenth of the window. Right: what happens across a conversation.*

The left panel is why "we have a big window" is the wrong question. The window is mostly empty; the problem is not space, it's what you pay for and what the model attends to. The right panel shows the naive approach (all 30 tools, 8 documents, full history) growing by one turn's worth of tokens every call, forever. The engineered approach (10 relevant tools, 3 documents, the last 4 turns plus a 150-token summary) stops growing once compaction kicks in.

**Try it** (predict first, then edit `context_engineering.py`):

1. Set `KEEP = 10` in the engineered setup. Predict where the green line flattens and its height.
2. Double `TURN` to 500 tokens. Predict which line changes slope and by how much.
3. In the ordering demo, add a second document with score 1.0. Predict where edges-first ordering puts it.
4. Change the query in the tool-selection demo to mention "calendar" and "email". Predict the number of selected tools and tokens saved.

## 3. The math

### Symbols

| Symbol | Meaning | Unit |
|---|---|---|
| $W$ | context window size | tokens |
| $R$ | output reserve | tokens |
| $F$ | fixed input per call: system + tools + documents + query | tokens |
| $h$ | tokens added to history per turn | tokens |
| $k$ | number of recent turns kept verbatim | count |
| $m$ | size of the running summary | tokens |
| $c_t$ | input tokens for call $t$ | tokens |
| $T$ | number of calls in the conversation | count |

### Budget

The input must fit in what's left after the reserve:

```math
F + \text{history}_t \le W - R
```

For the synthetic assistant: $W = 128{,}000$, $R = 4{,}000$, so the input budget is $124{,}000$. Components: $500 + 4{,}500 + 4{,}000 + 3{,}000 + 200 = 12{,}200$, which is $12{,}200 / 128{,}000 = 9.5\%$ of the window. Room left: $124{,}000 - 12{,}200 = 111{,}800$.

### Per-call input, naive

Keeping full history, call $t$ re-reads $t - 1$ earlier turns:

```math
c_t = F + h\,(t - 1), \qquad \sum_{t=1}^{T} c_t = T F + h\,\frac{T(T-1)}{2}
```

This is the same arithmetic series as the [agent loop](../19-agents/01-tool-use-and-agents.md): total billed input grows with $T^2$.

### Per-call input, with compaction

Keeping the last $k$ turns plus an $m$-token summary:

```math
c_t = F + \begin{cases} h\,(t-1) & t - 1 \le k \\ h\,k + m & t - 1 > k \end{cases}
```

Once $t > k + 1$, $c_t$ is constant, so the total grows linearly in $T$.

### Worked example: 20 calls

Naive: 30 tools × 150 = 4,500; 8 documents × 500 = 4,000; system 500; query 200. $F = 9{,}200$. $h = 250$.

1. Call 1: $9{,}200$. Call 20: $9{,}200 + 250 \times 19 = 9{,}200 + 4{,}750 = 13{,}950$.
2. Total: $20 \times 9{,}200 + 250 \times \frac{20 \times 19}{2} = 184{,}000 + 250 \times 190 = 184{,}000 + 47{,}500 = 231{,}500$.

Engineered: 10 tools × 150 = 1,500; 3 documents × 500 = 1,500; system 500; query 200. $F = 3{,}700$. $k = 4$, $m = 150$.

3. Calls 1 to 5 (history 0, 250, 500, 750, 1,000): $5 \times 3{,}700 + 250 \times (0+1+2+3+4) = 18{,}500 + 2{,}500 = 21{,}000$.
4. Calls 6 to 20 (15 calls), each $3{,}700 + 250 \times 4 + 150 = 3{,}700 + 1{,}000 + 150 = 4{,}850$: $15 \times 4{,}850 = 72{,}750$.
5. Total: $21{,}000 + 72{,}750 = 93{,}750$, which is $93{,}750 / 231{,}500 = 0.40$ of the naive bill.

Selection (fewer tools and documents) cut the fixed part from 9,200 to 3,700. Compaction cut the growing part. Both changes also leave fewer distractors in the window.

### Edges-first ordering

Sort items by relevance score, best first: $d_1, d_2, \dots, d_n$. Put the odd ranks at the front and the even ranks, reversed, at the back:

```math
d_1, d_3, d_5, \dots, d_6, d_4, d_2
```

With six documents scored $[1.0, 0.4, 0.2, 0, 0, 0]$, the order is $1.0, 0.2, 0, 0, 0, 0.4$: the two best at the two edges, the weakest in the middle. The relevance score here is simple word overlap, the fraction of query words in the document: for "refund window for damaged items" and "damaged items ship back free", 2 of 5 query words match, so $0.4$.

## 4. Implementation

### Build it from scratch

No model call is needed: the subject is what you would send. Start with a token counter. Real counts come from the model's tokenizer; 1.3 tokens per English word is a planning estimate.

```python
def count_tokens(text):
    """Rough estimate. Use the provider's token-count endpoint or tokenizer in production."""
    return int(len(text.split()) * 1.3)
```

*Predict:* is this estimate likely to be high or low for source code full of symbols?

The budget object reserves output space up front and trims each component to the lesser of its cap and what is left. Components added first get space first, which is why you add the essentials first.

```python
class ContextBudget:
    def __init__(self, window, reserve_output):
        self.available = window - reserve_output       # output tokens share the window
        self.used = {}

    def add(self, name, text, cap=None):
        limit = min(cap or 10**9, self.available - sum(self.used.values()))
        words = text.split()[: int(limit / 1.3)]         # keep the words that fit (1.3 tokens/word)
        self.used[name] = count_tokens(" ".join(words))
        return " ".join(words)
```

*Predict:* with an 8,000 window, 1,000 reserve, a 400-word system prompt, and 3,000 words of documents capped at 2,000 tokens, how many tokens remain? (Script: 4,481.)

Ordering: score relevance, then place the best items at the edges.

```python
def score_relevance(query, docs):
    q = set(query.lower().split())
    return [len(q & set(d.lower().split())) / len(q) for d in docs]

def edges_first(items, scores):
    """Best item first, second best last, weakest in the middle."""
    ranked = [it for _, it in sorted(zip(scores, items), key=lambda p: -p[0])]
    return ranked[::2] + ranked[1::2][::-1]
```

*Predict:* with only two documents, does edges-first change anything?

History compaction keeps the last few turns verbatim and replaces the rest with a summary. Here the summary is a stub; in production it's a cheap model call ("summarize decisions, open questions, and user preferences").

```python
def compact_history(turns, max_tokens, keep_recent=4, summarize=None):
    summarize = summarize or (lambda ts: "Summary of earlier turns: " + "; ".join(t[:40] for t in ts))
    total = sum(count_tokens(t) for t in turns)
    if total <= max_tokens:
        return turns
    old, recent = turns[:-keep_recent], turns[-keep_recent:]
    return [summarize(old)] + recent                     # a real system asks an LLM for the summary
```

*Predict:* if you re-summarize the summary every turn, what slowly happens to early details?

Tool selection keeps only the tools for the areas the query mentions. Real systems classify intent with a small model or embed tool descriptions and retrieve the closest ones.

```python
def select_tools(query):
    hits = {area for area in set(TOOLS.values()) if area in query.lower() or (area == "files" and "file" in query)}
    return [t for t, area in TOOLS.items() if area in hits]
```

*Predict:* the query "search the git log and read the file that changed" selects which areas, and how many of the 30 tools?

### Use the library

In production, each piece maps to a real component:

- **Token counts:** the provider's token-counting endpoint or the model's tokenizer (for example `tiktoken` for OpenAI models, or the Anthropic API's token-count endpoint).
- **Retrieval and ordering:** your [RAG pipeline](../16-rag/01-rag-pipeline.md) returns ranked chunks; apply a relevance threshold and a top-k, then order. A reranker gives better scores than word overlap.
- **Compaction:** a cheap model call that summarizes old turns, triggered by a token threshold. Some agent SDKs and APIs offer built-in context editing or compaction that clears old tool results automatically.
- **Memory:** a file the agent reads at session start (for example a project notes file), a key-value store of user preferences, or a vector index of past sessions queried like RAG.
- **Tool selection:** connect tools through [MCP](../../glossary.md#model-context-protocol-mcp) servers and expose only the relevant servers or tools per task.

The assembled context goes to the model as the system prompt, tool list, and messages of an ordinary API call.

### Run it

```bash
cd guide
uv run code/28-prompting-and-context/context_engineering.py
```

Key output:

```
  total input                   12,200  (9.5% of the window)
  edges-first order (what the prompt sees, top to bottom):
    1.00  refund window is 30 days for damaged items
    0.20  refund goes to the original card
    ...
    0.40  damaged items ship back free
  14 turns, 1,120 tokens -> 5 messages, 416 tokens
  all tools 30 x 150 = 4,500 tokens; selected 10 -> 1,500 tokens (67% saved)
  naive (30 tools, 8 docs, full history)     call 1  9,200  call 20 13,950  total  231,500
  engineered (10 tools, 3 docs, compacted)   call 1  3,700  call 20  4,850  total   93,750
```

Full script: [`code/28-prompting-and-context/context_engineering.py`](../../code/28-prompting-and-context/context_engineering.py).

## 5. Engineering

**Layout for caching.** Put stable content first and in a fixed order: system prompt, tool schemas, long reference documents that rarely change. Append new turns at the end; don't edit or reorder earlier messages. Prefix caches only match byte-identical prefixes, so a timestamp at the top of the system prompt can disable caching for every call.

**Retrieval quality beats quantity.** Three relevant chunks usually beat ten mediocre ones: fewer tokens, fewer distractors, less for the model to reconcile. Tune top-k and a relevance threshold on a labelled set, the same way you tune chunk size in [RAG](../16-rag/01-rag-pipeline.md).

**Repeat the critical instruction.** For long inputs, put the key instruction both before the material and right before the question ("Using only the documents above, answer…"). Placing the question after long documents generally works better than before them.

**Compaction loses information.** A summary drops details, and summaries of summaries drift. Keep task-critical facts (IDs, decisions, constraints) in a structured notes block that is updated, not re-summarized. Let the agent re-fetch old tool outputs by ID rather than keeping them all.

**Memory is untrusted input.** Anything written to long-term memory from user or web content can carry injected instructions into future sessions. Store memories with provenance, let users view and delete them, and never treat a memory as a permission (see [Securing AI systems](../21-safety-security/01-ai-security.md)).

**Measure it.** Log tokens per component per call. Build a small evaluation set where the answer depends on one fact placed at different depths in a long context, and check accuracy for your model before trusting long inputs.

### Common mistakes

- Treating a large window as permission to include everything. You pay for every token on every call, and accuracy can drop.
- Forgetting the output reserve, so long inputs cut the answer short.
- Putting changing content (dates, user names) at the top of the system prompt, which defeats prefix caching.
- Exposing dozens of overlapping tools; the model picks the wrong one more often.
- Summarizing history with no record of key decisions, so the agent repeats work or contradicts itself.
- Storing everything in memory and retrieving it all; long-term memory needs the same relevance filtering as RAG.

## 6. Knowledge check

<!-- quiz:context-engineering -->
**[Take the context engineering quiz](../../quizzes/context-engineering.md)**
<!-- /quiz -->

**Practice exercise.** A support bot has a 1,000-token system prompt, 2,000 tokens of tool schemas, and 3,000 tokens of retrieved documents per call, plus a 300-token query. Each turn adds 400 tokens of history. Compute total input tokens over 10 calls with full history, then with compaction that keeps the last 3 turns plus a 200-token summary.

<details>
<summary>Solution</summary>

$F = 1{,}000 + 2{,}000 + 3{,}000 + 300 = 6{,}300$. $h = 400$.

Full history: $10 \times 6{,}300 + 400 \times \frac{10 \times 9}{2} = 63{,}000 + 400 \times 45 = 63{,}000 + 18{,}000 = 81{,}000$.

Compaction, $k = 3$, $m = 200$: calls 1 to 4 have history $0, 400, 800, 1{,}200$, total $4 \times 6{,}300 + 2{,}400 = 27{,}600$. Calls 5 to 10 (6 calls) each have $6{,}300 + 1{,}200 + 200 = 7{,}700$, total $46{,}200$. Sum: $27{,}600 + 46{,}200 = 73{,}800$.

Only a 9% saving: history is small next to the fixed 6,300. The bigger lever here is $F$: retrieving 2 documents instead of 6 and selecting tools would cut far more.

</details>

**Implementation challenge.** Add a "needle" test to the script: build contexts of 20 documents where one contains a planted fact, place it at each position 1 to 20, and for each position record the fact's position as a fraction of the context. Then plug in a real model client and measure accuracy by position on 10 planted facts.

<details>
<summary>Solution sketch</summary>

Generate 19 filler documents (synthetic, on unrelated topics) and one with a planted fact ("The vault code is 4817"). For each position, build the context, append the question "What is the vault code?", call the model at temperature 0, and check whether the answer contains the code. Repeat with 10 different facts and fillers, and plot accuracy against position. If the curve dips in the middle for your model, use edges-first ordering and fewer documents; if it's flat at your context sizes, you have evidence that ordering matters less for this model.

</details>

## Summary

- The context window holds everything the model uses: system prompt, tools, examples, memory, retrieved documents, history, the query, and the output reserve.
- Long windows are available but not free: every token costs money and latency, and accuracy can drop on facts buried in long inputs.
- Select (fewer, more relevant items), order (stable prefix first, best items at the edges, question last), compress (summaries, trimmed tool output), and remember outside the window (retrieval over stored notes).
- Full history makes total input grow with T²; compaction makes per-call input constant after a few turns.
- Measure tokens per component and test long-context accuracy for your model instead of assuming it.

**Next:** [The RAG pipeline](../16-rag/01-rag-pipeline.md)

**Related:** [RAG pipeline](../16-rag/01-rag-pipeline.md) · [Prompt engineering](01-prompt-engineering.md) · [Few-shot and chain-of-thought](02-few-shot-cot.md) · [LLM inference](../18-llm-inference/01-llm-inference.md) · [Reliable agents](../19-agents/02-reliable-agents.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Context Engineering: Windows, Budgets, Memory, and Retrieval](https://aiengineeringfromscratch.com/lesson?path=phases/11-llm-engineering/05-context-engineering).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>What is context engineering, and how is it different from prompt engineering?</strong></summary>

Prompt engineering is how you word the instructions. Context engineering is deciding everything that goes into the window for each call and where it sits: which retrieved chunks, which tools, how much history, what long-term memory, which examples, and how much room to leave for the output. In a chatbot with one prompt, they're nearly the same thing. In RAG systems and agents, the context is assembled dynamically on every call and changes far more than the instructions do, so most quality and cost problems live there: the right document wasn't retrieved, it was buried among irrelevant ones, old turns crowded out instructions, or too many similar tools confused selection. The core moves are select, order, compress, and store outside the window.

</details>

<details>
<summary><strong>The model has a one-million-token window. Why not put the whole knowledge base in every call?</strong></summary>

Three reasons. Cost: you pay for every input token on every call, so a million-token context at a few dollars per million tokens is dollars per question instead of fractions of a cent. Latency: prefill time grows with input length, so the first token arrives much later, and the server holds a large KV cache per request, which cuts throughput. Quality: accuracy on long-context tests generally falls as irrelevant material grows, and facts in the middle of a long input are often used less reliably than facts near the edges. Long contexts are great for one-off analysis of a single large document. For repeated questions over a knowledge base, retrieval of a few relevant chunks is cheaper, faster, and usually more accurate.

</details>

<details>
<summary><strong>An agent gets worse the longer a session runs. What do you look at?</strong></summary>

I'd log tokens per component per call and look at what's growing. Usually it's history: raw tool outputs and old turns pile up until instructions and key facts are a small fraction of the context, or until the window overflows and something gets truncated silently. I'd check whether the system prompt or task statement is still intact, whether old tool results are being kept verbatim, and whether summaries are dropping decisions so the agent repeats work. Fixes: trim tool outputs to the needed fields, keep the last few turns verbatim and summarize the rest, maintain a structured notes block of decisions and constraints that is updated rather than re-summarized, and let the agent re-fetch old results by ID. Then I'd add a long-session case to the evaluation set.

</details>

<details>
<summary><strong>Fixed context is 6,000 tokens and each turn adds 500. Roughly how many input tokens over 30 turns, and what would you change?</strong></summary>

With full history, call t reads 6,000 plus 500 times t minus 1. The total is 30 times 6,000, which is 180,000, plus 500 times 30 times 29 over 2, which is 500 times 435, or 217,500. So about 397,500 input tokens, more than half from re-reading history. Keeping the last 5 turns plus a 300-token summary caps each call at 6,000 plus 2,500 plus 300, or 8,800, so the total becomes roughly 30 times 8,800 minus a little for the early turns, around 255,000. Then I'd attack the fixed 6,000: select tools per intent and retrieve fewer, better chunks. And I'd keep the stable prefix byte-identical so prefix caching discounts what remains.

</details>
