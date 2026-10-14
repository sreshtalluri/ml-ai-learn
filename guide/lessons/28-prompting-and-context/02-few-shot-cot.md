---
title: Few-shot and chain-of-thought
summary: Use examples to teach format, make the model reason in steps, vote over several sampled chains, and know when each technique is worth its tokens.
skill: llms
minutes: 40
prerequisites: [prompt-engineering, decoding]
related: [context-engineering, tool-use-and-agents, reliable-agents, llm-evaluation]
adapted_from: [phases/11-llm-engineering/02-few-shot-cot]
---

# Few-shot and chain-of-thought

**Mental model.** Few-shot prompting is showing a worked example instead of describing the format. Chain-of-thought is asking the model to show its working, so each step is a short, easy prediction instead of one big leap. Self-consistency is asking several students and taking the most common answer.

**You will learn to**

- Distinguish zero-shot, few-shot, zero-shot chain-of-thought, and few-shot chain-of-thought prompts.
- Choose and order few-shot examples, and explain why format copying is the main thing they buy.
- Explain why writing intermediate steps reduces errors on multi-step problems, using compounding error.
- Compute the accuracy and cost of self-consistency voting, and spot when voting makes things worse.
- Place tree-of-thought, ReAct, and built-in reasoning models relative to these techniques.

**Why it matters.** Many production tasks have more than one step: extract then compute, check a policy then decide, read a log then diagnose. These techniques are the cheapest way to raise accuracy on such tasks without training, and they are standard interview material. They also cost tokens and latency, so you need to know when they pay off.

## 1. Intuition

**Zero-shot vs few-shot.** A zero-shot prompt gives only the instruction. A [few-shot](../../glossary.md#few-shot-prompting) prompt adds a handful of input/output examples before the real input. Language models are strong pattern-completers, so an example is often a clearer specification than a paragraph of rules: the model copies the format, the label names, the length, and the level of detail. The main thing few-shot buys on a capable model is *consistency of format and style*; it can also help on unusual tasks the model rarely saw in training.

**Choosing examples.**

- Cover every label or case you care about, so the model doesn't learn "the answer is usually positive" from an unbalanced set.
- Prefer examples similar to the input. Many systems retrieve the nearest examples by embedding similarity for each query ("dynamic few-shot"), the same machinery as [RAG](../16-rag/01-rag-pipeline.md).
- Order matters: models can be biased toward the label of the last example. Shuffle or balance, and measure.
- Two to five examples is usually enough. More costs tokens on every call.

**[Chain-of-thought](../../glossary.md#chain-of-thought) (CoT).** Ask for the reasoning before the answer, either with an instruction ("Let's think step by step", called zero-shot CoT) or by including worked examples whose answers show their steps (few-shot CoT). Why it helps: a transformer does a fixed amount of computation per generated token. Asking for the final number immediately forces a multi-step calculation into one token's worth of computation. Writing "17 − 5 = 12" first puts the intermediate result into the context, and every later token can read it. Each step becomes a small, easy prediction.

**[Self-consistency](../../glossary.md#self-consistency).** Sample several CoT answers at a temperature above zero, extract each final answer, and take the most common one. Independent mistakes tend to scatter over different wrong answers, while correct chains agree. It costs N times as much.

**Beyond a single chain.**

- **Tree-of-thought (ToT)** branches: generate several candidate next steps, score them (often with the model itself), keep the best, and expand again. Worth it for search problems (puzzles, planning) where a single chain often commits to a dead end. It needs many calls.
- **ReAct** interleaves reasoning with tool calls and observations. That is the agent loop from [Tool use and agents](../19-agents/01-tool-use-and-agents.md).
- **Reasoning models** (models trained to think at length before answering, with extended reasoning built in) already do CoT internally, and many expose a "thinking budget" setting. For these, "think step by step" is redundant; provider guidance generally recommends clear goals and constraints over hand-written reasoning examples.

> [!WARNING]
> **The written chain is not a guaranteed explanation.** The steps a model writes can look plausible while the answer came from elsewhere, and a correct answer can follow flawed steps. Check final answers (and, where it matters, each step) programmatically. Don't treat the reasoning text as an audit log.

## 2. Visualization

![Line chart of accuracy versus number of reasoning steps (1 to 8) on 300 synthetic problems per point. Answering directly falls from 0.85 at one step to about 0.27 at eight steps, tracking the dotted curve 0.85 to the power s. Few-shot chain-of-thought falls from 0.95 to about 0.63, tracking 0.95 to the power s. Self-consistency with five voted chains stays between 0.94 and 1.0.](../../figures/few-shot-cot.png)

*Synthetic: generated sticker-counting word problems and a scripted stand-in model whose per-step slip rates are assumptions (15% when answering directly, 5% when writing steps). The curves show the compounding mechanism, not any real model's accuracy.*

The dotted curves are the formula in section 3: if each step goes wrong independently with probability ε, a whole $s$-step answer is right with probability $(1-\varepsilon)^s$. Lowering ε from 0.15 to 0.05 keeps a long chain usable much longer. Voting over five chains lifts accuracy further, because the toy's slips land on different wrong numbers and rarely outvote the right one.

**Try it** (predict first, then edit `few_shot_cot.py`):

1. Set `EPS_COT = EPS_DIRECT`. Predict whether CoT still beats answering directly, and what self-consistency does.
2. Run self-consistency at `temperature=0`. Predict the five answers. (Greedy decoding gives the same chain every time, so voting does nothing.)
3. Make every slip the same (`new += 1` instead of a random offset). Predict how much self-consistency still helps at 8 steps.
4. Remove the `Answer:` line from `EXAMPLE`. Predict what the strict extractor finds in few-shot outputs.

## 3. The math

### Symbols

| Symbol | Meaning | Shape / unit |
|---|---|---|
| $s$ | number of reasoning steps in a problem | count |
| $\varepsilon$ | probability one step goes wrong | probability |
| $p$ | probability one chain gives the right final answer | probability |
| $N$ | number of sampled chains in self-consistency | count |
| $K$ | number of chains that are right | count |
| $b, d$ | tree-of-thought branching factor and depth | count |

### Compounding error

If each step fails independently with probability $\varepsilon$, a chain is right only if all $s$ steps are right:

```math
p = (1 - \varepsilon)^s
```

Worked example, $s = 4$ steps:

1. Answering directly, $\varepsilon = 0.15$: $0.85^2 = 0.7225$, then $0.7225^2 = 0.5220$.
2. Writing the steps, $\varepsilon = 0.05$: $0.95^2 = 0.9025$, then $0.9025^2 = 0.8145$.

The simulation measured 0.530 and 0.820 on 300 problems each, within sampling noise of 0.522 and 0.815.

### Self-consistency as a vote

With $N$ independent chains each right with probability $p$, the number right $K$ is binomial. A strict majority of right chains guarantees the vote is right:

```math
P(K > N/2) = \sum_{k = \lfloor N/2 \rfloor + 1}^{N} \binom{N}{k} p^k (1 - p)^{N - k}
```

Worked example, $p = 0.7$, $N = 5$ (need $k \ge 3$):

1. $k = 3$: $\binom{5}{3} \times 0.7^3 \times 0.3^2 = 10 \times 0.343 \times 0.09 = 0.30870$.
2. $k = 4$: $\binom{5}{4} \times 0.7^4 \times 0.3 = 5 \times 0.2401 \times 0.3 = 0.36015$.
3. $k = 5$: $0.7^5 = 0.16807$.
4. Sum: $0.30870 + 0.36015 + 0.16807 = 0.83692$.

One chain gives 0.70; five voting chains give at least 0.837, for five times the tokens. This is a *lower bound*: the vote uses the most common answer, not a strict majority, so when wrong answers scatter, two right chains can win against three different wrong ones. That is why the simulated self-consistency line sits above the bound (at 4 steps: bound 0.953, measured 0.990).

**When voting hurts.** If wrong chains agree (a systematic misreading, the same wrong formula every time) and $p = 0.4$, the vote picks the wrong answer whenever $K \le 2$. Then $P(K \ge 3) = 10 \times 0.064 \times 0.36 + 5 \times 0.0256 \times 0.6 + 0.01024 = 0.2304 + 0.0768 + 0.01024 = 0.31744$, worse than a single chain's 0.4. Voting amplifies whatever the majority does.

### Cost

If a direct answer costs $a$ output tokens and CoT adds $r$ reasoning tokens, self-consistency costs about $N(a + r)$ output tokens plus $N$ reads of the prompt (cacheable, since the prompt is identical). Tree-of-thought with branching $b$ and depth $d$ generates $b + b^2 + \dots + b^d$ thoughts before pruning; with $b = 3$, $d = 3$ that is $3 + 9 + 27 = 39$ generation calls, plus scoring calls. Pruning cuts this, but the bill grows fast.

## 4. Implementation

### Build it from scratch

The script builds three prompt styles, a stand-in model, answer extraction, and a voting loop. First the prompts. The few-shot CoT example shows both the reasoning style and the exact `Answer: N` line we will parse.

```python
EXAMPLE = ("Q: Mia has 7 stickers. She buys 3 more. She doubles them. How many stickers does she have?\n"
           "A: Start with 7. Buys 3 more: 7 + 3 = 10. Doubles them: 10 * 2 = 20.\nAnswer: 20")

def zero_shot(q):
    return f"Q: {q}\nA:"

def zero_shot_cot(q):
    return f"Q: {q}\nA: Let's think step by step."

def few_shot_cot(q):
    return f"{EXAMPLE}\n\nQ: {q}\nA:"
```

*Predict:* which of the three prompts tells the model what the final line should look like?

The stand-in model reads the question from the prompt, decides whether it was asked to reason, and walks the steps. Each step slips with probability ε. If the prompt contains an `Answer:` example, it copies that format; otherwise it picks one of several phrasings, as real models do.

```python
    cot = "step by step" in prompt or "Start with" in prompt          # reasoning requested or shown
    eps = EPS_COT if cot else EPS_DIRECT
    key = f"{q}|{seed if temperature > 0 else 0}"                     # greedy: same answer every call
    rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    for full, buy, give in steps:
        new = x + int(buy) if buy else x - int(give) if give else 2 * x
        if rng.random() < eps:                                         # a slip: off by 1 to 3
            new += int(rng.choice([-3, -2, -1, 1, 2, 3]))
        lines.append(f"{full}: {new}.")
        x = new
```

*Predict:* the random generator is seeded from the question and, only when temperature is above zero, the seed. What does self-consistency at temperature 0 return?

Extraction is where few-shot pays off. A strict parser only accepts the format you asked for; a lenient one takes the last number, which works until the model writes "18 stickers, up from 12".

```python
def extract_strict(text):
    m = re.search(r"Answer:\s*(-?\d+)", text)
    return int(m.group(1)) if m else None

def extract_last_number(text):
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None
```

*Predict:* zero-shot CoT output ends with "So she ends up with 18". Which extractor finds 18?

Self-consistency is five sampled chains and a vote. `Counter.most_common` gives the plurality, which is why it beats the strict-majority bound when wrong answers scatter.

```python
def self_consistency(q, n=5, temperature=0.7):
    answers = [extract_strict(call_model(few_shot_cot(q), temperature, seed)) for seed in range(n)]
    votes = Counter(a for a in answers if a is not None)
    return votes.most_common(1)[0][0] if votes else None, answers
```

*Predict:* answers come back `[10, 10, 12, 9, 13]`. What does the vote return, and how many chains were right if the truth is 10?

Finally, the binomial bound from section 3, used to check the simulation:

```python
def p_majority(p, n):
    """P(more than half of n independent chains are right): a lower bound on voting accuracy."""
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))
```

*Predict:* `p_majority(0.5, 5)`? (By symmetry, exactly 0.5.)

### Use the library

With a real model, `call_model` sends the prompt and returns text (see the client code in [Prompt engineering](01-prompt-engineering.md)). Two production adjustments:

- Most chat APIs take few-shot examples as alternating `user` / `assistant` messages rather than one long string. That keeps the example answers clearly in the assistant's voice.
- For self-consistency, some APIs return several samples from one request (an `n` parameter); others need N calls, which you run concurrently. Keep the prompt byte-identical so [prefix caching](../../glossary.md#prefix-caching) bills the repeated input at a discount.

For reasoning models, skip the CoT scaffolding: set the provider's reasoning or thinking budget, state the goal and output format, and parse the final answer. Frameworks such as DSPy go further and select few-shot examples automatically by optimizing against your test set.

### Run it

```bash
cd guide
uv run code/28-prompting-and-context/few_shot_cot.py
```

Key output:

```
Format: strict 'Answer: N' found in 24% of zero-shot CoT outputs, 100% of few-shot CoT.
steps   direct  (0.85^s)     CoT  (0.95^s)   SC N=5  (majority bound)
    4    0.530     0.522   0.820     0.815    0.990             0.953
    8    0.267     0.272   0.630     0.663    0.937             0.785
Worked example: p = 0.7 per chain, N = 5 -> P(at least 3 right) = 0.83692
Systematic error (wrong chains agree): p = 0.4, N = 5 -> 0.31744, worse than one chain
```

Full script: [`code/28-prompting-and-context/few_shot_cot.py`](../../code/28-prompting-and-context/few_shot_cot.py).

## 5. Engineering

**Picking a technique.**

| Technique | Calls | Extra tokens | Use when |
|---|---|---|---|
| Zero-shot | 1 | none | simple, well-specified tasks |
| Few-shot | 1 | examples on every call | format, labels, or style must match exactly |
| Chain-of-thought | 1 | reasoning tokens | multi-step reasoning on a non-reasoning model |
| Self-consistency | N | about N times | high-stakes answers with a checkable final value |
| Tree-of-thought | many | tens of times | search problems where single chains dead-end |
| Reasoning model | 1 | hidden reasoning tokens (billed) | hard reasoning without hand-built scaffolding |

**Latency.** CoT and reasoning models add output tokens, and output tokens are generated one at a time (see [LLM inference](../18-llm-inference/01-llm-inference.md)). A 300-token chain at 50 tokens per second adds 6 seconds. For interactive features, measure whether the accuracy gain is worth it, or reason once offline and cache.

**Hiding the reasoning.** If users should see only the answer, ask for the reasoning inside tags (`<thinking>…</thinking>`) and the answer in a separate tag, then show only the answer. Or use structured output with a `reasoning` field followed by an `answer` field, so reasoning comes first in generation order.

**Few-shot maintenance.** Examples are data: version them, keep them out of your test set (otherwise you're grading the model on questions it was shown), and refresh them when the input distribution shifts.

**Verification beats voting.** When you can check an answer (run the code, recompute the arithmetic, validate against a schema), a generate-then-verify loop is cheaper and stronger than voting. Self-consistency is for answers you can compare but not check.

### Common mistakes

- Adding "think step by step" to a reasoning model, or to a simple lookup, and paying for tokens that don't help.
- Few-shot examples that all have the same label, so the model learns the label instead of the task.
- Putting test-set items into the few-shot examples and reporting inflated accuracy.
- Running self-consistency at temperature 0 (every chain identical) or with a lenient extractor that misreads intermediate numbers as answers.
- Trusting the written reasoning as an explanation for compliance or debugging, without checking it.
- Using voting on a problem where the model makes the same mistake every time; it confirms the mistake.

## 6. Knowledge check

<!-- quiz:few-shot-cot -->
**[Take the few-shot and chain-of-thought quiz](../../quizzes/few-shot-cot.md)**
<!-- /quiz -->

**Practice exercise.** A model answers a 6-step problem directly with per-step error 0.1, or with chain-of-thought at per-step error 0.03. Compute both single-chain accuracies. Then compute the strict-majority lower bound for self-consistency with N = 3 chains of CoT.

<details>
<summary>Solution</summary>

Direct: $0.9^6$. $0.9^2 = 0.81$, $0.81^3 = 0.81 \times 0.6561 = 0.5314$.

CoT: $0.97^6$. $0.97^2 = 0.9409$, $0.9409^3 = 0.9409 \times 0.88529 = 0.8330$.

Self-consistency, $N = 3$, need $k \ge 2$, $p = 0.833$:
$k = 2$: $3 \times 0.833^2 \times 0.167 = 3 \times 0.6939 \times 0.167 = 0.3476$.
$k = 3$: $0.833^3 = 0.5780$.
Sum: $0.9256$. Three chains move accuracy from 0.833 to at least 0.926, at three times the cost.

</details>

**Implementation challenge.** Add a verifier to the script: for each CoT chain, recompute every step line ("buys 4 more: 16") from the previous value and reject chains with any wrong step. Compare "first chain that passes verification" against five-chain voting on 8-step problems, in accuracy and in average number of chains generated.

<details>
<summary>Solution sketch</summary>

Parse each line with a regex into (operation, operand, claimed result), recompute from the previous claimed value, and flag mismatches. Generate chains one at a time and stop at the first that verifies (cap at 5). Because the toy's slips are always visible arithmetic errors, the verifier catches every one, so accuracy approaches 1.0 and most problems need only 1 or 2 chains (expected chains is about 1 / 0.663 for 8 steps, capped at 5). Voting always uses 5. The lesson: a cheap check beats more samples, when a check exists.

</details>

## Summary

- Few-shot examples specify format and style by demonstration; choose balanced, relevant examples and keep them out of the test set.
- Chain-of-thought turns one hard prediction into several easy ones; with per-step error ε, an s-step chain is right with probability (1 − ε)^s.
- Self-consistency votes over N sampled chains; it helps when errors scatter and hurts when errors are systematic, at N times the cost.
- Tree-of-thought searches over branches for puzzle-like problems; ReAct adds tools; reasoning models build the chain in.
- Prefer a verifier to voting whenever answers can be checked.

**Next:** [Context engineering](03-context-engineering.md)

**Related:** [Prompt engineering](01-prompt-engineering.md) · [Tool use and agents](../19-agents/01-tool-use-and-agents.md) · [Reliable agents](../19-agents/02-reliable-agents.md) · [Decoding](../15-llms/02-decoding.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Voting, Self-Consistency, and Debate Topology](https://aiengineeringfromscratch.com/lesson?path=phases/16-multi-agent-and-swarms/15-voting-debate-topology)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Few-Shot, Chain-of-Thought, Tree-of-Thought](https://aiengineeringfromscratch.com/lesson?path=phases/11-llm-engineering/02-few-shot-cot).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Why does chain-of-thought prompting improve accuracy if the model's weights do not change?</strong></summary>

Because it changes how much computation goes into the answer. A transformer does a fixed amount of work per generated token. If I ask for the final number immediately, a multi-step calculation has to happen inside the forward pass for that one token. With chain-of-thought, the model writes intermediate results as tokens, and every later token can attend to them, so each step is a short, easy prediction conditioned on the previous result. Errors still compound: with per-step error ε, an s-step chain is right with probability (1 − ε)^s. CoT helps by lowering ε, not by removing compounding. It costs output tokens and latency, it helps little on single-step tasks, and reasoning models already do it internally, so adding it there is redundant.

</details>

<details>
<summary><strong>When is self-consistency worth five times the cost, and when does it fail?</strong></summary>

It's worth it when the final answer is short and comparable (a number, a label, a choice), single-chain accuracy is moderate, and errors are fairly random, so wrong chains disagree with each other. Then the plurality is usually right: with p = 0.7 per chain, five chains give at least 0.84 by the majority bound, often more because scattered wrong answers can't outvote the right one. It fails when errors are systematic. If the model misreads the problem the same way most of the time, voting confirms the misreading; at p = 0.4 with agreeing wrong chains, five-way voting drops to about 0.32. It also does nothing at temperature zero. If I can verify answers instead, a generate-and-check loop is cheaper and more reliable.

</details>

<details>
<summary><strong>Your few-shot classifier labels almost everything as the class of the last example. What is going on?</strong></summary>

That's a known in-context bias: models lean toward labels that appear often or recently in the examples. I'd check three things. First, label balance: if three of four examples are "positive", the prior shifts. Second, order: if the last example's label dominates, the model is copying recency. Third, similarity: if the last example is closest to most inputs, the model may be matching surface form. Fixes: balance examples across labels, shuffle or rotate order and measure the effect, retrieve the most similar examples per input rather than a fixed set, and give the label definitions explicitly in the instruction too. Then measure per-class precision and recall on a held-out set to confirm the bias is gone, not just moved.

</details>

<details>
<summary><strong>A 5-step task has per-step error 0.1 without chain-of-thought and 0.04 with it. What accuracy gain do you expect, and what does it cost?</strong></summary>

Without CoT: 0.9 to the fifth, which is 0.81 squared times 0.9, about 0.59. With CoT: 0.96 to the fifth. 0.96 squared is 0.9216, squared again 0.8493, times 0.96 gives about 0.815. So roughly 59 to 82 percent, if steps fail independently. The cost is output tokens: if reasoning adds 150 tokens per answer and the direct answer was 10, output grows 16 times, and output tokens are the slow, expensive ones. At 50 tokens a second that's about 3 extra seconds. For an offline batch job that's an easy yes. For an interactive feature, I'd compare against a reasoning model or caching before accepting the latency.

</details>
