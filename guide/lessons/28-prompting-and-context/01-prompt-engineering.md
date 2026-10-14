---
title: Prompt engineering
summary: Write prompts as specifications (role, task, labels, format, edge-case rules), pick decoding settings, and prove a prompt change helped with a test set and a pass rate.
skill: llms
minutes: 40
prerequisites: [decoding, tokenization-and-pretraining]
related: [few-shot-cot, context-engineering, llm-evaluation, tool-use-and-agents, ai-security]
adapted_from: [phases/11-llm-engineering/01-prompt-engineering]
---

# Prompt engineering

**Mental model.** A prompt is a specification handed to a very literal contractor. Everything you leave unsaid, the model fills in with its most typical guess. Prompt engineering is the habit of writing down the label set, the format, and the edge cases, then checking on a test set that the outputs changed the way you wanted.

**You will learn to**

- Name the parts of a chat request (system message, user message, optional prefill) and what each controls.
- Turn a vague request into a specification: role, task, allowed outputs, format, constraints, and edge-case rules.
- Choose temperature for a task and explain what it changes.
- Build a small evaluation harness: a template, a model call, a scorer, and pass rates on a fixed test set.
- Decide whether a prompt change is a real improvement using the standard error of a pass rate.

**Why it matters.** The prompt is the cheapest lever you have. It changes behaviour in minutes, with no training run. But it is also the easiest place to fool yourself: you try a new wording on three examples, they look better, you ship, and quality drops on the inputs you never looked at. This lesson treats prompts as code with tests.

## 1. Intuition

**A prompt narrows a distribution.** A language model predicts the next token given everything before it (see [decoding](../15-llms/02-decoding.md)). "What do you think of this review?" is compatible with thousands of reasonable answers: a paragraph, a star rating, a one-word mood. A prompt that says "answer with one of: positive, negative, mixed, as JSON" is compatible with three. Fewer valid outputs means a higher chance of the one you can use.

**The three parts of a request.**

- The **system message** ([system prompt](../../glossary.md#system-prompt)) sets standing rules: role, scope, tone, output format. It is sent on every call, so it also costs tokens on every call.
- The **user message** carries the task and the input for this call.
- Some APIs let you **prefill** the start of the assistant's reply (for example `{`) so the model continues from there. Support varies by provider and model. Structured outputs (schema-constrained decoding, see [constrained decoding](../../glossary.md#constrained-decoding)) are the more robust way to force a format.

**What a good prompt specifies.**

1. **Role and audience.** "You are a careful analyst of customer reviews." A specific role biases the model toward text written by that kind of author. It does not add knowledge the model lacks.
2. **Task and allowed outputs.** The exact label set, the length, what to include and exclude.
3. **Format.** JSON keys, bullet count, delimiters. Say "output only the JSON" if code will parse it.
4. **Edge-case rules.** What to do with the inputs that are hard: "use mixed when the review praises one thing and criticizes another", "if the answer is not in the text, say you don't know".
5. **Delimiters around inputs.** Wrap untrusted input in tags like `<review>…</review>` so the model can tell data from instructions. This helps, but it does not stop [prompt injection](../../glossary.md#prompt-injection).
6. **Examples.** Two or three input/output pairs often teach a format faster than prose. That is the next lesson, [few-shot and chain-of-thought](02-few-shot-cot.md).

**Temperature is part of the prompt contract.** For extraction and classification you want the most probable answer every time, so use a low temperature. For brainstorming you want variety. A high temperature on a classification prompt does not make it smarter; it makes it noisier.

> [!IMPORTANT]
> **Prompt quality is measured, not judged by eye.** Every prompt change is a hypothesis. Keep a fixed test set with expected outputs, score each version automatically, and compare pass rates. Reading three outputs is not evidence.

## 2. Visualization

![Left: grouped bars for four prompt versions on 24 synthetic reviews. v0 vague scores 0 on all three checks. v1 adds a label set: labels valid 1.00, correct 0.58, but JSON parse 0. v2 asks for JSON but the model adds a preamble, so parse is still 0 and correct 0.58. v3 adds an edge-case rule and "output only the JSON": parse 1.00, valid 1.00, correct 0.92. Right: accuracy of prompt v3 versus temperature, mean of 10 seeds falling from 0.92 at temperature 0 to about 0.63 at 1.5, with a widening min-max band.](../../figures/prompt-engineering.png)

*Synthetic: 24 made-up reviews and a scripted stand-in model (ToyLLM) that follows instructions literally and guesses wherever the prompt is silent. The pass rates show the mechanism, not the quality of any real model.*

Read the left panel from left to right. Each revision fixes one specific failure. Adding the label set fixes the vocabulary ("good" becomes "positive"). Asking for JSON is not enough: the model wraps it in "Here is the analysis:", so a strict `json.loads` still fails. Saying "output only the JSON" fixes parsing, and defining "mixed" fixes the mixed reviews. The last two errors (22 of 24 correct) are reviews the model misreads ("Not bad at all" read as negative); no wording fixes those, which is a capability limit, not a prompt bug.

**Try it** (predict first, then run the script):

1. Delete "Output only the JSON, no other text." from v3. Predict the three bars for v3.
2. Remove the "mixed" rule but keep everything else in v3. Which reviews change label, and what does accuracy drop to?
3. Run v3 at temperature 1.0 with a different seed range. Predict whether the spread across seeds shrinks or grows as you add reviews to the test set.
4. Add three new reviews that use sarcasm ("Great, it broke on day one"). Predict accuracy before you look.

## 3. The math

### Symbols

| Symbol | Meaning | Shape / unit |
|---|---|---|
| $n$ | number of test cases | count |
| $k$ | number that pass a check | count |
| $\hat p$ | pass rate, $k / n$ | fraction |
| $\text{SE}(\hat p)$ | standard error of the pass rate | fraction |
| $z$ | difference in pass rates divided by its standard error | unitless |
| $T$ | sampling temperature | unitless |
| $s$ | system prompt length | tokens |
| $N$ | calls per day | count |

### How noisy is a pass rate?

Treat each test case as a pass/fail coin with true pass probability $p$. The estimate $\hat p = k/n$ has standard error

```math
\text{SE}(\hat p) = \sqrt{\frac{\hat p\,(1 - \hat p)}{n}}
```

To compare two prompt versions A and B, a rough check (treating them as independent) is

```math
z = \frac{\hat p_B - \hat p_A}{\sqrt{\text{SE}(\hat p_A)^2 + \text{SE}(\hat p_B)^2}}
```

and $|z|$ above about 2 suggests the difference is not just noise. Because both prompts run on the *same* test cases, a paired test (McNemar's test on the cases where they disagree) is tighter; the formula above is the quick version.

### Worked example: v1 versus v3

From the script: v1 gets $k = 14$ of $n = 24$ right, v3 gets 22 of 24.

1. $\hat p_1 = 14/24 = 0.5833$, $\hat p_3 = 22/24 = 0.9167$.
2. $\hat p_1(1 - \hat p_1) = 0.5833 \times 0.4167 = 0.2431$. Divide by 24: $0.010128$. Square root: $\text{SE}_1 = 0.1006$.
3. $\hat p_3(1 - \hat p_3) = 0.9167 \times 0.0833 = 0.0764$. Divide by 24: $0.003183$. Square root: $\text{SE}_3 = 0.0564$.
4. Variance of the difference: $0.010128 + 0.003183 = 0.013311$. Square root: $0.1154$.
5. $z = (0.9167 - 0.5833) / 0.1154 = 0.3333 / 0.1154 = 2.89$.

So the gain is larger than the noise. Now run the same arithmetic on a smaller change. If a new wording moves 22 of 24 to 23 of 24, the difference is $1/24 = 0.042$ while the standard error is about $0.05$, so $z < 1$: you cannot tell. **How many cases do you need?** For $\text{SE} \le 0.05$ at the worst case $\hat p = 0.5$: $0.5 \times 0.5 / n \le 0.0025$, so $n \ge 100$.

### Temperature, briefly

The model turns logits $z_i$ into probabilities with $p_i = e^{z_i / T} / \sum_j e^{z_j / T}$ (full treatment in [decoding](../15-llms/02-decoding.md)). As $T \to 0$ the top token gets all the mass (greedy). In the script, v3 scores 0.917 at $T = 0$, a mean of 0.779 at $T = 0.7$ (range 0.667 to 0.875 across 10 seeds), and 0.704 at $T = 1.0$. Same prompt, same model; only the sampling changed.

### System prompt cost

Every call re-sends the system prompt. With $s = 600$ tokens and $N = 200{,}000$ calls a day, that is $600 \times 200{,}000 = 120{,}000{,}000$ input tokens a day. At a synthetic price of USD 3 per million input tokens, that is $120 \times 3 = 360$ USD a day for the instructions alone. [Prefix caching](../../glossary.md#prefix-caching) bills a repeated, byte-identical prefix at a discount, which is one reason to keep the system prompt stable and put it first.

## 4. Implementation

### Build it from scratch

The script is a complete prompt-testing loop with four parts: templates, a model call, a scorer, and an evaluator. Start with the templates. A template is just a string with named slots, kept in one place so every version is reviewable and diffable.

```python
PROMPTS = {
    "v0 vague": "What do you think of this review?\n\n{review}",
    "v1 +labels": "Classify the sentiment of the review as one of: positive, negative, mixed.\n\n{review}",
    "v2 +JSON": ("Classify the sentiment of the review as one of: positive, negative, mixed.\n"
                 'Respond as JSON: {{"sentiment": "<label>"}}.\n\n<review>{review}</review>'),
    "v3 +rules": ("Classify the sentiment of the review as one of: positive, negative, mixed.\n"
                  "Use mixed when the review praises one thing and criticizes another.\n"
                  'Respond as JSON: {{"sentiment": "<label>"}}. Output only the JSON, no other text.\n\n'
                  "<review>{review}</review>"),
}
```

*Predict:* the doubled braces `{{` are there because of `str.format`. What would happen to v2 if you wrote single braces?

Next, the model. ToyLLM is a scripted stand-in so the harness runs offline. It scores the review with a word list, then follows only the instructions it finds in the prompt: it uses the label set if one is given, it uses "mixed" sensibly only if the rule is defined, and it adds a chatty preamble unless told "output only".

```python
        knows_mixed_rule = "praises one thing" in user          # edge case defined?
        scores = {"positive": pos - neg, "negative": neg - pos,
                  "mixed": (min(pos, neg) + 0.5) if knows_mixed_rule else (min(pos, neg) - 1.0)}
        label = self._pick(scores, temperature, user, review, seed)
        if "one of: positive, negative, mixed" not in user:      # no label set: free vocabulary
            label = {"positive": "good", "negative": "bad", "mixed": "it is mixed"}[label]
        if "JSON" not in user:                                   # no format: chatty prose
            return f"Overall the customer seems {label} about it."
        body = json.dumps({"sentiment": label})
        return body if "Output only the JSON" in user else f"Here is the analysis:\n{body}"
```

*Predict:* for "Great screen but the battery died fast." (one positive word, one negative word), what label does v1 produce, and why does v3 fix it?

`_pick` applies temperature exactly as in the decoding lesson: greedy at $T = 0$, otherwise a softmax of `scores / T` and a seeded random draw. All model access goes through one function, which is where a real client plugs in:

```python
def call_model(system, user, temperature=0.0, seed=0):
    """The one place a real client plugs in. Offline default: ToyLLM."""
    return TOY.generate(system, user, temperature, seed)
```

*Predict:* if you swap in a real model and run the same test set twice at temperature 0, should you expect identical pass rates? (Usually close, not always identical: batching and hardware can change floating-point results.)

The scorer separates "did it parse" from "was it right". Keeping the checks separate tells you *which* instruction failed.

```python
def score(output, gold):
    try:                                       # strict: what a downstream json.loads sees
        label, parsed = json.loads(output)["sentiment"], True
    except (json.JSONDecodeError, KeyError, TypeError):
        parsed = False                         # lenient: first allowed label word anywhere
        found = re.findall(r"\b(positive|negative|mixed)\b", output.lower())
        label = found[0] if found else None
    return {"parses": parsed, "valid_label": label in LABELS, "correct": label == gold}
```

*Predict:* v2 outputs `Here is the analysis:\n{"sentiment": "positive"}`. Which of the three checks pass?

The evaluator runs every case and averages each check. That is the whole harness.

```python
def evaluate(template, temperature=0.0, seed=0):
    rows = [score(call_model(SYSTEM, build_prompt(template, review=r), temperature, seed), g) for r, g in REVIEWS]
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
```

*Predict:* with 24 cases, what is the smallest change in accuracy this harness can show?

### Use the library

Replace the body of `call_model` with a real client. With the Anthropic Python SDK:

```python
import anthropic

client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY

def call_model(system, user, temperature=0.0, seed=0):
    msg = client.messages.create(
        model=MODEL_ID,                 # a current model id from the provider's docs
        max_tokens=100,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
    )
    return msg.content[0].text
```

Other providers' SDKs differ in field names (for example, a system message inside `messages`), but the harness does not change. For a hard format guarantee, use the provider's structured output or tool-calling feature with a JSON schema instead of asking politely. Evaluation frameworks (promptfoo, OpenAI Evals, LangSmith, and others) add test-case files, caching, and dashboards on top of the same loop.

### Run it

```bash
cd guide
uv run code/28-prompting-and-context/prompt_engineering.py
```

Key output:

```
prompt        parses   valid  correct
v0 vague        0.00    0.00     0.00
v1 +labels      0.00    1.00     0.58
v2 +JSON        0.00    1.00     0.58
v3 +rules       1.00    1.00     0.92
Is v3 really better than v1?  p1 = 0.583, p3 = 0.917, n = 24
  SE(p1) = 0.101, SE(p3) = 0.056, SE(diff) = 0.115, z = 2.89
  T = 0.0  mean 0.917  min 0.917  max 0.917
  T = 1.0  mean 0.704  min 0.583  max 0.792
```

Full script: [`code/28-prompting-and-context/prompt_engineering.py`](../../code/28-prompting-and-context/prompt_engineering.py).

## 5. Engineering

**A prompt workflow that holds up.**

1. Write the test set first: 30 to 200 real inputs with expected outputs, weighted toward hard and edge cases. Keep a held-out slice you don't tune on.
2. Write the simplest prompt that specifies labels and format. Run the harness.
3. Read the failures, group them, and fix one group per revision. Re-run everything each time.
4. Version prompts in git like code. Log the prompt version with every production call so you can trace a bad output to a prompt.
5. Re-run the suite when you change model versions. A prompt tuned on one model can regress on the next.

**Where prompts stop working.** If the model lacks the knowledge, use [retrieval](../16-rag/01-rag-pipeline.md). If it needs to act, use [tools](../19-agents/01-tool-use-and-agents.md). If you need a consistent style or format across millions of calls and prompting plateaus, consider [fine-tuning](../15-llms/03-adapting-llms.md). Prompting changes behaviour; it does not add facts.

**Format reliability.** Asking for JSON gets you JSON most of the time. "Most" breaks pipelines. Use schema-constrained structured outputs when code consumes the result, validate anyway, and decide what happens on a parse failure (retry once, then fall back).

**Prompt length.** Long system prompts cost tokens on every call and add latency to the first token. Contradictory rules ("be concise" and "cover every edge case") force the model to pick one. Audit long prompts for conflicts and dead rules.

**Portability.** Different models respond differently to the same wording. Plain language, explicit format, and XML-style delimiters travel best across providers. Measure on each model you ship.

**Security.** User text inside your prompt can contain instructions ("ignore the above"). Delimiters and a firm system message reduce but do not remove the risk. Treat model output as untrusted, and keep authority in code (see [Securing AI systems](../21-safety-security/01-ai-security.md)).

### Common mistakes

- Tuning on three examples you can see, then shipping. Use a test set and pass rates.
- Asking for a format but not saying "only", then parsing with `json.loads`.
- Leaving the label set open ("what's the sentiment?") and then string-matching labels.
- Classifying at a high temperature and calling the noise "creativity".
- Changing two things in one revision, so you can't tell which one helped.
- Expecting a role ("you are a world-class expert") to supply knowledge the model does not have.
- Putting secrets or access rules in the system prompt and assuming users can't extract them.

## 6. Knowledge check

<!-- quiz:prompt-engineering -->
**[Take the prompt engineering quiz](../../quizzes/prompt-engineering.md)**
<!-- /quiz -->

**Practice exercise.** A prompt passes 70 of 100 test cases. A new version passes 78 of 100. Compute the standard error of each pass rate, the $z$ value of the difference, and say whether you would ship on this evidence.

<details>
<summary>Solution</summary>

$\hat p_A = 0.70$: $\text{SE}_A = \sqrt{0.70 \times 0.30 / 100} = \sqrt{0.0021} = 0.0458$.
$\hat p_B = 0.78$: $\text{SE}_B = \sqrt{0.78 \times 0.22 / 100} = \sqrt{0.001716} = 0.0414$.
Standard error of the difference: $\sqrt{0.0021 + 0.001716} = \sqrt{0.003816} = 0.0618$.
$z = 0.08 / 0.0618 = 1.29$.

Below 2, so this could be noise. Before shipping, check the paired view: on how many cases did B fix A's failures versus break A's passes? If B fixed 10 and broke 2, McNemar's test is much more convincing than the unpaired $z$. Otherwise add cases.

</details>

**Implementation challenge.** Extend the script with a paired comparison: for v1 and v3, count cases where v1 failed and v3 passed ($b$), and v1 passed and v3 failed ($c$). Print McNemar's statistic $(b - c)^2 / (b + c)$ and compare it with the unpaired $z^2$.

<details>
<summary>Solution sketch</summary>

Run both prompts, keep per-case `correct` flags, then `b = sum(not a and b for a, b in zip(v1, v3))` and `c = sum(a and not b ...)`. Here v3 keeps all 14 of v1's correct cases and fixes 8 of the mixed reviews, so $b = 8$, $c = 0$, and the statistic is $64 / 8 = 8.0$, above the 3.84 cutoff for 5% significance. The unpaired $z^2 = 2.89^2 = 8.35$ happens to be similar here; the paired test pays off when the two prompts trade wins on different cases.

</details>

## Summary

- A prompt is a specification. State the role, task, label set, format, and edge-case rules; whatever you leave out, the model guesses.
- The system message holds standing rules and is re-sent on every call; prefill and structured outputs control format.
- Temperature is part of the contract: low for extraction and classification, higher only when you want variety.
- Measure every prompt change on a fixed test set with separate checks (parses, valid, correct) and compare pass rates with their standard errors.
- Prompting changes behaviour, not knowledge: retrieval, tools, and fine-tuning cover what prompts can't.

**Next:** [Few-shot and chain-of-thought](02-few-shot-cot.md)

**Related:** [Decoding](../15-llms/02-decoding.md) · [Context engineering](03-context-engineering.md) · [LLM evaluation](../17-llm-evaluation/01-llm-evaluation.md) · [Tool use and agents](../19-agents/01-tool-use-and-agents.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Prompt Engineering: Techniques & Patterns](https://aiengineeringfromscratch.com/lesson?path=phases/11-llm-engineering/01-prompt-engineering).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Walk me through how you would improve a prompt that gives inconsistent results.</strong></summary>

First I'd make "inconsistent" measurable. I collect 50 to 100 real inputs with expected outputs, write automatic checks (does it parse, is the label in the allowed set, is it correct), and run the current prompt to get a baseline. Then I read the failures and group them. Usually the groups map to missing specification: no fixed label set, no "output only the JSON", no rule for the ambiguous case, or temperature too high for a classification task. I fix one group per revision and re-run the whole suite, so I know which change helped and nothing regressed. I version prompts in git and log the version with each call. If the remaining failures are knowledge gaps rather than instruction gaps, the fix is retrieval or a stronger model, not more wording.

</details>

<details>
<summary><strong>When would you stop prompt engineering and fine-tune instead?</strong></summary>

When prompting has plateaued on a measured test set and the remaining gap is about consistent behaviour rather than missing facts. Examples: a house style the model keeps drifting from, a narrow output format at very high volume, or a classification task with many fine-grained labels where few-shot examples eat too much context. Fine-tuning can also cut cost, because a tuned small model with a short prompt may match a large model with a long one. I would not fine-tune to add knowledge that changes; that is retrieval's job. And I'd try structured outputs, better examples, and a stronger model before training, because a training run brings data collection, evaluation, and a model to maintain.

</details>

<details>
<summary><strong>Your JSON-extraction prompt works in testing but 3 percent of production calls fail to parse. What do you check?</strong></summary>

I pull the failing outputs and classify them. Common causes: the model adds a preamble ("Here is the JSON:") or wraps it in code fences; the output is truncated because max_tokens is too small for long inputs; the input itself contains quotes or braces that confuse the model; or temperature is above zero so rare phrasings appear. Fixes in order: use the provider's structured output or tool calling with a schema so the decoder cannot emit invalid JSON; raise max_tokens and log the stop reason; say "output only the JSON"; set temperature to zero. Then I validate every response against the schema in code, retry once on failure, and add the failing inputs to the test set.

</details>

<details>
<summary><strong>Two prompts score 0.88 and 0.92 on a 50-case test set. Is the second one better?</strong></summary>

Probably not provably. The standard error of a pass rate is the square root of p times one minus p over n. For 0.88 that's the square root of 0.88 times 0.12 over 50, about 0.046; for 0.92 it's about 0.038. The difference of 0.04 over a combined error of about 0.06 gives z near 0.67, well inside noise. A paired view helps because both ran on the same cases: if the new prompt fixed 2 failures and broke none, that's suggestive; if it fixed 4 and broke 2, it's a wash. To detect a 4-point gain reliably I'd need a few hundred cases, so I'd grow the test set, especially with hard cases, before deciding.

</details>
