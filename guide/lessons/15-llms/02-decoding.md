---
title: Decoding
summary: Turn next-token distributions into text with greedy decoding, temperature, top-k, top-p, and beam search; compute their effects by hand; and choose settings for a task.
skill: llms
minutes: 30
prerequisites: [tokenization-and-pretraining, logistic-regression]
related: [tokenization-and-pretraining, adapting-llms, llm-evaluation]
adapted_from: [phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding, phases/11-llm-engineering/03-structured-outputs]
video: { id: decoding-explained, minutes: 3 }
---

# Decoding

> **Mental model.** At each step the model hands you a probability for every possible next token. Decoding is the rule for picking one. Always take the top token and you get predictable, sometimes repetitive text; sample from the distribution and you get variety, sometimes nonsense. The knobs control where you land between the two.

**You will learn to**
- Generate text autoregressively and explain why each step depends on the previous ones.
- Apply temperature to logits and compute the resulting probabilities.
- Filter candidates with top-k and top-p (nucleus) and renormalize.
- Explain greedy decoding and beam search and their trade-offs.
- Choose decoding settings for extraction, chat, and creative tasks.

**Why it matters.** The same model can give crisp, reliable JSON or rambling prose depending on decoding settings. Decoding also explains some failure modes (repetition loops, inconsistent answers between runs) and how to make outputs reproducible.

## 1. Intuition

Generation is a loop: feed the prompt, get a distribution over the next token, pick one, append it, repeat until an end token or a length limit.

- **Greedy:** always pick the most likely token. Deterministic, good for short factual or structured outputs, prone to repetition in long text.
- **Temperature:** divide the logits by $T$ before softmax. $T < 1$ sharpens the distribution (more confident), $T > 1$ flattens it (more random). $T \to 0$ approaches greedy.
- **Top-k:** keep only the $k$ most likely tokens, renormalize, sample.
- **Top-p (nucleus):** keep the smallest set of tokens whose probabilities add up to at least $p$, renormalize, sample. The candidate set adapts: small when the model is confident, large when it isn't.
- **Beam search:** keep the $B$ highest-scoring partial sequences at each step instead of one. Useful for constrained tasks like translation; tends to produce bland text for open-ended generation.

## 2. Visualization

<!-- lab:decoding -->
![Left: next-token probabilities after "The capital of France is" at temperatures 0.5, 1.0, and 1.5; lower temperature concentrates mass on "Paris". Middle: tokens sorted by probability with a cumulative curve; top-p 0.9 keeps two tokens. Right: per-token loss −ln p versus the probability assigned to the true token.](../../figures/decoding.png)

*Synthetic logits, illustrative only. At $T = 0.5$, "Paris" gets 0.994; at $T = 1.5$ it drops to 0.630 and the entropy rises from 0.06 to 2.02 bits.*

*Interactive version: adjust temperature, top-k, and top-p on the same logits, then sample and count outcomes. [Open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/decoding/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Predict how many tokens top-p = 0.9 keeps, then set it and count.
2. Set T = 0.1 and sample 20 times. Then T = 1.8. What happens to the variety and to the chance of a wrong city?

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $z_i$ | logit for token $i$ |
| $T$ | temperature, $T > 0$ |
| $p_i$ | probability of token $i$ |
| $k$ | top-k cutoff |
| $p$ | top-p cumulative-probability threshold |
| $B$ | beam width |

### Temperature

```math
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
```

### Top-k and top-p

Let $S$ be the kept set (the $k$ largest $p_i$, or the smallest set with $\sum_{i \in S} p_i \ge p$). Then

```math
p'_i = \begin{cases} p_i / \sum_{j \in S} p_j & i \in S \\ 0 & \text{otherwise} \end{cases}
```

### Beam search score

Beam search ranks sequences by total log-probability $\sum_k \log P(t_k \mid t_{<k})$, usually divided by a length penalty so longer sequences are not unfairly punished.

### Worked example 1: temperature

Logits $[2, 1, 0]$.

- **$T = 1$:** $e^2 = 7.389$, $e^1 = 2.718$, $e^0 = 1$; sum $11.107$; probabilities $[0.665, 0.245, 0.090]$.
- **$T = 0.5$:** logits become $[4, 2, 0]$; exponentials $[54.60, 7.389, 1]$; sum $62.99$; probabilities $[0.867, 0.117, 0.016]$.
- **$T = 2$:** logits become $[1, 0.5, 0]$; exponentials $[2.718, 1.649, 1]$; sum $5.367$; probabilities $[0.506, 0.307, 0.186]$.

Ranking never changes; only how peaked the distribution is.

### Worked example 2: top-k and top-p

Probabilities $[0.5, 0.3, 0.15, 0.05]$.

- **Top-k with $k = 2$:** keep $[0.5, 0.3]$, sum $0.8$, renormalized $[0.625, 0.375, 0, 0]$.
- **Top-p with $p = 0.9$:** cumulative $0.5, 0.8, 0.95$, so the first three are needed to reach 0.9. Renormalized $[0.5, 0.3, 0.15]/0.95 = [0.526, 0.316, 0.158, 0]$.
- **Top-p with $p = 0.8$:** the first two reach exactly 0.8, giving the same result as top-k = 2.

## 4. Implementation

The full script is [`code/15-llms/decoding_scratch.py`](../../code/15-llms/decoding_scratch.py) (NumPy only). It reproduces both worked examples, samples the lab's "capital of France" logits, runs greedy and beam search on a toy model, and ends with constrained decoding: the technique behind "JSON mode" and structured outputs.

### Build it from scratch

**Step 1: temperature.** Divide the logits by $T$, then softmax. Subtracting the max first keeps `exp` from overflowing and changes nothing mathematically. It also lets a logit of $-\infty$ (used in step 5) become a probability of exactly 0.

```python
def softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    e = np.exp(z - z.max())                       # -inf logits become exactly 0
    return e / e.sum()
```

*Predict:* what does `softmax([2, 1, 0], T=100)` look like? And `T=0.01`?

**Step 2: top-k and top-p.** Both filters zero out the tail and renormalize. Top-k keeps a fixed count. Top-p sorts, takes the cumulative sum, and keeps the shortest prefix whose mass reaches the threshold, so the count adapts to how confident the model is.

```python
def top_k(p, k):
    keep = np.zeros_like(p, dtype=bool)
    keep[np.argsort(-p)[:k]] = True
    q = np.where(keep, p, 0.0)
    return q / q.sum()                            # renormalize what is left

def top_p(p, threshold):
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    n_keep = np.searchsorted(cum, threshold - 1e-12) + 1   # smallest prefix with mass >= threshold
    q = np.zeros_like(p)
    q[order[:n_keep]] = p[order[:n_keep]]
    return q / q.sum()
```

The `- 1e-12` handles worked example 2's edge case: with $p = 0.8$, the cumulative sum $0.5 + 0.3$ is $0.8$ in exact arithmetic, but floating point may store it just below or above $0.8$.

*Predict:* if the model puts 0.95 on one token, how many tokens does top-p 0.9 keep? How many does top-k 50 keep?

**Step 3: one decoding step.** Order matters: temperature first, then filters, then sample. $T = 0$ is special-cased as greedy because dividing by zero is undefined.

```python
def sample_next(logits, rng, T=1.0, k=None, p=None):
    if T == 0:
        return int(np.argmax(logits))             # greedy: temperature -> 0
    probs = softmax(logits, T)
    if k is not None:
        probs = top_k(probs, k)
    if p is not None:
        probs = top_p(probs, p)
    return int(rng.choice(len(probs), p=probs))
```

*Predict:* at $T = 1.5$ the tail grows. Does adding top-p 0.9 raise or lower the share of "Paris" among 1,000 samples?

**Step 4: beam search.** Greedy commits to the best token at each step. Beam search keeps the $B$ best partial sequences by total log-probability, expands each, and prunes back to $B$. In the toy model (synthetic probabilities), "the" beats "a" at step one (0.5 vs 0.4), but "a cat" is the most likely full sentence.

```python
def beam_search(model, B, start="<s>"):
    beams = [([start], 0.0)]                      # (sequence, total log-probability)
    while any(s[-1] != "</s>" for s, _ in beams):
        cand = []
        for seq, lp in beams:
            if seq[-1] == "</s>":
                cand.append((seq, lp)); continue  # finished beams carry over
            for w, pr in model[seq[-1]].items():
                cand.append((seq + [w], lp + math.log(pr)))
        beams = sorted(cand, key=lambda c: -c[1])[:B]   # keep the B best partial sequences
    return beams[0]
```

Greedy finds "the cat" with $P = 0.5 \times 0.4 = 0.20$. Beam search with $B = 2$ keeps "a" alive and finds "a cat" with $P = 0.4 \times 0.9 = 0.36$.

*Predict:* with $B = 1$, what does beam search return?

**Step 5: constrained decoding.** To guarantee valid output, edit the logits before choosing: every token that cannot continue a valid answer gets $-\infty$, so its probability is exactly 0. Here the valid answers are three labels and the check is a prefix test; real libraries compile a regex, JSON Schema, or grammar into the same "which tokens are allowed now?" function.

```python
def allowed_next(prefix, choices):
    nxt = {c[len(prefix)] for c in choices if c.startswith(prefix) and len(c) > len(prefix)}
    if prefix in choices:
        nxt.add("<eos>")
    return nxt

def constrained_generate(logit_fn, vocab, choices):
    out = ""
    while True:
        logits = logit_fn(out)
        ok = allowed_next(out, choices)
        masked = np.where([v in ok for v in vocab], logits, -np.inf)   # invalid tokens get -inf
        tok = vocab[int(np.argmax(masked))]                              # greedy over valid tokens only
        if tok == "<eos>":
            return out
        out += tok
```

The toy "chatty" model wants to write "the sentiment is positive." At step one only `p` and `n` are allowed; it leans to `p`, and after that every character is forced.

*Predict:* if the labels were "positive" and "positively", which extra token must `allowed_next("positive", ...)` allow?

### Use the library

Hugging Face `transformers` runs the same loop:

```python
out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
greedy = model.generate(**inputs, max_new_tokens=50, do_sample=False)   # deterministic
beams = model.generate(**inputs, max_new_tokens=50, num_beams=4, early_stopping=True)
```

What the library adds: batched generation, a KV cache so each step reuses earlier work, stopping strings, repetition penalties, and `LogitsProcessor` hooks where step 5's mask plugs in. Serving engines (vLLM, SGLang, llama.cpp) and libraries like Outlines and XGrammar compile a JSON Schema, regex, or grammar into token masks over the full vocabulary of 100k+ tokens. Hosted APIs expose the same idea as structured outputs with a JSON Schema.

### Run it

```bash
uv run code/15-llms/decoding_scratch.py
```

Key lines of output:

```text
T=0.5: [0.867 0.117 0.016]
top-p p=0.9: [0.526 0.316 0.158 0.   ]
T=0   top_p=None: Paris 100.0%, distinct tokens 1
T=1.0 top_p=None: Paris  85.3%, distinct tokens 10
T=1.5 top_p=None: Paris  61.1%, distinct tokens 10
T=1.5 top_p=0.9: Paris  66.3%, distinct tokens 6
greedy   : the cat  log-prob -1.609  P = 0.20
beam B=2 : a cat    log-prob -1.022  P = 0.36
unconstrained: 'the sentiment is positive.'
constrained  : 'positive'
```

The first two lines match worked examples 1 and 2. At $T = 1.5$, top-p cuts the four rarest tokens and gives some mass back to "Paris". The figure script for section 2 is [`code/15-llms/llms.py`](../../code/15-llms/llms.py).

## 5. Engineering

| Task | Typical settings |
|---|---|
| Extraction, classification, structured output (JSON) | greedy or $T \approx 0$; constrain the format with schemas |
| Factual Q&A with retrieved context | low temperature (0 to 0.3) |
| Chat and general writing | $T \approx 0.7$, top-p $\approx 0.9$ |
| Brainstorming, creative writing | higher temperature, maybe 1.0+ |
| Translation and short constrained outputs | beam search can help |

**Reproducibility.** Sampling is random; fix seeds where the API allows, or use greedy decoding for tests. Even greedy outputs can vary slightly across hardware and batching due to floating-point effects.

**Structured output.** For anything a program parses, prefer constrained decoding (a schema-backed structured-output mode, or a grammar in your serving engine) over "please return JSON" plus retries. A mask guarantees the output parses; it does not guarantee the values are right. Field order is part of the logic: the model writes fields in schema order, so a `reasoning` field placed after `answer` cannot influence the answer. Put reasoning first if you want it, and keep schemas small, since a mask that forces an unnatural format can lower quality.

**Repetition.** Long greedy outputs can loop; frequency and presence penalties or sampling help.

**Cost.** Each generated token is a forward pass (cheap with a KV cache, but sequential). Output tokens usually cost more than input tokens and dominate latency for long answers.

> [!IMPORTANT]
> Lowering temperature makes outputs more consistent, not more correct. If the model's most likely answer is wrong, greedy decoding returns that wrong answer every time.

> [!WARNING]
> **Failure modes.** High temperature producing incoherent or off-format text; low temperature producing repetitive loops; top-p near 1 with high temperature letting junk tokens through; beam search yielding generic, short responses in open-ended tasks.

### Common mistakes

- Setting temperature to 0 and expecting it to fix factual errors.
- Comparing two prompts with a single sampled run each.
- Forgetting a `max_new_tokens` limit.
- Treating schema-valid JSON as correct JSON: constrained decoding fixes the format, never the facts.

## 6. Knowledge check

<!-- quiz:decoding -->
**[Take the decoding quiz](../../quizzes/decoding.md)**
<!-- /quiz -->

**Practice exercise.** Logits $[\ln 6, \ln 3, \ln 1]$. Compute the probabilities at $T = 1$, then apply top-p with $p = 0.6$.

<details>
<summary>Solution</summary>

Exponentials $[6, 3, 1]$, sum 10: probabilities $[0.6, 0.3, 0.1]$. Top-p 0.6: the first token alone reaches 0.6, so only it is kept: $[1, 0, 0]$, which is greedy here.
</details>

**Implementation challenge.** Write a generation loop around `sample_next` for a small open model, generate 20 continuations at temperatures 0.2, 0.7, and 1.2, and measure distinct-bigram ratio (diversity) and how often the output stays on topic.

## Summary

- Generation is autoregressive: predict a distribution, choose a token, append, repeat.
- Temperature rescales logits: lower is sharper, higher is flatter; ranking is unchanged.
- Top-k keeps a fixed number of candidates; top-p keeps an adaptive set reaching probability $p$; both renormalize.
- Greedy and beam search are deterministic; sampling trades consistency for diversity.
- Decoding controls style and consistency, not truthfulness.

**Next:** [Adapting LLMs](03-adapting-llms.md)

**Related:** [Tokenization and pretraining](01-tokenization-and-pretraining.md) · [LLM evaluation](../17-llm-evaluation/01-llm-evaluation.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Structured Outputs & Constrained Decoding](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding), [Structured Outputs: JSON, Schema Validation, Constrained Decoding](https://aiengineeringfromscratch.com/lesson?path=phases/11-llm-engineering/03-structured-outputs).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>What does temperature do to the next-token distribution, mathematically?</strong></summary>

It divides the logits by $T$ before the softmax: $p_i = \exp(z_i/T)/\sum_j \exp(z_j/T)$. Dividing by $T < 1$ stretches the gaps between logits, so the distribution sharpens toward the top token; $T > 1$ shrinks the gaps and flattens it; $T \to 0$ approaches greedy decoding. The ranking of tokens never changes, only how peaked the distribution is. With logits $[2, 1, 0]$: at $T = 1$ the probabilities are $[0.665, 0.245, 0.090]$; at $T = 0.5$ the logits become $[4, 2, 0]$ and the probabilities $[0.867, 0.117, 0.016]$; at $T = 2$ they are $[0.506, 0.307, 0.186]$. Two consequences follow. Lowering temperature makes outputs more consistent, not more correct: if the top answer is wrong, $T = 0$ returns it every time. And high temperature lifts the long tail of junk tokens, which is why it is combined with top-p or top-k.

</details>

<details>
<summary><strong>Top-k or top-p sampling: what is the difference, and when would you prefer each?</strong></summary>

Top-k keeps a fixed number of candidates; top-p keeps the smallest set whose cumulative probability reaches $p$. Both renormalize the survivors and sample. With probabilities $[0.5, 0.3, 0.15, 0.05]$, top-k with $k = 2$ gives $[0.625, 0.375]$. Top-p with $p = 0.9$ needs three tokens, since the cumulative sum goes 0.5, 0.8, 0.95, giving $[0.526, 0.316, 0.158]$. Top-p adapts to the model's confidence: when one token has 0.97 of the mass it keeps only that token, and when the distribution is flat it keeps many. A fixed $k$ is too permissive in the first case and too restrictive in the second. That is why top-p around 0.9 with moderate temperature is the common default for chat. Top-k is still useful as a hard cap on candidates, often combined with top-p, and when you need predictable compute. For extraction or JSON, use neither: decode greedily.

</details>

<details>
<summary><strong>Your model's long answers fall into repetition loops. How do you diagnose and fix it?</strong></summary>

Repetition is a known failure of greedy or very low-temperature decoding: once a phrase repeats, the context makes repeating it again more likely, and the argmax locks into a cycle. First confirm the settings: is it running greedy or $T$ near 0 on a long-form task? Moving to $T \approx 0.7$ with top-p 0.9 often fixes it. Next, add frequency or presence penalties, which subtract from the logits of tokens already generated, and set `max_new_tokens` and proper stop sequences so a loop can't run forever. Then check the prompt format: a wrong chat template or a missing end-of-turn token is a common cause, because the model never sees its usual cue to stop. If loops persist with sane settings, it may be a model-quality issue, such as a small or over-fine-tuned model. Track a repetition metric like distinct n-gram ratio in your evaluation so the regression is visible.

</details>

<details>
<summary><strong>You need the model to return JSON that matches a schema on every request. How do you set up decoding?</strong></summary>

Use greedy or very low-temperature decoding, constrain the output to the schema, and still validate. Sampling adds variance you don't want for extraction, so set $T$ to 0 or close to it. The strongest tool is constrained decoding, offered as structured outputs or JSON mode in many APIs and as grammar-constrained generation in open-source servers. At each step it masks out any token that would make the output violate the JSON grammar or schema, setting those logits to $-\infty$ before the softmax, so syntactically invalid output becomes impossible. It guarantees format, not correctness: field values can still be wrong. So validate the parsed object with a schema library, retry once on failure with the error message, then fall back to a safe default or human review. Set `max_new_tokens` high enough that long objects aren't truncated mid-JSON, a common cause of "invalid JSON" errors. Beam search adds nothing here.

</details>
