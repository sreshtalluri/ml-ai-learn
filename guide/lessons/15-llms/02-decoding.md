---
title: Decoding
summary: Turn next-token distributions into text with greedy decoding, temperature, top-k, top-p, and beam search; compute their effects by hand; and choose settings for a task.
skill: llms
minutes: 30
prerequisites: [tokenization-and-pretraining, logistic-regression]
related: [tokenization-and-pretraining, adapting-llms, llm-evaluation]
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

```python
import numpy as np

def sample_next(logits, temperature=1.0, top_k=None, top_p=None, rng=np.random.default_rng()):
    z = logits / max(temperature, 1e-6)
    p = np.exp(z - z.max()); p /= p.sum()
    order = np.argsort(-p)
    keep = np.ones_like(p, dtype=bool)
    if top_k is not None:
        keep[order[top_k:]] = False
    if top_p is not None:
        cum = np.cumsum(p[order])
        keep[order[np.searchsorted(cum, top_p) + 1:]] = False
    p = np.where(keep, p, 0); p /= p.sum()
    return rng.choice(len(p), p=p)
```

With Hugging Face `transformers`:

```python
out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
greedy = model.generate(**inputs, max_new_tokens=50, do_sample=False)        # deterministic
beams = model.generate(**inputs, max_new_tokens=50, num_beams=4, early_stopping=True)
```

Runnable script (temperature, nucleus, and loss figures): [`code/15-llms/llms.py`](../../code/15-llms/llms.py).

## 5. Engineering

| Task | Typical settings |
|---|---|
| Extraction, classification, structured output (JSON) | greedy or $T \approx 0$; constrain the format with schemas |
| Factual Q&A with retrieved context | low temperature (0 to 0.3) |
| Chat and general writing | $T \approx 0.7$, top-p $\approx 0.9$ |
| Brainstorming, creative writing | higher temperature, maybe 1.0+ |
| Translation and short constrained outputs | beam search can help |

**Reproducibility.** Sampling is random; fix seeds where the API allows, or use greedy decoding for tests. Even greedy outputs can vary slightly across hardware and batching due to floating-point effects.

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
