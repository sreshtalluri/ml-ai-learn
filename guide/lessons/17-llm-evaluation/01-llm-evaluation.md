---
title: Evaluating LLM systems
summary: Build an evaluation set, measure retrieval with recall@k, MRR, and nDCG, score answers for correctness and groundedness, use LLM judges carefully, track latency and cost, and compare system versions soundly.
skill: llms
minutes: 45
prerequisites: [rag-pipeline, classification-metrics, probability-and-statistics]
related: [rag-pipeline, reliability-cost-and-observability, ai-security]
---

# Evaluating LLM systems

> **Mental model.** An LLM system is a pipeline, so evaluate it like one: check each layer (retrieval, grounding, task quality, safety, operations) on a fixed set of realistic cases, and compare versions on the *same* cases. A demo that looks good is a sample size of one.

**You will learn to**
- Design an evaluation set with expected outputs, contexts, and failure categories.
- Compute recall@k, precision@k, MRR, and nDCG by hand.
- Score answers with exact match, token F1, rubrics, and groundedness checks.
- Use LLM-as-judge with its known biases in mind.
- Compare two system versions with paired statistics, alongside latency and cost.

**Why it matters.** Without evaluation, every prompt change is a guess and every model upgrade is a gamble. Evaluation turns "it seems better" into "it fixed 6 cases, broke 2, costs 47% more, and the difference is not yet statistically clear."

## 1. Intuition

Start with **test cases**: real or realistic user questions, each with what a good answer must contain, the documents that should be retrieved, and tags (topic, difficulty, adversarial). Fifty well-chosen cases beat five thousand random ones at first; grow the set from production failures.

Then measure by layer:

| Layer | Question | Example metrics |
|---|---|---|
| Retrieval | Did we fetch the right evidence? | recall@k, MRR, nDCG, context relevance |
| Grounding | Is every claim supported by the context? | claim support rate, citation correctness, faithfulness |
| Task quality | Is the answer right and useful? | exact match, F1, rubric score, human rating |
| Safety | Does it resist misuse? | harmful-content rate, prompt-injection success rate, privacy leaks |
| Operations | Can we afford and serve it? | p50/p95 latency, tokens/sec, error rate, cost per request, cache hit rate |

**Failure categorization** turns a score into a to-do list: retrieval misses mean fix search; unsupported claims mean fix grounding instructions or add verification; wrong formats mean fix schemas.

## 2. Visualization

![Three panels. Left: bootstrap distributions of the accuracy difference between two versions; the paired distribution is narrower than the unpaired one, and both overlap zero. Middle: failure categories for v1 and v2 over 40 cases; v2 cuts retrieval misses from 6 to 1 but adds an unnecessary refusal. Right: latency box plots; v2 is slower, and its cost per request is higher.](../../figures/llm-evaluation.png)

*Synthetic evaluation of two versions on 40 cases. v2 sends more retrieved context: accuracy 0.75 → 0.85, but p95 latency rises from 1.99 s to 2.74 s and cost per request from about 0.9 to 1.35 cents (illustrative prices).*

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $\text{rel}_i \in \{0, 1\}$ | whether the result at rank $i$ is relevant (graded relevance also works) |
| $R$ | total number of relevant items for the query |
| $\text{rank}_q$ | rank of the first relevant result for query $q$ |
| $Q$ | number of queries |

### Retrieval metrics

```math
\text{recall@}k = \frac{\sum_{i=1}^{k}\text{rel}_i}{R}, \qquad
\text{precision@}k = \frac{\sum_{i=1}^{k}\text{rel}_i}{k}, \qquad
\text{MRR} = \frac{1}{Q}\sum_{q=1}^{Q}\frac{1}{\text{rank}_q}
```

```math
\text{DCG@}k = \sum_{i=1}^{k}\frac{\text{rel}_i}{\log_2(i + 1)}, \qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
```

IDCG is the DCG of the ideal ordering. nDCG rewards putting relevant results near the top.

### Answer metrics

- **Exact match:** 1 if the normalized answer equals the reference.
- **Token F1:** precision and recall of answer tokens against reference tokens, then F1.
- **Groundedness:** fraction of the answer's claims supported by the provided context.

### Worked example 1: ranking metrics

Five results with relevance $[0, 1, 0, 1, 0]$; two relevant documents exist ($R = 2$).

- recall@1 $= 0/2 = 0$; recall@3 $= 1/2 = 0.5$; recall@5 $= 2/2 = 1.0$. precision@3 $= 1/3 = 0.33$.
- Reciprocal rank: first relevant at rank 2, so $1/2 = 0.5$.
- DCG@5 $= \frac{1}{\log_2 3} + \frac{1}{\log_2 5} = 0.631 + 0.431 = 1.062$.
- IDCG@5 (relevant at ranks 1 and 2) $= \frac{1}{\log_2 2} + \frac{1}{\log_2 3} = 1 + 0.631 = 1.631$.
- nDCG@5 $= 1.062 / 1.631 = 0.651$.

Over four queries whose first relevant results are at ranks 1, 2, none, and 3: MRR $= (1 + 0.5 + 0 + 0.333)/4 = 0.458$.

### Worked example 2: token F1

Reference "30 days from purchase" (4 tokens); answer "within 30 days" (3 tokens). Overlap: "30", "days" (2 tokens). Precision $= 2/3$, recall $= 2/4$, F1 $= \frac{2 \times 0.667 \times 0.5}{1.167} = 0.571$. Exact match is 0 even though the answer is correct, which is why exact match alone undercounts.

### Worked example 3: comparing versions properly

On the same 40 cases, v2 fixed 6 cases and broke 2, so accuracy rose from 0.75 to 0.85 (+0.10). A paired bootstrap (resampling *cases* and recomputing both versions on each resample) gives a 95% interval of $[-0.025, +0.225]$. An unpaired bootstrap that ignores the pairing gives the wider $[-0.075, +0.275]$. Both include 0: with 40 cases, a 10-point gain is promising but not conclusive. Add cases (especially in the categories that changed) before declaring victory, and weigh the 47% cost increase.

## 4. Implementation

```python
import numpy as np

def recall_at_k(rels, k, total_relevant):
    return sum(rels[:k]) / total_relevant

def ndcg_at_k(rels, k):
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels[:k]))
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(sorted(rels, reverse=True)[:k]))
    return dcg / idcg if idcg else 0.0

def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """95% CI for mean(b) - mean(a), resampling the same case indices for both."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = [b[idx].mean() - a[idx].mean() for idx in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return np.percentile(diffs, [2.5, 97.5])
```

An evaluation record should hold, per case: input, expected answer or rubric, expected sources, retrieved contexts, generated answer, per-metric scores, failure category, latency, token counts, cost, and the versions of prompt, model, and index.

Runnable scripts: [`code/17-llm-evaluation/eval.py`](../../code/17-llm-evaluation/eval.py) (version comparison) and [`code/16-rag/rag_eval.py`](../../code/16-rag/rag_eval.py) (ranking metrics).

## 5. Engineering

**LLM-as-judge.** Useful for scaling rubric grading, but judges have known biases: position bias (preferring the first answer shown), verbosity bias (preferring longer answers), self-preference (favoring their own model family), and inconsistency across runs. Mitigate with clear rubrics, randomized order, reference answers, pairwise comparisons, and regular agreement checks against human labels.

**Human evaluation** remains the ground truth for usefulness and tone; sample regularly and measure agreement between raters.

**Offline versus online.** Offline evals gate releases. Online signals (thumbs up or down, user corrections, task completion, escalation rate) and A/B tests measure real impact. Feed production failures back into the offline set.

**Operational metrics matter as much as quality:** report p50 and p95 latency, tokens per second, error and timeout rates, cost per request, and cache hit rate for every version.

**Safety evaluation:** include adversarial cases such as prompt injection in retrieved documents, attempts to extract system prompts or other users' data, and requests for disallowed actions.

> [!WARNING]
> **Failure modes.** Evaluating only fluency; tiny or unrepresentative test sets; overfitting prompts to the eval set; LLM judges grading their own outputs; ignoring cost and latency; comparing versions on different cases.

### Common mistakes

- Reporting a single aggregate score without failure categories.
- Treating a 2-point change on 50 cases as meaningful.
- Using only exact match for free-form answers.
- Letting the eval set go stale as the product changes.

## 6. Knowledge check

<!-- quiz:llm-evaluation -->
**[Take the LLM evaluation quiz](../../quizzes/llm-evaluation.md)**
<!-- /quiz -->

**Practice exercise.** Results have relevance $[1, 0, 1, 0, 0]$ with $R = 3$. Compute recall@3, precision@3, reciprocal rank, and nDCG@5.

<details>
<summary>Solution</summary>

recall@3 $= 2/3 = 0.667$; precision@3 $= 2/3 = 0.667$; reciprocal rank $= 1$.
DCG@5 $= 1/\log_2 2 + 1/\log_2 4 = 1 + 0.5 = 1.5$. With $R = 3$ the ideal ordering puts three relevant items first, but only two were retrieved; using the retrieved list's ideal ordering $[1, 1, 0, 0, 0]$ gives IDCG $= 1 + 0.631 = 1.631$ and nDCG $= 0.920$. (If IDCG is computed from all 3 relevant documents, IDCG $= 1 + 0.631 + 0.5 = 2.131$ and nDCG $= 0.704$. State which convention you use.)
</details>

**Implementation challenge.** Build a 30-case evaluation set for the RAG script's documents. For each case store the question, the expected answer, and the expected source. Score retrieval recall@3, answer token F1 against your expected answers (using any LLM or a template answer), and tag every failure with a category.

## Summary

- Evaluate each layer: retrieval, grounding, task quality, safety, and operations.
- recall@k and precision@k count hits; MRR rewards the first hit; nDCG rewards good ordering.
- Use token F1 and rubrics for free-form answers; check claims against context for groundedness.
- LLM judges scale grading but are biased; calibrate them against humans.
- Compare versions on the same cases with paired statistics, and always report latency and cost.

**Next:** [Production architecture](../18-production-ai/01-production-architecture.md)

**Related:** [The RAG pipeline](../16-rag/01-rag-pipeline.md) · [Classification metrics](../04-classification/02-classification-metrics.md) · [Securing AI systems](../19-safety-security/01-ai-security.md)
