---
title: LLM evaluation
summary: Metrics for every layer of an LLM system and how to compare versions soundly.
---

# LLM evaluation

| Layer | Metrics |
|---|---|
| Task quality | exact match, token F1, accuracy, rubric score, human rating |
| Retrieval | recall@k, precision@k, MRR, nDCG, context relevance |
| Grounding | claim support rate, citation correctness, faithfulness |
| Safety | harmful-content rate, prompt-injection success rate, data-leak tests |
| Operations | p50/p95 latency, time to first token, tokens/sec, error rate, cost per request, cache hit rate |
| Human factors | usefulness, user correction rate, escalation rate |

**Formulas:** recall@k $= \frac{\text{relevant in top }k}{\text{total relevant}}$; MRR $= \text{mean}(1/\text{rank of first relevant})$; DCG@k $= \sum_i \frac{\text{rel}_i}{\log_2(i+1)}$, nDCG $=$ DCG/IDCG.

**Worked example:** relevance $[0,1,0,1,0]$, 2 relevant: recall@3 = 0.5, RR = 0.5, nDCG@5 = 0.651.

**LLM-as-judge biases:** position, verbosity, self-preference, inconsistency. Use rubrics, references, randomized order, and human agreement checks.

**Comparing versions:** same cases, per-case diffs (fixed versus broken), paired bootstrap interval, plus latency and cost. Small eval sets give wide intervals.

**Every case should record:** input, expected answer/rubric, expected sources, retrieved context, output, scores, failure category, latency, tokens, cost, and prompt/model/index versions.

Lessons: [Evaluating LLM systems](../lessons/17-llm-evaluation/01-llm-evaluation.md) · [Classification metrics](../lessons/04-classification/02-classification-metrics.md)
