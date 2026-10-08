---
title: LLM evaluation
summary: Measure LLM systems layer by layer: task quality, retrieval, grounding, safety, and operations.
skill: llms
---

# Module 17: LLM evaluation

You cannot improve what you cannot measure, and LLM outputs are hard to measure. This module builds an evaluation stack from exact-match metrics to rubric grading, retrieval metrics, groundedness checks, LLM-as-judge (and its limits), and operational metrics like latency and cost.

| Layer | Example metrics and checks |
|---|---|
| Task quality | accuracy, F1, exact match, rubric score |
| Retrieval | recall@k, MRR, nDCG, context relevance |
| Grounding | claim support, citation correctness, faithfulness |
| Safety | harmful-content rate, prompt-injection resistance, privacy |
| Operations | p50/p95 latency, tokens/sec, error rate, cost per request |
| Human factors | usefulness, calibration, user correction rate |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Evaluating LLM systems](01-llm-evaluation.md) | design a test set, compute retrieval and answer metrics, and run a sound comparison between two system versions |
