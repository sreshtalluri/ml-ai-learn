---
name: Reranker
tags: [supervised, nlp]
lessons: [rag-pipeline, llm-evaluation]
labs: []
---

# Reranker (cross-encoder)

## Problem type

Relevance scoring of (query, document) pairs to reorder retrieval candidates.

## Input

A query and a candidate passage, processed together.

## Output

A relevance score per pair.

## Mental model

The retriever quickly finds a few dozen plausible candidates; the reranker reads each one alongside the question, carefully, and puts the best on top.

## Core objective

Supervised ranking loss (pointwise, pairwise, or listwise) on labeled query-passage relevance data.

## Training process

Fine-tune a transformer encoder on relevance labels, often with hard negatives mined from a first-stage retriever.

## Preprocessing

Truncate query plus passage to the context length; keep passages chunk-sized.

## Assumptions

The first stage retrieves the relevant item somewhere in the candidate list.

## Key hyperparameters

Number of candidates to rerank (e.g. 20 to 100), model size, max length.

## Good use cases

Improving precision at the top of RAG and search results.

## Poor use cases

First-stage retrieval over millions of documents (too slow); latency budgets of a few milliseconds.

## Strengths

Much more accurate than bi-encoder similarity because query and passage interact through attention.

## Weaknesses

Cost grows linearly with the number of candidates; adds latency; can't fix a first stage that missed the evidence.

## Computational cost

One forward pass per candidate per query.

## Evaluation metrics

nDCG@k, MRR, precision@k; end-to-end answer quality and added latency.

## Failure modes

Reranking a candidate list that lacks the answer; truncation hiding the relevant sentence; domain mismatch.

## Minimal implementation

```python
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = ce.predict([(query, c) for c in candidates])
ranked = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
```

## Compared with neighbors

- **Embedding model (bi-encoder):** fast, independent encoding, less precise.
- **LLM judge:** more flexible, much more expensive.

## Learn more

[The RAG pipeline](../lessons/16-rag/01-rag-pipeline.md)
