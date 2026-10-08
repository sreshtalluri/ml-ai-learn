---
title: RAG pipeline
summary: Every stage of retrieval-augmented generation, its knobs, and its failure modes.
---

# RAG pipeline

| Stage | Knobs | Failure mode | Check |
|---|---|---|---|
| Ingest | parsers, cleaning, metadata, permissions | garbled tables and PDFs, missing text | spot-check extracted text |
| Chunk | size, overlap, structure-aware splitting | answer split across chunks; diluted chunks | recall@k by chunk setting |
| Embed and index | embedding model, sparse index, versioning | wrong model, stale index, version mismatch | index freshness, model version in traces |
| Retrieve | k, hybrid weighting, metadata filters | low recall, vocabulary mismatch, permission leaks | recall@k, MRR on a labeled set |
| Rerank | cross-encoder, candidate count | added latency; can't fix missing candidates | nDCG@k, latency |
| Construct context | ordering, source labels, token budget | too much or irrelevant context | context relevance |
| Generate | "answer only from sources, cite, say I don't know" | ignoring context, unsupported claims | groundedness, citation correctness |
| Evaluate | test set, judges, human review | judging fluency only | layer-by-layer metrics |

**Chunk count:** $\lceil (N - O)/(S - O)\rceil$ for $N$ words, size $S$, overlap $O$.

**Hybrid search:** sparse (exact terms, IDs) + dense (paraphrases), fused by weighted scores or reciprocal rank fusion.

**Security:** filter by permissions at retrieval time; treat retrieved text as untrusted (prompt injection).

**Debug order:** is the evidence retrieved? → is it in the final context? → is the answer grounded in it?

Lessons: [The RAG pipeline](../lessons/16-rag/01-rag-pipeline.md) · [Evaluating LLM systems](../lessons/17-llm-evaluation/01-llm-evaluation.md) · [Securing AI systems](../lessons/19-safety-security/01-ai-security.md)
