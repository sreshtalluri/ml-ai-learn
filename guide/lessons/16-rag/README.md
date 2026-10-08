---
title: Retrieval-augmented generation
summary: "Ground LLM answers in your documents: ingest, chunk, embed, retrieve, rerank, generate, and evaluate."
skill: llms
---

# Module 16: Retrieval-augmented generation

RAG gives a model the right context at query time instead of hoping the answer is stored in its weights. Most RAG failures are retrieval failures, so this module treats retrieval as a search system with its own metrics.

| Step | What happens | Failure mode |
|---|---|---|
| Ingest | parse and clean sources | missing or corrupted text |
| Chunk | split documents into retrievable units | too small loses context; too large dilutes relevance |
| Embed and index | store vectors plus metadata | wrong model, stale index |
| Retrieve | find relevant chunks | poor query or low recall |
| Rerank | score candidates more precisely | extra latency |
| Generate | answer using the supplied context | hallucination or weak grounding |
| Evaluate | measure retrieval and answer quality | only judging fluency |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [The RAG pipeline](01-rag-pipeline.md) | build each stage, compute retrieval metrics, and decide between RAG and fine-tuning |
