---
name: Embedding model
tags: [self-supervised, nlp, low-latency]
lessons: [word-embeddings, rag-pipeline]
labs: []
---

# Embedding model

## Problem type

Representation learning for similarity: semantic search, clustering, deduplication, retrieval for RAG.

## Input

Text (or images, code) up to the model's maximum length.

## Output

A fixed-size dense vector, often normalized to unit length.

## Mental model

Map items to points in space so that similar meanings land close together; compare by cosine similarity.

## Core objective

Contrastive learning: pull matching pairs (query and relevant passage) together and push non-matching pairs apart, e.g. an InfoNCE loss over in-batch negatives.

## Training process

Pretrained encoder, then contrastive training on large sets of paired data; optional domain fine-tuning.

## Preprocessing

Chunk long documents; use the model's tokenizer and any required instruction prefixes (some models expect "query:" / "passage:").

## Assumptions

The model's notion of similarity matches your task; the domain is covered.

## Key hyperparameters

Model choice, dimension, max input length, normalization, chunk size.

## Good use cases

Dense retrieval in RAG, semantic search, clustering texts, recommendation, near-duplicate detection.

## Poor use cases

Exact matches on IDs and rare terms (pair with sparse search); fine-grained reasoning (use a reranker or LLM).

## Strengths

Captures paraphrases and meaning; fast at query time with ANN indexes; reusable.

## Weaknesses

Domain mismatch; weak on exact terms and numbers; vectors from different models are incompatible.

## Computational cost

One encoder forward pass per item at index time and per query at search time; storage $n \times d$ floats.

## Evaluation metrics

Recall@k, MRR, nDCG on a labeled retrieval set from your domain.

## Failure modes

Mixing model versions between queries and documents; stale indexes; truncation of long inputs.

## Minimal implementation

```python
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("all-MiniLM-L6-v2")
V = enc.encode(texts, normalize_embeddings=True)       # [n, 384]; cosine = dot product
```

## Compared with neighbors

- **TF-IDF / BM25:** exact-term matching, no training, complementary (use hybrid search).
- **Reranker:** slower, more accurate scoring of a short candidate list.

## Learn more

[Word embeddings](../lessons/09-classical-nlp/02-word-embeddings.md) · [The RAG pipeline](../lessons/16-rag/01-rag-pipeline.md)
