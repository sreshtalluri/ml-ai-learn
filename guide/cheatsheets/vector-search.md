---
title: Vector search
summary: Similarity metrics, exact vs approximate nearest-neighbor indexes, their knobs and memory, hybrid search with RRF, and filtering pitfalls.
---

# Vector search

**Metrics:** dot $q \cdot x$, cosine $\frac{q \cdot x}{\lVert q\rVert\lVert x\rVert}$, L2 $\lVert q - x\rVert$. For unit vectors all three rank the same ($\lVert q - x\rVert^2 = 2 - 2\cos$), so normalize and use dot. Otherwise use the metric the model was trained with.

| Index | How it works | Knobs | Cost / memory | Use when |
|---|---|---|---|---|
| Exact (flat) | compare with every vector | none | $O(N \cdot d)$ per query; $4Nd$ bytes in float32 | up to a few hundred thousand vectors, small filtered subsets, ground truth |
| IVF | k-means cells, scan nearest $n_{\text{probe}}$ | $n_{\text{list}}$ (start near $\sqrt{N}$), $n_{\text{probe}}$ | $(n_{\text{list}} + \frac{n_{\text{probe}}}{n_{\text{list}}}N)\,d$ | large $N$, cheap build, works with PQ |
| HNSW | layered neighbor graph, greedy descent | $M$, efConstruction, efSearch ($\ge k$) | vectors + about $2M \times 4$ bytes/vector of links | fits in RAM, want high recall at low latency |
| PQ | $m$ subvectors, each a 1-byte codebook ID | $m$, codebook size (256) | $m$ bytes/vector (e.g. 3 KB to 96 B, $32\times$) | memory-bound; re-score top candidates exactly |

**Index recall@k** $= |\text{ANN top-}k \cap \text{exact top-}k| / k$. Measure on real queries; tune for a recall target at a latency budget.

**Reciprocal rank fusion:** $\text{RRF}(d) = \sum_j \frac{1}{60 + \text{rank}_j(d)}$. Dense A,B,C,D + BM25 E,F,A,B gives A, B, E, F, C, D.

> [!WARNING]
> **Post-filtering** a top-$k$ list keeps about $k \times$ selectivity results (0.2 of 10 at 2%). Filter inside the search, partition by tenant, or brute-force small subsets. Enforce access control in retrieval, never in the LLM.

**Operations:** store model name and version with every vector; re-embed everything when the model changes (blue-green); HNSW deletes are tombstones until rebuild; IVF centroids go stale after heavy inserts.

**Where to run it:** in-memory (NumPy, FAISS) for small or batch; pgvector when data and metadata already live in Postgres; a dedicated vector database or search engine for scale, hybrid search, and multi-tenancy.

Lessons: [Vector search](../lessons/16-rag/02-vector-search.md) · [The RAG pipeline](../lessons/16-rag/01-rag-pipeline.md)
