---
title: Vector search
summary: Pick a similarity metric and an approximate nearest-neighbor index (IVF, HNSW, product quantization), trade recall against latency and memory, and combine BM25 with dense retrieval using reciprocal rank fusion.
skill: llms
minutes: 40
prerequisites: [rag-pipeline, word-embeddings, k-nearest-neighbors]
related: [k-means, text-to-vectors, production-architecture]
---

# Vector search

**Mental model.** Vector search is a librarian who has shelved every document by meaning. Checking every book against your question is exact but slow. A good librarian walks to the two or three most promising shelves and checks only those books. That is fast, and it is usually right, but sometimes the best book was shelved one aisle over. Every vector index is a different way of deciding which shelves to walk to.

**You will learn to**
- Choose between cosine similarity, dot product, and Euclidean (L2) distance, and explain what normalization does to each.
- Compute the cost of exact k-nearest-neighbor search and of an IVF index.
- Explain how IVF, HNSW, and product quantization work and which knobs trade recall for speed or memory.
- Measure recall@k of an approximate index against brute force.
- Combine BM25 and dense results with reciprocal rank fusion, and avoid metadata-filtering pitfalls.
- Plan index updates, embedding-model upgrades, and the choice between a vector database, pgvector, and an in-memory index.

**Why it matters.** In the [RAG pipeline](01-rag-pipeline.md), retrieval sets the ceiling on answer quality. Behind "retrieve top-$k$" sits a search index with its own accuracy, latency, and memory budget. When a RAG system misses an obvious document, the cause is often an index setting (`nprobe`, `efSearch`), a filter applied in the wrong place, or a query embedded with a different model than the documents. These are the questions interviewers ask when a role includes search or RAG.

## 1. Intuition

**Embeddings as points.** An [embedding](../../glossary.md#embedding) model turns each chunk of text into a vector of a few hundred to a few thousand numbers. Texts with similar meaning land close together. A query is embedded with the same model, and search means "find the $k$ stored vectors closest to this one".

**Exact search is simple and linear.** Compare the query with every stored vector, keep the best $k$. With a million vectors, that is a million distance computations per query. It is perfectly accurate and fine for tens of thousands of vectors. It gets expensive at hundreds of millions, or at thousands of queries per second.

**[Approximate nearest neighbor (ANN) search](../../glossary.md#approximate-nearest-neighbor-search)** gives up a little accuracy for a lot of speed. Three families dominate:

- **IVF (inverted file).** Cluster the vectors with [k-means](../07-unsupervised/01-k-means.md) into `nlist` cells. At query time, find the `nprobe` cells whose centroids are closest to the query, and only scan vectors in those cells. A true neighbor that sits just across a cell boundary is missed unless you probe that cell too.
- **[HNSW](../../glossary.md#hnsw) (hierarchical navigable small world).** Build a graph where each vector links to a few dozen near neighbors, plus sparse upper layers with long-range links. Search starts at the top, greedily hops toward the query, then explores a candidate list of size `efSearch` at the bottom layer. Fast and accurate, but the links cost memory and the whole graph usually lives in RAM.
- **Product quantization (PQ).** Compress each vector by splitting it into chunks and replacing each chunk with the ID of its nearest codebook entry. A 3 KB vector becomes about 100 bytes. Distances become approximate, so PQ is often combined with IVF (IVF-PQ) and followed by re-scoring the top candidates with full vectors.

**Hybrid search.** Dense vectors match meaning but can miss exact tokens: error codes, product SKUs, rare names. Keyword search ([BM25](../../glossary.md#bm25)) matches those exactly. Running both and merging the ranked lists with [reciprocal rank fusion](../../glossary.md#reciprocal-rank-fusion) is a strong default.

## 2. Visualization

<!-- lab:vector-search -->
![Left: a 2D synthetic point cloud split into 16 IVF cells. The query, a star between two groups of points, probes the 2 nearest cells (blue points). Four of its true 10 nearest neighbors are inside the probed cells and found (teal rings); six sit in an unprobed cell just below and are missed (orange rings), so recall@10 is 0.4. Right: on 64-dimensional synthetic data with 128 cells, recall@10 rises from 0.58 at nprobe 1 (1.7% of brute-force distance computations) to 0.92 at nprobe 8 (6.9%) and 1.0 at nprobe 128.](../../figures/vector-search.png)
*Synthetic Gaussian clusters, fixed seeds. Right panel: 20,000 vectors, 200 held-out queries. The cost axis counts centroid distances plus scanned vectors, relative to scanning all 20,000.*
*Interactive version: [open the lab on the website](https://sreshtalluri.github.io/ml-ai-learn/labs/vector-search/).*
<!-- /lab -->

**Try it** (predict first, then check):
1. Set `nprobe` to 1 and drag the query onto a boundary between two cells. Predict recall@10. Then raise `nprobe` one step at a time until recall reaches 1.
2. Set `nprobe` equal to the number of clusters. Predict recall and the number of distance computations compared with brute force. Why is it slightly *more* than brute force?
3. Increase the number of clusters from 8 to 32 while keeping `nprobe` at 2. Predict what happens to cost and to average recall.
4. Raise $k$ from 5 to 20 with everything else fixed. Does recall go up or down, and why?

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $q$ | query embedding | $[d]$ |
| $x_i$ | stored vector $i$ | $[d]$ |
| $X$ | all stored vectors | $[N, d]$ |
| $N$ | number of stored vectors | scalar |
| $d$ | embedding dimension | scalar |
| $k$ | number of neighbors returned | scalar |
| $n_{\text{list}}$ | IVF number of cells (k-means clusters) | scalar |
| $n_{\text{probe}}$ | IVF cells scanned per query | scalar |
| $M$ | HNSW links per node (upper layers; $2M$ at layer 0) | scalar |
| $m$ | PQ number of subvectors | scalar |
| $c$ | RRF smoothing constant (commonly 60) | scalar |

### Similarity metrics

```math
\text{dot}(q, x) = q \cdot x, \qquad \cos(q, x) = \frac{q \cdot x}{\lVert q\rVert\,\lVert x\rVert}, \qquad \text{L2}(q, x) = \lVert q - x\rVert
```

If every vector has length 1, the three agree on the ranking: $\cos = \text{dot}$, and $\lVert q - x\rVert^2 = 2 - 2\cos(q, x)$. So normalize at write time and at query time, then use dot product (the cheapest).

Worked example. $q = [1, 0]$, $a = [0.8, 0.6]$ (length 1), $b = [3, 4]$ (length 5).
- Dot: $q \cdot a = 0.8$, $q \cdot b = 3$. Dot ranks $b$ first because $b$ is long.
- Cosine: $\cos(q, a) = 0.8/1 = 0.8$, $\cos(q, b) = 3/5 = 0.6$. Cosine ranks $a$ first.
- L2: $\lVert q - a\rVert = \sqrt{0.2^2 + 0.6^2} = \sqrt{0.4} = 0.632$, $\lVert q - b\rVert = \sqrt{2^2 + 4^2} = \sqrt{20} = 4.472$. L2 ranks $a$ first.
- After normalizing, $b \to [0.6, 0.8]$. Now dot $= 0.6$, and $\lVert q - b\rVert^2 = 0.4^2 + 0.8^2 = 0.8 = 2 - 2(0.6)$. All three metrics agree.

Use the metric the embedding model was trained with. Models trained with a dot-product objective may encode useful information (such as confidence or popularity) in the vector length, and normalizing them throws it away.

### Exact search cost

Brute force computes $N$ distances of $d$ multiply-adds each: $O(N \cdot d)$ per query, plus selecting the top $k$. With $N = 1{,}000{,}000$ and $d = 768$: $768{,}000{,}000$ multiply-adds per query. The work is a single matrix-vector product, so it runs at full hardware speed, which is why brute force on a GPU is competitive up to surprisingly large $N$.

### IVF cost

A query computes $n_{\text{list}}$ centroid distances, then scans about $\frac{n_{\text{probe}}}{n_{\text{list}}} N$ vectors (if cells are balanced).

```math
\text{cost}_{\text{IVF}} \approx \Big(n_{\text{list}} + \frac{n_{\text{probe}}}{n_{\text{list}}}N\Big)\, d
```

Worked example: $N = 1{,}000{,}000$, $d = 768$, $n_{\text{list}} = 1024$, $n_{\text{probe}} = 8$.
- Scanned vectors: $\frac{8}{1024} \times 1{,}000{,}000 = 7{,}812.5$.
- Distances: $1024 + 7{,}812.5 = 8{,}836.5$.
- Multiply-adds: $8{,}836.5 \times 768 = 6{,}786{,}432$.
- Fraction of brute force: $6{,}786{,}432 / 768{,}000{,}000 = 0.88\%$, about $113\times$ fewer.

A common starting point is $n_{\text{list}} \approx \sqrt{N}$ (here about 1,000) and then tuning $n_{\text{probe}}$ on a recall target. Cells are rarely balanced in practice; a few large cells make some queries much slower than average.

### Recall@k for an index

```math
\text{recall@}k = \frac{|\text{ANN top-}k \;\cap\; \text{exact top-}k|}{k}
```

This is recall *of the index against brute force*, not retrieval recall against human relevance labels. Example: exact top-10 is items 0 to 9. The index returns 0 to 7, plus 42 and 99. Overlap is 8, so recall@10 $= 8/10 = 0.8$. Measure it on a few hundred real queries, not random vectors, because real queries cluster where your data is dense.

### HNSW memory and knobs

Each node stores up to $2M$ neighbor IDs at the bottom layer (upper layers add a small fraction more). With $M = 32$ and 4-byte IDs: $64 \times 4 = 256$ bytes per vector, so $256$ MB of links for a million vectors, on top of the vectors themselves. Knobs: $M$ (more links: higher recall, more memory, slower build), `efConstruction` (build-time search width: better graph, slower build), and `efSearch` (query-time candidate list, must be at least $k$: higher recall, higher latency). `efSearch` is the knob you tune per query.

### Product quantization memory

Split each $d$-dimensional vector into $m$ subvectors of $d/m$ dimensions. For each position, learn a codebook of 256 centroids with k-means. Store each subvector as one byte (its centroid ID).

Worked example: $N = 1{,}000{,}000$, $d = 768$.
- Raw float32: $1{,}000{,}000 \times 768 \times 4 = 3{,}072{,}000{,}000$ bytes $= 3.072$ GB. In float16: $1.536$ GB.
- PQ with $m = 96$ (subvectors of $768/96 = 8$ dimensions): $96$ bytes per vector, so $96$ MB. Compression: $3072 / 96 = 32\times$.
- Codebooks: $96 \times 256 \times 8$ floats $\times 4$ bytes $= 786{,}432$ bytes, under 1 MB.
- Search: for each query, precompute a $96 \times 256$ table of distances from each query subvector to each centroid. Then each stored vector's approximate distance is the sum of 96 table lookups, with no multiplications.

### Reciprocal rank fusion

BM25 scores and cosine similarities live on different scales, so adding them directly is fragile. RRF uses only ranks:

```math
\text{RRF}(\text{doc}) = \sum_{\text{rankers } j} \frac{1}{c + \text{rank}_j(\text{doc})}, \qquad c = 60
```

Worked example. Dense ranking: A, B, C, D. BM25 ranking: E, F, A, B.
- A: $\frac{1}{60+1} + \frac{1}{60+3} = 0.016393 + 0.015873 = 0.032266$.
- B: $\frac{1}{62} + \frac{1}{64} = 0.016129 + 0.015625 = 0.031754$.
- E: $\frac{1}{61} = 0.016393$ (BM25 only).
- F: $\frac{1}{62} = 0.016129$.
- C: $\frac{1}{63} = 0.015873$.
- D: $\frac{1}{64} = 0.015625$.

Fused order: A, B, E, F, C, D. Documents both rankers like (A, B) rise to the top. E, BM25's first pick that dense search missed, still beats C and D. The large constant $c$ keeps one ranker's first place from dominating.

### Metadata filtering

Suppose a user may only see 2% of the corpus. **Post-filtering** (retrieve top 10, then drop disallowed ones) keeps on average $10 \times 0.02 = 0.2$ results: usually none. To expect 10 survivors you would need to fetch $10 / 0.02 = 500$ candidates. **Pre-filtering** (restrict the search to allowed vectors) returns 10 correct results, but it interacts with the index: an HNSW traversal that must skip 98% of nodes can get stuck, and IVF cells may contain few allowed vectors. Good engines use filter-aware traversal, per-tenant partitions, or fall back to brute force over the allowed subset when it is small. Access-control filters must be enforced in the retrieval query, never left to the LLM.

## 4. Implementation

```python
import numpy as np
from sklearn.cluster import KMeans

def brute_force(X, q, k):                       # X: [N, d], q: [d]
    return np.argsort(((X - q) ** 2).sum(1))[:k]

class IVF:
    def __init__(self, X, nlist):
        km = KMeans(n_clusters=nlist, n_init=1, random_state=0).fit(X)
        self.X, self.C = X, km.cluster_centers_                   # C: [nlist, d]
        self.lists = [np.flatnonzero(km.labels_ == c) for c in range(nlist)]

    def search(self, q, k, nprobe):
        cells = np.argsort(((self.C - q) ** 2).sum(1))[:nprobe]
        cand = np.concatenate([self.lists[c] for c in cells])
        return cand[np.argsort(((self.X[cand] - q) ** 2).sum(1))[:k]]

def rrf(rankings, c=60):
    s = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            s[doc] = s.get(doc, 0.0) + 1 / (c + rank)
    return sorted(s, key=s.get, reverse=True)
```

In production you would use a library or service rather than this loop. FAISS exposes the same ideas as index strings (for example an IVF index with PQ codes), and most vector databases and Postgres's pgvector offer HNSW and IVF indexes with `efSearch` or `probes` settings.

Runnable script (cost and memory arithmetic, recall@k, RRF, IVF recall vs `nprobe`, figure): [`code/16-rag/vector_search.py`](../../code/16-rag/vector_search.py).

## 5. Engineering

**Choosing the embedding model and dimension.** Evaluate candidate models on *your* queries and documents with a labeled set, not on a public leaderboard alone. Dimension drives memory and latency linearly: 3072 dimensions cost 4 times what 768 cost. Some models are trained so that a prefix of the vector is itself a good embedding (Matryoshka-style), letting you truncate to 256 or 512 dimensions and re-normalize. Quantizing stored vectors to int8 or binary with a full-precision re-score of the top candidates is another common memory lever.

**Reindexing when the model changes.** Vectors from two different embedding models are not comparable, even at the same dimension. Changing the model means re-embedding the whole corpus. Do it as a blue-green migration: build the new index alongside the old one, compare recall on your eval set, switch reads, then delete the old index. Store the model name and version with every vector and every request trace.

**Updates and deletes.** IVF appends new vectors to their nearest cell, but the centroids were trained on old data, so cells drift out of balance as the data changes; retrain periodically. HNSW inserts are incremental but deletes are usually tombstones that the search skips until a rebuild or compaction. Plan for re-index jobs, and track the fraction of deleted entries.

**Latency budget.** Tune on a recall target, for example "recall@10 at least 0.95 against brute force at p95 latency under 20 ms", and report both. A reranker after retrieval can recover precision, but it cannot recover a document the index never returned.

**Which system.**

| Option | Good for | Watch out for |
|---|---|---|
| In-memory (NumPy, FAISS in-process) | up to a few million vectors, batch jobs, prototypes | no persistence, filtering, or multi-writer story |
| pgvector (Postgres) | data already in Postgres, transactions, joins with metadata, moderate scale | tuning HNSW/IVF settings, filtered queries, memory on one node |
| Dedicated vector database or search engine | large scale, hybrid search, filtering, replication, multi-tenancy | another system to run, sync, and secure |

> [!WARNING]
> **Failure modes.** Query embedded with a different model or version than the corpus; mixing normalized and unnormalized vectors; post-filtering that returns nothing; `nprobe` or `efSearch` left at a tiny default; stale IVF centroids after heavy inserts; deleted documents still returned; measuring recall on random vectors instead of real queries.

### Common mistakes

- Using cosine for a model trained with dot product, or the reverse, without checking.
- Tuning the LLM prompt when the index recall against brute force is 0.7.
- Assuming exact search is always too slow. Under a few hundred thousand vectors, brute force is often fast enough and has perfect recall.
- Adding BM25 and cosine scores directly without normalization, instead of RRF or normalized scores.
- Forgetting that HNSW memory includes the graph, not just the vectors.

## 6. Knowledge check

<!-- quiz:vector-search -->
**[Take the vector search quiz](../../quizzes/vector-search.md)**
<!-- /quiz -->

**Practice exercise.** You store $N = 10{,}000{,}000$ vectors with $d = 1024$ in float32. (a) How much memory do the raw vectors take? (b) With PQ using $m = 128$ one-byte codes, how much? (c) An IVF index has $n_{\text{list}} = 4096$ and $n_{\text{probe}} = 16$. Approximately how many vectors does one query scan?

<details>
<summary>Solution</summary>

(a) $10^7 \times 1024 \times 4 = 40{,}960{,}000{,}000$ bytes $= 40.96$ GB.
(b) $10^7 \times 128 = 1{,}280{,}000{,}000$ bytes $= 1.28$ GB, a $32\times$ compression ($4096 / 128$).
(c) $\frac{16}{4096} \times 10^7 = 39{,}062.5$ vectors, plus 4096 centroid distances, about $43{,}159$ distances in total (0.43% of brute force).

</details>

**Implementation challenge.** Extend the script: build an IVF index on the synthetic data, then plot recall@10 against `nprobe` for $n_{\text{list}} \in \{32, 128, 512\}$ on the same cost axis. Which `nlist` gives the best recall at 5% of brute-force cost?

<details>
<summary>Solution sketch</summary>

Loop over `nlist`, build `IVF(X, nlist)`, and for each `nprobe` average recall and `(nlist + scanned) / N` over the held-out queries. Small `nlist` scans big cells (high recall per probe, but each probe is expensive); large `nlist` pays more for centroid comparisons and needs more probes because neighbors spread across more cells. The best setting is usually in the middle, near $\sqrt{N}$, and depends on how clustered the data is. That is why you tune it on real data.

</details>

## Summary

- Normalize embeddings and the three metrics agree; then use dot product. Otherwise use the metric the model was trained with.
- Exact search costs $O(N \cdot d)$ per query and has perfect recall. It is often fine up to a few hundred thousand vectors.
- IVF scans $n_{\text{probe}}/n_{\text{list}}$ of the data; HNSW walks a neighbor graph tuned by $M$ and `efSearch`; PQ compresses vectors about 32 times at some accuracy cost.
- Measure index recall@k against brute force on real queries, and tune for a recall target at a latency budget.
- Fuse BM25 and dense rankings with RRF, enforce permission filters inside retrieval, and re-embed everything when the embedding model changes.

**Next:** [Evaluating LLM systems](../17-llm-evaluation/01-llm-evaluation.md)

**Related:** [The RAG pipeline](01-rag-pipeline.md) · [K-nearest neighbors](../05-instance-and-probabilistic/01-k-nearest-neighbors.md) · [K-means](../07-unsupervised/01-k-means.md) · [Cheat sheet: vector search](../../cheatsheets/vector-search.md)

## Interview angle

<details>
<summary><strong>Explain how an HNSW index finds nearest neighbors, and what efSearch and M control.</strong></summary>

HNSW is a layered proximity graph. Every vector is a node in the bottom layer, linked to about $2M$ near neighbors; a random, exponentially shrinking subset also appears in upper layers with longer-range links. A query starts at an entry point in the top layer and greedily moves to whichever neighbor is closer to the query, drops a layer when it can't improve, and repeats. At the bottom it runs a best-first search keeping a candidate list of size `efSearch`, and returns the best $k$. $M$ is a build-time choice: more links give better recall and robustness but cost memory (about $2M \times 4$ bytes per vector, so 256 MB per million vectors at $M = 32$) and slower builds. `efSearch` is the query-time knob: raise it for recall, lower it for latency. It must be at least $k$.

</details>

<details>
<summary><strong>When would you pick IVF-PQ over HNSW, and when would you skip ANN entirely?</strong></summary>

Pick HNSW when the vectors fit in RAM and you want high recall at low latency with little tuning. Its cost is memory: full vectors plus the graph. Pick IVF-PQ when memory is the constraint, for example hundreds of millions of vectors: PQ stores each vector in tens of bytes instead of kilobytes, about 32 times smaller, and IVF limits each query to a few cells. You pay with lower raw recall, so you usually re-score the top few hundred candidates with full-precision vectors. Skip ANN when $N$ is small, up to a few hundred thousand vectors, or when filters cut the candidate set to a few thousand. Brute force is one matrix-vector product, has perfect recall, needs no tuning or rebuilds, and handles deletes trivially.

</details>

<details>
<summary><strong>After adding a "department" filter, users of small departments get zero or irrelevant results. What is happening and how do you fix it?</strong></summary>

Most likely the filter is applied after the vector search. The index returns the global top $k$, and when a department holds 1% of documents, on average $0.01 k$ of them survive, often zero. Confirm by logging how many candidates are fetched versus returned after filtering, broken down by department size. Fixes: use the engine's pre-filtering or filter-aware search so the constraint is applied during traversal; raise the candidate count adaptively based on filter selectivity; for very selective filters, fall back to brute force over the matching subset, which is cheap; or partition the index by tenant or department if queries always filter on it. Check HNSW specifically, because heavy filtering can strand the greedy traversal in regions with no allowed nodes. And if this is an access-control filter, make sure it's enforced in retrieval, never by the LLM.

</details>

<details>
<summary><strong>Estimate the memory for 50 million 768-dimensional embeddings, and how you would cut it.</strong></summary>

Raw float32 is $50{,}000{,}000 \times 768 \times 4 = 153.6$ GB. HNSW with $M = 32$ adds roughly $64 \times 4 = 256$ bytes per vector at the bottom layer, about 12.8 GB, so roughly 166 GB in RAM: several nodes or a very large one. Levers, in order of quality cost: float16 halves it to 76.8 GB; int8 scalar quantization quarters it to 38.4 GB with small recall loss; truncating a Matryoshka-trained model to 256 dimensions divides by 3; PQ with 96 one-byte codes gives $50{,}000{,}000 \times 96 = 4.8$ GB. With aggressive compression, keep full vectors on disk and re-score the top 100 to 200 candidates exactly. Then verify recall@k against brute force on real queries before shipping.

</details>
