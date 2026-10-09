<!-- GENERATED from vector-search.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Vector search

Covers the lesson [Vector search](../lessons/16-rag/02-vector-search.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/vector-search/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

An IVF index holds $N = 2{,}000{,}000$ vectors in $n_{\text{list}} = 2000$ balanced cells. With $n_{\text{probe}} = 10$, how many distance computations does one query make in total (centroids plus scanned vectors)?

<details>
<summary>Answer</summary>

**12000**

Centroids: 2,000. Scanned: $\frac{10}{2000} \times 2{,}000{,}000 = 10{,}000$. Total $= 12{,}000$, versus 2,000,000 for brute force (0.6%).

</details>

## 2. Calculation (medium)

How many MB (decimal) do PQ codes take for 5,000,000 vectors of dimension 1536 with $m = 192$ one-byte subvector codes? (Ignore codebooks.)

<details>
<summary>Answer</summary>

**960**

$5{,}000{,}000 \times 192 = 960{,}000{,}000$ bytes $= 960$ MB. Raw float32 would be $5{,}000{,}000 \times 1536 \times 4 = 30.72$ GB, a $32\times$ compression.

</details>

## 3. Calculation (medium)

With RRF and $c = 60$, a document is ranked 2nd by dense search and 5th by BM25. What is its fused score? (5 decimals)

<details>
<summary>Answer</summary>

**0.03151** (within ±0.00002)

$\frac{1}{62} + \frac{1}{65} = 0.016129 + 0.015385 = 0.031514$.

</details>

## 4. Calculation (easy)

Brute-force top-5 is {A, B, C, D, E}. The ANN index returns {A, C, E, X, Y}. What is recall@5?

<details>
<summary>Answer</summary>

**0.6**

Overlap is {A, C, E}, so $3/5 = 0.6$.

</details>

## 5. Multiple choice (medium)

All stored and query embeddings are normalized to length 1. Which statement is true?

- **A.** Cosine, dot product, and L2 distance produce the same ranking.
- **B.** Dot product ranks longer vectors higher, so it differs from cosine.
- **C.** L2 distance must be used because cosine is undefined for unit vectors.
- **D.** Only cosine is correct; dot product would need re-normalizing per query.

<details>
<summary>Answer</summary>

**A.** Cosine, dot product, and L2 distance produce the same ranking.

For unit vectors, $\cos = \text{dot}$ and $\lVert q - x\rVert^2 = 2 - 2\cos$, a decreasing function of cosine. All three rank identically, so use the cheapest (dot).

- **A:** Correct.
- **B:** All vectors have the same length, so length can't change the ranking.
- **C:** Cosine is well defined for any non-zero vector.
- **D:** Dot product already equals cosine for unit vectors.

</details>

## 6. Multiple choice (hard)

A tenant owns 0.5% of the vectors. Search retrieves the global top 20, then drops other tenants' results. Users of this tenant usually get nothing. What is the best fix?

- **A.** Increase nprobe.
- **B.** Apply the tenant filter inside the search (pre-filtering or a per-tenant index), or brute-force the tenant's small subset.
- **C.** Switch from cosine to dot product.
- **D.** Ask the LLM to ignore documents from other tenants.

<details>
<summary>Answer</summary>

**B.** Apply the tenant filter inside the search (pre-filtering or a per-tenant index), or brute-force the tenant's small subset.

Post-filtering keeps on average $20 \times 0.005 = 0.1$ results. The filter must constrain the search itself; for a tiny subset, exact search over it is cheap and perfect.

- **A:** More probes improve recall of the global top 20, which is still mostly other tenants.
- **B:** Correct.
- **C:** The metric doesn't change which tenant the results belong to.
- **D:** Access control must never depend on the model; the data shouldn't reach it.

</details>

## 7. Multiple choice (medium)

You switch to a better embedding model with the same dimension (768). What must happen to the existing index?

- **A.** Nothing, since the dimension matches.
- **B.** Re-embed the whole corpus with the new model and rebuild the index, then switch queries over.
- **C.** Only new documents need the new model.
- **D.** Normalize the old vectors so they match the new model.

<details>
<summary>Answer</summary>

**B.** Re-embed the whole corpus with the new model and rebuild the index, then switch queries over.

Different models define different vector spaces. Queries from the new model compared with documents from the old one give meaningless similarities. Re-embed everything, ideally as a blue-green migration.

- **A:** Same shape does not mean the same space.
- **B:** Correct.
- **C:** Mixed spaces make similarity scores incomparable.
- **D:** Normalization fixes length, not the coordinate system.

</details>

## 8. Select all that apply (hard)

Which situations favor plain exact (brute-force) search over an ANN index? Select all that apply.

- **A.** 50,000 vectors, a few queries per second.
- **B.** Queries always filtered to a subset of a few thousand vectors.
- **C.** 500 million vectors, 2,000 queries per second, tight memory budget.
- **D.** Building an evaluation set to measure an ANN index's recall.

<details>
<summary>Answer</summary>

**A, B, D**

Small collections and small filtered subsets are cheap to scan exactly, and exact search is the ground truth for measuring ANN recall. Hundreds of millions of vectors at high QPS need an ANN index, likely with compression.

- **A:** Correct.
- **B:** Correct.
- **C:** This scale is exactly what IVF-PQ or HNSW are for.
- **D:** Correct.

</details>

## 9. Reflection (medium)

Your HNSW index has recall@10 of 0.85 against brute force and latency to spare. Which knob do you change first, in which direction, and why?

<details>
<summary>Answer</summary>

**Model answer.** Raise efSearch, the query-time candidate list size. It increases recall at the cost of latency without rebuilding the index. Only if recall still plateaus would I rebuild with a larger M or efConstruction, which costs memory and build time.

efSearch is the per-query recall/latency knob; M and efConstruction require a rebuild.

</details>
