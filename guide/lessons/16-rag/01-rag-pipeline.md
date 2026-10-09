---
title: The RAG pipeline
summary: Build retrieval-augmented generation end to end (ingest, chunk, embed, index, retrieve, rerank, construct context, generate with citations) and diagnose its failures.
skill: llms
minutes: 45
prerequisites: [text-to-vectors, word-embeddings, adapting-llms]
related: [llm-evaluation, ai-security, production-architecture]
---

# The RAG pipeline

> **Mental model.** RAG is an open-book exam. Before answering, the system looks up the most relevant pages and puts them in front of the model, with instructions to answer only from those pages and cite them. If the lookup fails, even a brilliant model answers from the wrong pages.

**You will learn to**
- Describe every stage of a RAG pipeline and its main failure mode.
- Choose chunk size and overlap, and explain the trade-off.
- Compare sparse (BM25/TF-IDF), dense (embedding), and hybrid retrieval, plus metadata filtering and reranking.
- Construct a grounded prompt with citations and an "I don't know" path.
- Diagnose whether a bad answer is a retrieval or a generation failure.

**Why it matters.** RAG is the most common architecture for putting LLMs on private, changing data: support bots, internal search, documentation assistants. Most RAG quality problems are retrieval problems, which are fixable with search engineering and measurement.

## 1. Intuition

1. **Ingest:** load documents (PDFs, HTML, tickets), extract clean text, keep metadata (source, date, access permissions).
2. **Chunk:** split documents into retrievable pieces. Too small and a chunk lacks the context to answer; too large and the relevant sentence is diluted (and costs more tokens).
3. **Embed and index:** turn each chunk into a vector (dense retrieval) and/or a term index (sparse retrieval), stored with metadata.
4. **Retrieve:** turn the user's query into the same representation and fetch the top-$k$ chunks, optionally filtered by metadata (only this user's documents, only this product).
5. **Rerank:** a slower, more accurate model (often a cross-encoder) rescores the candidates.
6. **Construct context:** place the best chunks in the prompt with source labels.
7. **Generate:** instruct the model to answer only from the provided context, cite sources, and say when the answer isn't there.
8. **Evaluate:** measure retrieval and answers separately ([next lesson](../17-llm-evaluation/01-llm-evaluation.md)).

## 2. Visualization

<!-- lab:rag -->
![Left: evidence recall at 3 rises from 0.4 with 8-word chunks to 1.0 with 50-word chunks, while the number of context words sent to the model rises linearly. Right: a bar chart of five ranked results with relevant results at ranks 2 and 4, annotated with discounted gains, giving recall@3 0.5, reciprocal rank 0.5, and nDCG@5 0.651.](../../figures/rag-pipeline.png)

*Synthetic four-document collection and five queries, retrieved with TF-IDF. Very small chunks split the evidence sentence across chunk boundaries; bigger chunks find it but send more text (cost, latency, and distraction) to the model. The right panel is the metric example from the next lesson.*

*Interactive version: [open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/rag/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Pick the **Retrieval miss** question. Predict whether a better prompt could fix the answer, then turn on semantic matching instead.
2. Shrink chunks to 8 words. Predict what happens to evidence recall, then raise the chunk size and compare.
3. Set k = 1 and try each question in turn. Which still get their evidence into the context, and does the reranker change any of them?
4. Keep the default question and move the chunk size from 20 to 30 words. Predict where the evidence ranks, then check the worked example below to see why it moves.

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $q$ | query embedding |
| $c_i$ | embedding of chunk $i$ |
| $s_i$ | retrieval score for chunk $i$ |
| $k$ | number of chunks retrieved |
| $\alpha$ | weight between sparse and dense scores in hybrid search |

### Dense retrieval

```math
s_i = \cos(q, c_i) = \frac{q \cdot c_i}{\lVert q\rVert\,\lVert c_i\rVert}, \qquad \text{retrieve } \operatorname{top-}k(s)
```

### Hybrid scoring (one common form)

```math
s_i^{\text{hybrid}} = \alpha\, \tilde{s}_i^{\text{dense}} + (1 - \alpha)\, \tilde{s}_i^{\text{sparse}}
```

where $\tilde{s}$ are normalized scores. Reciprocal rank fusion, which adds $1/(60 + \text{rank})$ from each retriever, is a popular score-free alternative.

### Chunking arithmetic

With chunk size $S$ words and overlap $O$, consecutive chunks start every $S - O$ words. A document of $N$ words produces about $\lceil (N - O)/(S - O) \rceil$ chunks. Overlap reduces the chance that an answer sentence is cut in half, at the cost of more chunks.

### Worked example: one query, step by step

Collection: four short synthetic policy documents. Query: *"Are shipping costs refunded if my item was broken on arrival?"* Scores use the course's TF-IDF from [text to vectors](../09-classical-nlp/01-text-to-vectors.md): raw counts times $\ln(N/\text{DF})$, common words removed, cosine similarity. These are the numbers the lab shows.

With 20-word chunks and 5-word overlap (10 chunks):

| Rank | Source | TF-IDF cosine | Contains the evidence? |
|---|---|---|---|
| 1 | refunds | 0.408 | **yes**: "Shipping fees are not refundable unless the item arrived damaged." |
| 2 | shipping | 0.347 | no |
| 3 | shipping | 0.112 | no |

Now change only the chunk size, to 30 words (7 chunks):

| Rank | Source | TF-IDF cosine | Contains the evidence? |
|---|---|---|---|
| 1 | shipping | 0.303 | no |
| 2 | refunds | 0.283 | **yes** |

Same documents, same question, and the evidence drops from rank 1 to rank 2. The bigger refunds chunk also carries sentences about digital downloads and payment methods, which dilute its match, while the shipping chunk says "shipping" three times. With $k = 3$ the model still sees the evidence; with $k = 1$ it would not, and it would likely answer from shipping times. A reranker that reads the query and chunk together moves the refunds chunk back to the top. Chunk size is a retrieval hyperparameter: tune it against a labeled set, not by eye.

If you reproduce this with scikit-learn's `TfidfVectorizer`, the numbers differ, because its default smooths IDF to $\ln\frac{1+N}{1+\text{DF}} + 1$; with 20-word chunks it happens to rank the shipping chunk first (0.437 vs 0.425). Either way the point stands: lexical scores are fragile, so measure retrieval.

### Worked example: a retrieval miss

Query: *"Can I get reimbursed for a game I never opened?"* The answer exists ("Digital downloads are refundable only if they have not been accessed"), but the query shares no words with it: reimbursed versus refundable, game versus digital downloads, never opened versus not accessed. Every TF-IDF score is 0. A dense embedding retriever would likely match the meaning; a hybrid retriever gets the best of both. Without a fix, the generator either says "I don't know" (good) or invents an answer (bad).

## 4. Implementation

```python
# Index (once): chunk, embed, store with metadata
from sentence_transformers import SentenceTransformer          # pip install sentence-transformers
import numpy as np

encoder = SentenceTransformer("all-MiniLM-L6-v2")
chunks = [c for doc in docs for c in chunk(doc.text, size=200, overlap=40)]
C = encoder.encode([c.text for c in chunks], normalize_embeddings=True)    # [n_chunks, d]

# Query time: retrieve, (rerank), build a grounded prompt
def answer(question, k=5):
    q = encoder.encode([question], normalize_embeddings=True)[0]
    top = np.argsort(-(C @ q))[:k]                                          # cosine = dot for normalized vectors
    context = "\n\n".join(f"[{i}] ({chunks[i].source}) {chunks[i].text}" for i in top)
    prompt = (
        "Answer using ONLY the sources below. Cite sources like [3]. "
        "If the sources do not contain the answer, say you don't know.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}"
    )
    return llm(prompt), top
```

In production the in-memory array becomes a vector database or a search engine with vector support, and you add metadata filters and a reranker.

Runnable script (chunking, TF-IDF retrieval, the miss, recall versus chunk size, ranking metrics; fully offline): [`code/16-rag/rag_eval.py`](../../code/16-rag/rag_eval.py).

## 5. Engineering

**Retrieval first.** Measure retrieval recall@k on a labeled set of (question, relevant chunk) pairs before tuning prompts. If the evidence isn't retrieved, no prompt can fix the answer.

**Chunking.** Split on structure (headings, paragraphs) rather than fixed word counts where possible; keep titles and section headers with chunks; use overlap; tune size on your retrieval metrics.

**Hybrid search and reranking.** Sparse retrieval handles exact terms, IDs, and rare words; dense handles paraphrases. Hybrid is a strong default. Rerank the top 20 to 100 candidates with a cross-encoder, accepting the extra latency.

**Metadata filtering and permissions.** Filter by access rights *before* retrieval results reach the model. Never rely on the model to withhold documents the user shouldn't see.

**Freshness and versioning.** Re-index on document changes; version the index together with the embedding model; record which index version answered each request.

**RAG versus fine-tuning.** RAG changes the runtime context, so updating knowledge means re-indexing, and answers can cite sources. Fine-tuning changes weights and is the right tool for behavior and format, not for facts that change.

> [!WARNING]
> **Failure modes.** Parsing failures (tables and PDFs turned into garbage); chunks that split the answer; vocabulary mismatch for sparse retrieval and exact-term misses for dense retrieval; stale indexes; too much context, which dilutes attention and costs money; the model ignoring context and answering from memory; **retrieved documents containing prompt injections** (see [security](../21-safety-security/01-ai-security.md)).

### Common mistakes

- Judging a RAG system by how fluent its answers sound.
- Using different embedding models (or versions) for documents and queries.
- Skipping the "I don't know" instruction and the evaluation that rewards it.
- Retrieving across all users' documents and filtering afterwards.

## 6. Knowledge check

<!-- quiz:rag-pipeline -->
**[Take the RAG quiz](../../quizzes/rag-pipeline.md)**
<!-- /quiz -->

**Practice exercise.** A 1,000-word document is split into chunks of 200 words with 50 words of overlap. How many chunks result, and how many words are stored in total?

<details>
<summary>Solution</summary>

Chunks start every $200 - 50 = 150$ words: at 0, 150, 300, 450, 600, 750 (the last covers 750 to 950), and 900 (covers 900 to 1000). That is $\lceil (1000 - 50)/150 \rceil = \lceil 6.33 \rceil = 7$ chunks. Stored words: six full chunks of 200 plus a final chunk of 100 = 1,300 words, 30% more than the original because of overlap.
</details>

**Implementation challenge.** Extend the script: add a dense retriever (any sentence-embedding model), combine it with TF-IDF using reciprocal rank fusion, and report evidence recall@3 for sparse, dense, and hybrid on 20 questions you write about the documents, including paraphrased ones.

## Summary

- RAG retrieves relevant chunks at query time and asks the model to answer from them with citations.
- Chunk size and overlap trade answer completeness against context length and precision.
- Sparse, dense, and hybrid retrieval have complementary failure modes; reranking improves precision.
- Most failures are retrieval failures: measure recall@k first.
- RAG changes context, not weights, which makes it the right tool for private and changing knowledge.

**Next:** [Vector search](02-vector-search.md)

**Related:** [From text to vectors](../09-classical-nlp/01-text-to-vectors.md) · [Adapting LLMs](../15-llms/03-adapting-llms.md) · [Model card: embedding model](../../models/embedding-model.md) · [Model card: reranker](../../models/reranker.md)

## Interview angle

<details>
<summary><strong>Your RAG system gives a wrong answer even though the right document is in the corpus. How do you debug it?</strong></summary>

Split the failure into retrieval versus generation by checking whether the evidence chunk reached the prompt. Log the retrieved chunk IDs for the query and look for the gold chunk. If it is missing, it's a retrieval failure: check that the document parsed cleanly (tables and PDFs often turn to garbage), that chunking didn't split the answer across boundaries, that a metadata or permission filter didn't exclude it, that the index isn't stale, and that queries and documents use the same embedding model version. Vocabulary mismatch, like "reimbursed" versus "refundable", calls for dense or hybrid retrieval; exact IDs call for sparse. If it ranked just below $k$, raise $k$ or add a reranker. If the chunk was in the prompt, it's a generation failure: the model answered from memory, was distracted by too much context, or lost a mid-context chunk. Tighten grounding instructions and require citations. Then add the case to the evaluation set.

</details>

<details>
<summary><strong>How do you choose chunk size and overlap?</strong></summary>

Chunk size trades completeness against precision and cost, so tune it on retrieval metrics rather than picking a number. Chunks that are too small lose the context needed to answer and split evidence sentences: in the course's synthetic test, evidence recall@3 was 0.4 with 8-word chunks and 1.0 with 50-word chunks. Chunks that are too large dilute the relevant sentence's embedding with unrelated text, lower ranking precision, and send more tokens per retrieved chunk, which costs money, adds latency, and distracts the model. Overlap reduces the chance that an answer straddles a boundary, at the cost of more chunks and storage. A 1,000-word document with 200-word chunks and 50-word overlap gives 7 chunks and stores 1,300 words, 30% more. In practice, split on structure (headings, paragraphs) first, keep the section title with each chunk, start around a few hundred tokens, and compare candidates on recall@k with a labeled question set.

</details>

<details>
<summary><strong>Sparse, dense, or hybrid retrieval, and where does a reranker fit?</strong></summary>

Hybrid retrieval plus a reranker is the strong default, because sparse and dense retrieval fail differently. Sparse retrieval (BM25, TF-IDF) matches exact terms, so it is excellent for product codes, error messages, names, and rare words, but scores zero on paraphrases: "reimbursed for a game I never opened" shares no words with "digital downloads are refundable if not accessed." Dense retrieval embeds meaning, so it catches paraphrases but can miss exact identifiers and rare jargon. Hybrid combines them with a weighted sum of normalized scores or with reciprocal rank fusion, which adds $1/(60 + \text{rank})$ from each list and needs no score calibration. Both retrievers are bi-encoders: query and document are encoded separately, so documents are precomputed and search is fast. A cross-encoder reranker reads query and chunk together, which is far more accurate but needs one model call per pair, so apply it to only the top 20 to 100 candidates.

</details>

<details>
<summary><strong>Design retrieval for 10 million chunks across many customer tenants. How big is the index, and how do you enforce permissions?</strong></summary>

Raw vectors for 10M chunks at 768 dimensions in float32 take $10^7 \times 768 \times 4 = 30.7$ GB; fp16 halves that to 15.4 GB and int8 to 7.7 GB. ANN structures such as HNSW add graph overhead, and product quantization can compress further at some recall cost. That fits on one large node or a few shards; plan for replicas to meet throughput. Store text, source, tenant ID, access groups, document version, and embedding model version alongside each vector. Enforce permissions as a filter inside the retrieval query, either by partitioning by tenant (a separate index or namespace each, the simplest strong isolation) or by filtered ANN search on tenant and access-group metadata. Never retrieve globally and filter afterwards, and never ask the model to withhold documents. Add a sparse index for exact-term queries, re-index incrementally on document changes, and version the index together with the embedding model.

</details>
