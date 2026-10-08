<!-- GENERATED from rag-pipeline.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: The RAG pipeline

Covers the lesson [The RAG pipeline](../lessons/16-rag/01-rag-pipeline.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/rag-pipeline/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

A 900-word document is chunked with size 300 and overlap 0. How many chunks?

<details>
<summary>Answer</summary>

**3**

Chunks start at 0, 300, and 600.

</details>

## 2. Calculation (easy)

Normalized query vector $[0.6, 0.8]$ and normalized chunk vector $[0.8, 0.6]$. What is their cosine similarity?

<details>
<summary>Answer</summary>

**0.96** (within ±0.0001)

For unit vectors, cosine equals the dot product: $0.48 + 0.48 = 0.96$.

</details>

## 3. Arrange in order (easy)

Order the RAG pipeline stages.

- Generate a grounded answer with citations
- Rerank candidates
- Retrieve candidates for the query
- Embed and index chunks with metadata
- Chunk documents
- Ingest and clean documents

<details>
<summary>Answer</summary>

1. Ingest and clean documents
2. Chunk documents
3. Embed and index chunks with metadata
4. Retrieve candidates for the query
5. Rerank candidates
6. Generate a grounded answer with citations

Indexing happens ahead of time; retrieval, reranking, and generation happen per query.

</details>

## 4. Multiple choice (hard)

Users report wrong answers. For a failing question you find that the correct chunk is not in the top 20 retrieved results. Where should you work?

- **A.** The generation prompt
- **B.** Retrieval (chunking, query handling, hybrid search, embedding model)
- **C.** The decoding temperature
- **D.** Fine-tuning the LLM on the documents

<details>
<summary>Answer</summary>

**B.** Retrieval (chunking, query handling, hybrid search, embedding model)

If evidence never reaches the model, the prompt and decoding can't help. Fix retrieval and measure recall@k.

</details>

## 5. Select all that apply (medium)

When does sparse (BM25/TF-IDF) retrieval help a dense retriever? Select all that apply.

- **A.** Queries containing exact product codes or error IDs
- **B.** Rare technical terms the embedding model handles poorly
- **C.** Paraphrased questions with no shared words
- **D.** Names and acronyms

<details>
<summary>Answer</summary>

**A, B, D**

Sparse retrieval excels at exact terms; paraphrases are where dense retrieval shines. Hybrid combines both.

</details>

## 6. Reflection (hard)

An internal assistant indexes all company documents. How do you make sure an employee never receives content from documents they aren't allowed to see?

<details>
<summary>Answer</summary>

**Model answer.** Store access-control metadata with every chunk and apply the user's permissions as a filter in the retrieval query itself, so unauthorized chunks are never retrieved or placed in the prompt. Don't rely on the model to withhold information it was given. Also check permissions at ingestion, keep them in sync when access changes, and audit-log which chunks were served to whom.

Enforce authorization in the retrieval layer, before the model sees anything.

</details>
