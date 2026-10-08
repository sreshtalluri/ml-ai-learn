<!-- GENERATED from word-embeddings.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Word embeddings

Covers the lesson [Word embeddings](../lessons/09-classical-nlp/02-word-embeddings.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/word-embeddings/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

An embedding table has a vocabulary of 32,000 and dimension 512. How many parameters does it hold?

<details>
<summary>Answer</summary>

**16384000**

$32{,}000 \times 512 = 16{,}384{,}000$.

</details>

## 2. Calculation (medium)

Cosine similarity between $[3, 4]$ and $[4, 3]$?

<details>
<summary>Answer</summary>

**0.96** (within ±0.0001)

Dot product 24; both norms 5; $24/25 = 0.96$.

</details>

## 3. Multiple choice (medium)

In skip-gram, what does the model predict?

- **A.** The center word from its context
- **B.** The context words from the center word
- **C.** The next sentence
- **D.** A document's label

<details>
<summary>Answer</summary>

**B.** The context words from the center word

CBOW predicts the center from the context; skip-gram does the reverse.

</details>

## 4. Multiple choice (medium)

"I sat by the river bank" and "I deposited cash at the bank." What does a static Word2Vec embedding give "bank"?

- **A.** Two different vectors, one per sense
- **B.** One vector, blending both senses
- **C.** No vector, because "bank" is ambiguous
- **D.** A random vector each time

<details>
<summary>Answer</summary>

**B.** One vector, blending both senses

Static embeddings have one row per word. Contextual (transformer) embeddings differ by sentence.

</details>

## 5. Fill in (medium)

Token IDs have shape `[4, 20]` and the embedding dimension is 64. What is the shape after the embedding lookup? Answer like `[a, b, c]`.

<details>
<summary>Answer</summary>

**[4, 20, 64]** or **[4,20,64]** or **4,20,64** or **(4, 20, 64)**

Each ID is replaced by a 64-dimensional row.

</details>

## 6. Reflection (hard)

You upgrade the embedding model in a semantic search system and re-embed all incoming queries with it, but leave the stored document vectors alone. Search quality collapses. Why?

<details>
<summary>Answer</summary>

**Model answer.** Different embedding models produce vectors in different, incompatible spaces: dimension 3 of one model means nothing in the other. Query vectors from the new model compared against document vectors from the old model give meaningless similarities. Re-embed the whole corpus with the new model (and version the index) before switching queries.

Queries and documents must share the same embedding model and version.

</details>
