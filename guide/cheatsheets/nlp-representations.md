---
title: NLP representations
summary: From bag-of-words and TF-IDF to static and contextual embeddings, with trade-offs.
---

# NLP representations

| Representation | Vector | Captures | Misses | Cost |
|---|---|---|---|---|
| Bag-of-words counts | sparse, one column per word | which words appear | order, meaning, synonyms | tiny |
| N-grams | sparse, words + short phrases | local phrases ("not good") | long-range order | small, more features |
| TF-IDF | sparse, counts × $\ln(N/\text{DF})$ | distinctive words | meaning, synonyms | tiny |
| BM25 | sparse retrieval score | term relevance with saturation and length normalization | paraphrases | tiny |
| Static embeddings (Word2Vec, GloVe) | dense, one per word | similarity of usage | context (one vector for "bank") | small |
| Contextual embeddings (transformers) | dense, one per token occurrence | meaning in context | needs compute | larger |
| Sentence embeddings | dense, one per text | semantic similarity of whole texts | exact terms, numbers | moderate |

**Key formulas:** $\text{TF-IDF}(t,d) = \text{TF}(t,d)\times\ln\frac{N}{\text{DF}(t)}$; $\cos(a,b) = \frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$.

**Preprocessing:** tokenize; lowercase and normalize for classical models; keep negations; fit vocabularies on training data only. Transformers use their own tokenizer and need no stemming or stop-word removal.

**Choosing:** start with TF-IDF + logistic regression as the baseline; move to embeddings for semantic similarity or retrieval; use hybrid (BM25 + dense) search in RAG.

Lessons: [From text to vectors](../lessons/09-classical-nlp/01-text-to-vectors.md) · [Word embeddings](../lessons/09-classical-nlp/02-word-embeddings.md) · Lab: [TF-IDF](https://sreshtalluri.github.io/ml-ai-learn/labs/tfidf/)
