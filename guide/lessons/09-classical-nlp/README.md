---
title: Classical NLP
summary: Turn text into numbers with tokenization, bag-of-words, n-grams, TF-IDF, and word embeddings, then extract entities and discover topics.
skill: nlp
---

# Module 9: Classical NLP, from text to vectors

Models need numbers. This module turns text into vectors two ways: sparse counts weighted by rarity (bag-of-words, TF-IDF), which are fast, cheap, and an excellent baseline, and dense embeddings, which place related words near each other.

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [From text to vectors](01-text-to-vectors.md) | tokenize, build a document-term matrix, and compute TF-IDF and cosine similarity by hand |
| 2 | [Word embeddings](02-word-embeddings.md) | explain Word2Vec's skip-gram and CBOW objectives and the difference between static and contextual embeddings |
| 3 | [Named entity recognition](03-named-entity-recognition.md) | tag entities with BIO labels, decode valid sequences with Viterbi, and score with entity-level F1 |
| 4 | [Topic modeling](04-topic-modeling.md) | implement collapsed Gibbs LDA, choose the number of topics, and pick between LDA, NMF, and BERTopic |

**Interactive lab:** [TF-IDF calculator](https://sreshtalluri.github.io/ml-ai-learn/labs/tfidf/)
