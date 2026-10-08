---
title: Large language models
summary: Tokens, pretraining, decoding, adaptation, and the limits of models that predict probable text.
skill: llms
---

# Module 15: How large language models work

An LLM is a decoder-only transformer trained to predict the next token. Everything else, from chat to code generation, is built on top of that objective. This module covers what goes in (tokens), how it is trained (next-token cross-entropy), how text comes out (decoding), and how to adapt it (prompting, fine-tuning, LoRA, preference optimization).

> [!IMPORTANT]
> An LLM produces probability distributions over tokens. It is not a factual database, and confident wording is not calibrated certainty.

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Tokenization and pretraining](01-tokenization-and-pretraining.md) | explain subword tokenization, the next-token objective, cross-entropy, and perplexity |
| 2 | [Decoding](02-decoding.md) | turn logits into text with greedy, temperature, top-k, top-p, and beam search |
| 3 | [Adapting LLMs](03-adapting-llms.md) | choose between prompting, RAG, fine-tuning, LoRA, and preference optimization, and explain hallucinations |

**Interactive lab:** [Temperature, top-k, top-p](https://sreshtalluri.github.io/ml-ai-learn/labs/decoding/)
