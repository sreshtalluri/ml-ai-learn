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
| 4 | [Fine-tuning in practice](04-fine-tuning-in-practice.md) | build SFT data with loss masking, size GPU memory for full fine-tuning, LoRA, and QLoRA, and explain DPO and GRPO |
| 5 | [Mixture of experts](05-mixture-of-experts.md) | route tokens with top-k gates, compute expert capacity and dropped tokens, balance load, and separate total from active parameters |
| 6 | [Scaling laws](06-scaling-laws.md) | estimate training FLOPs with 6ND, find the compute-optimal model size and token count for a budget, and explain why served models are over-trained |
| 7 | [Open models: architecture walkthroughs](07-open-models-architecture-walkthroughs.md) | read a config.json, compute total and active parameters and KV-cache memory, and name the design choices behind Llama, Mixtral, Gemma, Qwen and DeepSeek |

**Interactive labs:** [Temperature, top-k, top-p](https://sreshtalluri.github.io/ml-ai-learn/labs/decoding/) · [LoRA fine-tuning memory](https://sreshtalluri.github.io/ml-ai-learn/labs/lora/) · [Mixture-of-experts routing](https://sreshtalluri.github.io/ml-ai-learn/labs/moe-routing/)
