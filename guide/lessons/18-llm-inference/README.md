---
title: LLM inference serving
summary: "Where the time and memory go when an LLM generates text, and how servers batch, cache, and parallelize to serve many users cheaply."
skill: llms
---

# Module 18: LLM inference serving

Training a model is a one-time cost; serving it is paid on every token. This module turns LLM inference into arithmetic you can do on a whiteboard: how big the KV cache gets, why decoding is limited by memory bandwidth, how many users fit on a GPU, and what a million tokens costs. Then it covers the scheduling and memory tricks that real serving engines use.

| Quantity | What sets it | Main levers |
|---|---|---|
| Weight memory | parameters × bytes per parameter | FP8, INT8, INT4 quantization; tensor parallelism |
| KV-cache memory | layers, KV heads, head size, precision, context, batch | GQA/MQA, KV quantization, paged allocation, prefix sharing |
| Time to first token | queueing plus prefill (compute-bound) | prefix caching, chunked prefill, more replicas |
| Time per output token | bytes read per decode step (bandwidth-bound) | quantization, smaller KV, speculative decoding, batch size |
| Cost per token | GPU price ÷ sustained throughput × utilization | continuous batching, right-sized batches, autoscaling |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [LLM inference](01-llm-inference.md) | compute KV-cache size and decode speed limits, and explain GQA, quantization, FlashAttention, speculative decoding, and MoE |
| 2 | [Serving LLMs](02-serving-llms.md) | simulate static and continuous batching, explain paged KV and prefix caching, choose a parallelism strategy, and price a deployment |

**Interactive labs:** [KV cache and inference memory](https://sreshtalluri.github.io/ml-ai-learn/labs/kv-cache/) · [Static vs continuous batching](https://sreshtalluri.github.io/ml-ai-learn/labs/batching/)

**Cheat sheet:** [LLM inference](../../cheatsheets/llm-inference.md)
