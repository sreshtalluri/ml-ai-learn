---
title: Prompting and context engineering
summary: "Steer a model without training it: prompts as tested specifications, few-shot examples, chain-of-thought and voting, and deciding what goes into the context window."
skill: llms
---

# Module 28: Prompting and context engineering

A model's behaviour at inference time is set by its weights and its context. This module is about the context: how to write instructions that specify the task, how examples and step-by-step reasoning raise accuracy, and how to assemble the window (instructions, tools, retrieved documents, history, memory) under a token budget. Every technique is measured on a test set, and all code runs offline with a scripted stand-in model that marks where a real client plugs in.

| Lever | What it changes | Failure mode |
|---|---|---|
| Instructions | label set, format, edge-case rules | vague spec, model guesses |
| Decoding settings | determinism versus variety | noisy outputs on extraction tasks |
| Few-shot examples | format and style by demonstration | biased or leaked examples |
| Chain-of-thought | per-step error on multi-step tasks | wasted tokens on simple tasks |
| Self-consistency | votes out random errors | amplifies systematic errors |
| Context assembly | what the model sees, in what order | buried facts, growing cost |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Prompt engineering](01-prompt-engineering.md) | write prompts as specifications, build a prompt test harness, and judge a change by pass rate and standard error |
| 2 | [Few-shot and chain-of-thought](02-few-shot-cot.md) | choose examples, explain why reasoning steps reduce compounding error, and compute self-consistency's accuracy and cost |
| 3 | [Context engineering](03-context-engineering.md) | budget the window, order retrieved material, compact history, select tools, and bound per-call cost |

Builds on [Decoding](../15-llms/02-decoding.md) and [the RAG pipeline](../16-rag/01-rag-pipeline.md); leads into [Tool use and agents](../19-agents/01-tool-use-and-agents.md).
