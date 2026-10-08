---
name: Large language model
tags: [self-supervised, generative, nlp]
lessons: [tokenization-and-pretraining, decoding, adapting-llms, rag-pipeline]
labs: [decoding]
---

# Large language model (LLM)

## Problem type

General-purpose text (and multimodal) generation, instruction following, and tool use.

## Input

A prompt of tokens: system instructions, user messages, retrieved context, tool results.

## Output

Generated tokens, sampled from next-token distributions; optionally structured outputs or tool calls.

## Mental model

A very large decoder-only transformer trained to predict the next token, then tuned to follow instructions. It produces probable text, which is not the same as true text.

## Core objective

Pretraining: next-token cross-entropy. Adaptation: supervised fine-tuning on instruction-response pairs and preference optimization (RLHF, DPO).

## Training process

Pretraining on trillions of tokens; instruction tuning; preference optimization; safety training. Users typically adapt via prompting, RAG, or LoRA.

## Preprocessing

The model's tokenizer and chat template; prompt construction with clear separation of trusted instructions and untrusted content.

## Assumptions

The task can be expressed in text; needed knowledge is in context or reliably in weights.

## Key hyperparameters

Model choice and size; context length; decoding (temperature, top-p, max tokens); prompt and retrieval design; LoRA rank if fine-tuning.

## Good use cases

Drafting, summarization, extraction into schemas, Q&A over documents with RAG, code assistance, agents with bounded tools.

## Poor use cases

Exact arithmetic or lookups without tools; high-stakes decisions without verification; facts that change without retrieval.

## Strengths

Broad capability; few-shot learning; natural-language interface.

## Weaknesses

Hallucinations; prompt injection; cost and latency; non-determinism; uncalibrated confidence.

## Computational cost

Roughly $2 \times$ parameters FLOPs per token; memory for weights plus KV cache.

## Evaluation metrics

Task accuracy and rubric scores; groundedness and citation correctness; safety and injection resistance; latency, tokens/sec, cost per request.

## Failure modes

Confident fabrication; following injected instructions; truncated context; regressions after model or prompt changes.

## Minimal implementation

```python
messages = [{"role": "system", "content": "Answer only from the provided sources; cite them."},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}]
answer = client.chat(model=MODEL, messages=messages, temperature=0.2)   # any chat-completions API
```

## Compared with neighbors

- **Small fine-tuned encoder:** cheaper and faster for fixed classification tasks.
- **RAG system:** an LLM plus retrieval for current or private knowledge.

## Learn more

[Tokenization and pretraining](../lessons/15-llms/01-tokenization-and-pretraining.md) · [Adapting LLMs](../lessons/15-llms/03-adapting-llms.md)
