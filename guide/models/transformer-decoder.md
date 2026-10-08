---
name: Transformer decoder
tags: [self-supervised, generative, nlp]
lessons: [transformer-architecture, self-attention, tokenization-and-pretraining]
labs: [attention, decoding]
---

# Transformer decoder

## Problem type

Autoregressive generation: text, code, and other token sequences.

## Input

Token IDs up to the context length.

## Output

Logits over the vocabulary for the next token at every position.

## Mental model

A transformer where each token may only look at tokens before it (a causal mask), trained to predict what comes next.

## Core objective

```math
\min_\theta -\frac{1}{T}\sum_t \log P_\theta(t_k \mid t_{<k})
```

## Training process

Self-supervised next-token pretraining on large corpora; then instruction tuning and preference optimization for assistants.

## Preprocessing

Byte-level BPE tokenization; packing sequences into the context window.

## Assumptions

Patterns in the training text generalize to the use case; the needed information fits in context or in weights.

## Key hyperparameters

Size (layers, width, heads), context length, learning rate schedule, batch size, data mixture; at inference, decoding settings.

## Good use cases

Generation, chat, summarization, code, few-shot task solving, agents with tools.

## Poor use cases

Low-latency classification where a small encoder suffices; facts that must be exactly current without retrieval.

## Strengths

General-purpose; in-context learning; scales well.

## Weaknesses

Expensive; hallucinations; sequential generation latency; quadratic attention in context length.

## Computational cost

About $2 \times$ parameters per generated token with a KV cache, plus attention over the context.

## Evaluation metrics

Perplexity (pretraining); task accuracy, groundedness, human preference, latency, and cost (applications).

## Failure modes

Missing causal mask in training; hallucination; prompt injection; context-window overflow.

## Minimal implementation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2"); model = AutoModelForCausalLM.from_pretrained("gpt2")
out = model.generate(**tok("The transformer", return_tensors="pt"), max_new_tokens=20, do_sample=True, top_p=0.9)
print(tok.decode(out[0]))
```

## Compared with neighbors

- **Transformer encoder:** bidirectional, for understanding and embeddings.
- **Encoder-decoder:** cross-attention to a separate input, for translation and summarization.

## Learn more

[The transformer architecture](../lessons/14-transformers/02-transformer-architecture.md) · [Self-attention](../lessons/14-transformers/01-self-attention.md)
