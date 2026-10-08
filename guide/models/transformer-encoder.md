---
name: Transformer encoder
tags: [self-supervised, supervised, classification, nlp]
lessons: [transformer-architecture, self-attention]
labs: [attention]
---

# Transformer encoder

## Problem type

Understanding tasks: classification, token tagging, extraction, and producing embeddings.

## Input

Token IDs (plus attention masks), up to a maximum context length.

## Output

A contextual vector for every token; pooled vectors for sequence-level tasks.

## Mental model

Every token looks at every other token in both directions, repeatedly, so each token's vector comes to reflect its full context.

## Core objective

Pretraining with masked-language modeling (predict hidden tokens); then fine-tuning with task losses, or contrastive training for embeddings.

## Training process

Large-scale self-supervised pretraining, then supervised fine-tuning on labeled data.

## Preprocessing

The model's own tokenizer; truncation and padding to the context length.

## Assumptions

Enough context fits in the window; the pretraining domain resembles the task domain.

## Key hyperparameters

Model size (layers, width, heads), max length, fine-tuning learning rate, epochs, pooling strategy.

## Good use cases

Text classification, named-entity recognition, semantic search embeddings, rerankers (cross-encoders).

## Poor use cases

Open-ended text generation; documents far longer than the context window without chunking.

## Strengths

Bidirectional context; strong transfer learning; much cheaper than large decoders for understanding tasks.

## Weaknesses

Not generative; quadratic attention cost in length; needs fine-tuning data for best results.

## Computational cost

About $2 \times$ parameters per token plus $O(n^2 d)$ attention per layer.

## Evaluation metrics

Task metrics (F1, accuracy); retrieval metrics for embeddings (recall@k, nDCG).

## Failure modes

Truncated inputs silently dropping key text; domain shift; tokenizer mismatches.

## Minimal implementation

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
logits = model(**tok(["great product"], return_tensors="pt")).logits
```

## Compared with neighbors

- **Transformer decoder:** causal, generative.
- **TF-IDF + linear model:** far cheaper baseline, weaker semantics.

## Learn more

[The transformer architecture](../lessons/14-transformers/02-transformer-architecture.md)
