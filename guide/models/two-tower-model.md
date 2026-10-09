---
name: Two-tower model
tags: [supervised, low-latency]
lessons: [ml-system-design]
labs: [recsys-funnel]
---

# Two-tower model

## Problem type

Retrieval: candidate generation for recommendation, search, and ads, where you must find the few hundred most relevant items out of millions within milliseconds.

## Input

Query-side features (user history, profile, context, or a search query) for one tower; item-side features (ID, content embedding, metadata) for the other.

## Output

A $d$-dimensional vector per user and per item (typically $d$ = 32 to 256); the relevance score is their dot product.

## Mental model

Place users and items in the same space so that a user sits close to the items they will engage with. Because items never interact with users before the final dot product, every item vector can be computed ahead of time and searched with an approximate nearest neighbor index.

## Core objective

Contrastive or sampled softmax: for each (user, engaged item) pair, raise the dot product with the engaged item relative to negatives, usually the other items in the batch, with a $\log Q(i)$ correction so popular items are not over-penalized.

## Training process

Train both towers jointly on logged positive interactions (clicks, watches, purchases) with in-batch and random negatives, optionally mixing in hard negatives (shown but skipped). Precompute item vectors, build the ANN index, and refresh it as items and the item tower change.

## Preprocessing

Hash or vocabulary-encode IDs; normalize numeric features; build fixed-length histories of recent interactions; often L2-normalize output vectors and use a temperature on the logits.

## Assumptions

Relevance can be approximated by a single dot product between independent user and item representations; logged engagement is a usable (if biased) proxy for relevance.

## Key hyperparameters

Embedding dimension, tower depth and width, number and type of negatives, softmax temperature, ANN index parameters (graph degree, search breadth, quantization).

## Good use cases

Candidate generation over very large catalogs; cold-start items, when the item tower uses content features; semantic search, where it is the same idea as a bi-encoder.

## Poor use cases

Final ranking that needs cross features (user × item interactions); small catalogs where a full ranker can score every item anyway.

## Strengths

Sublinear retrieval through ANN; item vectors cached offline; a single user-tower pass per request; works with content features for new items.

## Weaknesses

No early user-item interaction limits accuracy; vectors from different model versions are incompatible; popularity bias from in-batch negatives; stale index after item changes.

## Computational cost

Per request: one user-tower forward pass plus an ANN search (milliseconds). Offline: one item-tower pass per item, plus index build; storage $N \times d$ floats (10M items × 64 dims × 4 bytes = 2.56 GB before quantization).

## Evaluation metrics

Recall@k against held-out future engagements (time-based split); coverage and diversity of retrieved items; online, the downstream ranker's metrics.

## Failure modes

User and item towers from different training runs; index not rebuilt after retraining; candidates dominated by popular items; recall measured on random splits that leak the future.

## Minimal implementation

```python
import torch, torch.nn.functional as F
u = F.normalize(user_tower(user_feats), dim=-1)      # [B, d]
v = F.normalize(item_tower(item_feats), dim=-1)      # [B, d], the engaged item per row
logits = u @ v.T / 0.05 - log_q[None, :]             # [B, B]; other rows' items are negatives
loss = F.cross_entropy(logits, torch.arange(len(u)))
```

## Compared with neighbors

- **Embedding model (bi-encoder for text):** the same architecture applied to queries and passages.
- **Gradient boosting ranker:** uses rich cross features, far more accurate per item, far too expensive to run over the catalog.
- **Reranker (cross-encoder):** reads query and item together; the most accurate and most expensive stage.

## Learn more

[ML system design](../lessons/22-ml-system-design/01-ml-system-design.md) · [Word embeddings](../lessons/09-classical-nlp/02-word-embeddings.md) · [Model card: embedding model](embedding-model.md) · [Model card: gradient boosting](gradient-boosting.md) · [Model card: reranker](reranker.md)
