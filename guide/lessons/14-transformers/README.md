---
title: Transformers
summary: Self-attention, multi-head attention, positional encoding, and the full transformer block.
skill: transformers
---

# Module 14: Transformers

Transformers let every token gather information from every other relevant token through attention, while processing all positions in parallel during training. That combination made them the architecture behind modern language, vision, and multimodal models.

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Self-attention](01-self-attention.md) | compute scaled dot-product attention by hand with every shape |
| 2 | [The transformer architecture](02-transformer-architecture.md) | trace a token through embeddings, attention, residuals, normalization, and the feed-forward network, and compare encoder, decoder, and encoder-decoder models |
| 3 | [BERT and masked language modeling](03-bert-masked-language-modeling.md) | implement the 80/10/10 masking rule and MLM loss, and choose an encoder or a decoder for a task |
| 4 | [Attention variants](04-attention-variants.md) | draw causal, sliding-window, strided, and global-token masks and size their compute, KV cache, and receptive field |

**Interactive labs:** [Self-attention](https://sreshtalluri.github.io/ml-ai-learn/labs/attention/) · [Attention masks](https://sreshtalluri.github.io/ml-ai-learn/labs/attention-masks/)
