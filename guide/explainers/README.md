---
title: Visual explainers
summary: Short, picture-first walkthroughs of the ideas people find hardest. On the website the interactive lab stays pinned beside the text and changes as you scroll.
---

# Visual explainers

Each explainer tells one idea as a sequence of pictures. Read them here on GitHub with the static figures, or on the [website](https://sreshtalluri.github.io/ml-ai-learn/explainers/), where the lab moves as you scroll and every control stays live.

| Explainer | Lab | Full lesson |
|---|---|---|
| [Underfitting, overfitting, and the U-shaped curve](bias-variance.md) | Underfitting vs overfitting | [Overfitting and the bias-variance trade-off](../lessons/02-ml-workflow/02-overfitting-and-bias-variance.md) |
| [How gradient descent finds the bottom](gradient-descent.md) | Gradient descent | [Gradient descent](../lessons/11-gradient-descent-backprop/01-gradient-descent.md) |
| [How backpropagation assigns blame](backpropagation.md) | Backpropagation, step by step | [Backpropagation](../lessons/11-gradient-descent-backprop/02-backpropagation.md) |
| [How self-attention mixes words](attention.md) | Self-attention | [Self-attention](../lessons/14-transformers/01-self-attention.md) |
| [How retrieval-augmented generation finds its evidence](rag.md) | RAG pipeline | [The RAG pipeline](../lessons/16-rag/01-rag-pipeline.md) |
| [Why long contexts eat GPU memory](kv-cache.md) | KV cache and inference memory | [LLM inference](../lessons/18-llm-inference/01-llm-inference.md) |

Every lab, not just these six, also has a guided tour on the website: press **Watch** for a captioned, animated walkthrough with optional narration.

## How they're built

An explainer is markdown with `<!-- step:<id> -->` markers. Each marker names a step of the lab's guided tour (the same steps the lab's **Watch** button plays), and the section below it is shown while that step is on screen. Rules are in [AUTHORING.md](../AUTHORING.md#visual-explainers-and-guided-tours).
