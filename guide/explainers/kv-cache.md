---
title: Why long contexts eat GPU memory
summary: Watch the KV cache grow with every token and every user until it outgrows the model itself, then see how grouped-query attention and 8-bit caches win the memory back, and why decode speed is set by memory bandwidth.
lab: kv-cache
lesson: llm-inference
minutes: 7
---

# Why long contexts eat GPU memory

<!-- lab:kv-cache -->
![Left: memory used by weights plus KV cache for a synthetic 8B-style model at batch 16 as context grows from 1k to 32k tokens. Multi-head attention with 32 KV heads crosses an 80 GB line before 8k tokens; grouped-query attention with 8 KV heads reaches about 85 GB at 32k; multi-query attention with 1 KV head stays near the 16 GB weights line. Right: at 8k context and 3.0 TB/s, the upper bound on aggregate decode throughput rises from 176 tokens per second at batch 1 to about 2,100 at batch 48, while per-sequence speed falls from 176 to 44 tokens per second.](../figures/llm-inference.png)
<!-- /lab -->

A model that fits comfortably on a GPU can still run out of memory the moment users send long conversations. The culprit is the KV cache: the notes the model keeps on every token it has read. This explainer builds it up one picture at a time, using a synthetic 8B-style model on a synthetic 80 GB accelerator.

<!-- step:weights -->
## The fixed cost: weights

The model has $P = 8 \times 10^9$ parameters, each stored in 16 bits (2 bytes):

```math
8 \times 10^9 \times 2 = 16 \text{ GB}
```

That is the purple band. The red dashed line is the accelerator's 80 GB. So far there is plenty of room: the weights use a fifth of it.

The important thing about this 16 GB is that it never changes. One user or a hundred, ten tokens or a hundred thousand, the weights are loaded once and shared. Everything that grows comes from somewhere else.

<!-- step:per-token -->
## Every token leaves a trace

In causal attention a token's key and value vectors never change once computed, so the model saves them instead of recomputing the whole prefix for every new token. It saves one key and one value, in every layer, for every key-value head.

To show the problem clearly, start with full multi-head attention: 32 layers, 32 KV heads, 128 numbers per head, 2 bytes each.

```math
2 \cdot 32 \cdot 32 \cdot 128 \cdot 2 = 524{,}288 \text{ bytes} = 512 \text{ KiB per token}
```

The leading 2 counts keys and values. An 8,192-token conversation therefore holds $524{,}288 \times 8192 \approx 4.29$ GB. Look at the **KV per token** readout: that single number drives everything that follows.

<!-- step:context -->
## The cache grows with the context

Now slide the context from 512 tokens up to 64k and watch the orange marker move right. The blue line, weights plus cache, rises in a straight line from the top of the purple band. That is the whole story of the formula:

```math
\text{KV bytes} = 2 \cdot L \cdot n_{kv} \cdot d_{\text{head}} \cdot b \cdot T \cdot B
```

Everything except $T$ is fixed for this model, so memory is linear in context length. Twice the tokens, twice the cache. At 64k tokens one user's cache is $524{,}288 \times 65{,}536 \approx 34.4$ GB, already more than twice the weights.

<!-- step:batch -->
## Every user brings their own

A server rarely runs one conversation at a time. Batching lets each weight read from memory serve many users at once, which is how servers get good throughput. But keys and values belong to one conversation, so each user in the batch carries their own cache.

Hold the context at 8k and raise the batch $B$ from 1 to 8. The blue line's slope multiplies by 8: the cache goes from 4.29 GB to $4.29 \times 8 \approx 34.4$ GB, and the total to about 50 GB. Context length and batch size multiply each other.

<!-- step:overflow -->
## Then it stops fitting

Keep adding users. The total crosses the red line between 14 and 15 users, and at 16 it needs

```math
16 + 16 \times 4.29 \approx 84.7 \text{ GB}
```

on an 80 GB card. The marker turns red and **fits?** says no. Notice what ran out: the model still takes the same 16 GB. The cache is now more than four times the weights, and it alone decides how many users one GPU can serve. That is why a long-context feature can halve a server's throughput without changing the model.

<!-- step:gqa -->
## Share the keys: grouped-query attention

Grouped-query attention (GQA) lets a group of query heads share one key-value head. This model keeps all 32 query heads but stores only 8 KV heads, so each group of 4 shares. The cache per token drops by $32 / 8 = 4$, to 128 KiB.

Same 16 users at 8k: $17.18$ GB of cache, $33.18$ GB total, comfortably under the line. The largest batch that fits jumps to 59. Quality stays close to full multi-head attention, which is why most current open-weight models are built this way. Note that this is decided when the model is trained; you can't switch it on afterwards.

<!-- step:fp8-kv -->
## Store the cache in 8 bits

The other factor you can cut is $b$, the bytes per stored number. Keeping the cache in FP8 instead of 16-bit halves it: 8.59 GB for our batch.

Now stretch the context from 8k to 32k. In 16 bits this batch would need $16 + 68.7 \approx 84.7$ GB and fail again. In FP8 it needs about 50 GB and fits. The price is a small accuracy loss, usually showing up first in long-document recall, so check those evals before shipping it.

<!-- step:bandwidth -->
## Memory also sets the speed

Generating each new token means reading every weight and every cached key and value once. The arithmetic is tiny; the reading is the bottleneck. So at 3.0 TB/s:

```math
\text{tokens/s per user} \le \frac{3.0 \times 10^{12}}{16 \times 10^9 + \text{KV bytes}}
```

Raise the batch from 1 to 48 at 8k. Aggregate throughput climbs from about 176 to about 2,100 tokens/s, because the weights are read once for everyone. But each user slows from 176 to 44 tokens/s, because everyone's cache must be read every step. A smaller cache means more users fit and every step is faster.

**Try it yourself:** pick the **70B on one GPU** preset and find the cheapest combination of precision and GPU count that fits, or read the [full lesson](../lessons/18-llm-inference/01-llm-inference.md) for every number worked out.
