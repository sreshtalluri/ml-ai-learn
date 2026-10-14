---
title: Gradient checkpointing
summary: Trade compute for memory by storing only some activations and recomputing the rest during backward; size activation memory and pick a checkpointing policy.
skill: deep-learning
minutes: 35
prerequisites: [backpropagation, transformer-architecture, gradient-clipping-amp]
related: [fine-tuning-in-practice, zero-parameter-sharding, llm-inference]
adapted_from: [phases/10-llms-from-scratch/34-gradient-checkpointing]
---

# Gradient checkpointing

> **Mental model.** Backpropagation is like retracing a hike in reverse: you need to know where you were at every point. You can photograph every step on the way up (lots of memory), or photograph only a few landmarks and re-walk each short stretch from its landmark when you need it (more walking, far fewer photos).

**You will learn to**
- Explain which tensors the backward pass needs and why they make [activation](../../glossary.md#activation) memory grow with layers and tokens.
- Implement checkpointing: keep segment inputs in the forward pass, re-run each segment's forward during backward, and get identical [gradients](../../glossary.md#gradient).
- Compute activation memory for a transformer with and without checkpointing, and the extra compute it costs.
- Choose between full, selective, and no checkpointing, or offloading, for a given memory budget.

**Why it matters.** Weights, gradients, and optimizer state are fixed per parameter, but activations grow with batch size and sequence length. Long-context fine-tuning hits the activation wall first. Gradient checkpointing is the standard fix, it is on by default in most fine-tuning recipes, and "what does it cost?" is a classic interview follow-up.

## 1. Intuition

**What backward needs.** To compute the gradient of a linear layer $y = xW$ with respect to $W$, backward needs the input $x$. To push the gradient through a ReLU it needs to know which inputs were positive. So the forward pass keeps, for every layer, the intermediate tensors that backward will read. Autograd does this for you.

**Why it adds up.** Those saved tensors scale with tokens times hidden size, for every layer. A 7B-style model with 32 layers at 2,048 tokens keeps about 9 GB of them. At 8,192 tokens it is about 37 GB, more than the bf16 weights.

**The trade.** Don't keep everything. Keep only the input to each segment of layers (a *checkpoint*). During backward, when you reach a segment, run its forward again from the saved input to rebuild the intermediates, then do its backward and throw them away. Only one segment's intermediates exist at a time.

**The price.** Each layer's forward runs twice. A backward pass costs about twice a forward, so a step goes from $1 + 2 = 3$ units to $1 + 1 + 2 = 4$ units: about 33% more compute. The gradients are exactly the same.

**Smarter options.**
- *Selective recomputation* keeps the cheap-to-store, expensive-to-recompute tensors and recomputes only the ones that are big but cheap to redo (classically the attention scores). Korthikanti et al. (2022) report it recovers most of the memory for an overhead of a few percent.
- *FlashAttention* already recomputes attention scores tile by tile in its backward, so the $T \times T$ score matrix is never stored even without checkpointing.
- *Offloading* copies activations to CPU memory and brings them back for backward, trading PCIe bandwidth for recompute.

## 2. Visualization

![Left: activation memory for a synthetic 7B-style model at 2,048 tokens as a function of segment size k from 1 to 32. It starts at about 0.82 GB for k equals 1 and rises almost linearly to about 9.1 GB at k equals 32, the same as the no-checkpointing line at 9.13 GB. Right: activation memory against tokens per micro-batch from 1K to 32K. Without checkpointing it rises linearly from about 4.6 GB to 146 GB, crossing the 80 GB line between 16K and 32K tokens. With checkpointing every layer it rises from about 0.4 GB to 13 GB.](../../figures/gradient-checkpointing.png)

*Synthetic 7B-style shape (32 layers, hidden size 4096). Estimates from the rule of thumb in section 3, the same one the [LoRA lab](../15-llms/04-fine-tuning-in-practice.md) uses; real numbers depend on the kernels.*

**Try it** (predict first, then run `code/27-training-at-scale/gradient_checkpointing.py` and edit it):
1. In the toy network, predict the peak floats for `k = 8` (one segment covering all 8 layers) compared with full storage. Then check: why is it slightly *higher*?
2. Change `34` to `60` in `activation_bytes` (a model without fused attention kernels). Predict whether `k = 1` is still the best segment size.
3. Set `Lt = 80` and `D = 8192` (a 70B-style shape). Predict the no-checkpointing activation memory at 2,048 tokens before running.
4. Count the recomputed layer forwards for every `k`. Predict why it is always 8, even though peak memory changes.

## 3. The math

### Symbols

| Symbol | Meaning | Value in the example |
|---|---|---|
| $L$ | number of layers | 32 |
| $T$ | tokens per micro-batch (batch times sequence length) | 2,048 |
| $d$ | hidden size | 4,096 |
| $A$ | bytes of saved activations per token per hidden unit per layer | 34 (16-bit, fused attention) |
| $b$ | bytes per value of a stored layer input | 2 (bf16) |
| $k$ | layers per checkpoint segment | 1 |
| $F$ | compute of one forward pass | 1 unit |

### Memory

Without checkpointing every layer keeps its activations:

```math
M_{\text{none}} = L \cdot A \cdot T \cdot d
```

With a checkpoint every $k$ layers, we keep $\lceil L/k \rceil$ segment inputs of size $b \cdot T \cdot d$, and during backward one segment's full activations, $k \cdot A \cdot T \cdot d$:

```math
M_{\text{ckpt}}(k) = \left\lceil \frac{L}{k} \right\rceil b\,T d + k\,A\,T d
```

The $A = 34$ rule of thumb (Korthikanti et al. 2022, for a standard transformer layer with fused attention, ignoring the $T^2$ term) is rough; it varies with the MLP width and the kernels. The point is the ratio: a layer's input is $b = 2$ bytes per value, its full saved state about 17 times more.

**Minimizing over k.** If $b$ and $A$ were equal, the two terms would balance at $k = \sqrt{L}$: the classic result of Chen et al. (2016). Here $A$ is 17 times $b$, so the second term dominates and the best $k$ is small. For $L = 32$: $k = 1$ gives $32 \cdot 2 + 1 \cdot 34 = 98$ units of $T d$ bytes, $k = 2$ gives $16 \cdot 2 + 2 \cdot 34 = 100$, $k = 6$ gives $6 \cdot 2 + 6 \cdot 34 = 216$. Frameworks checkpoint every transformer block ($k = 1$) for this reason.

### Worked example: 7B-style model, 2,048 tokens

1. $T \cdot d = 2{,}048 \times 4{,}096 = 8{,}388{,}608$.
2. No checkpointing: $32 \times 34 \times 8{,}388{,}608 = 1{,}088 \times 8{,}388{,}608 = 9{,}126{,}805{,}504$ bytes $\approx 9.13$ GB.
3. Checkpointing every layer, stored inputs: $32 \times 2 \times 8{,}388{,}608 = 536{,}870{,}912$ bytes.
4. One layer's full activations during its backward: $34 \times 8{,}388{,}608 = 285{,}212{,}672$ bytes.
5. Total: $536{,}870{,}912 + 285{,}212{,}672 = 822{,}083{,}584$ bytes $\approx 0.822$ GB, about 11 times less.

This 0.822 GB is the activation term in the [fine-tuning lesson's](../15-llms/04-fine-tuning-in-practice.md) 108.63 GB full fine-tuning estimate.

At 8,192 tokens everything is 4 times bigger: $9.13 \times 4 = 36.5$ GB without checkpointing, $0.822 \times 4 = 3.29$ GB with it.

### Compute

Per layer, forward costs $F$ and backward about $2F$ (one matrix multiply for the input gradient, one for the weight gradient). With checkpointing, every layer's forward runs once more:

```math
\text{overhead} = \frac{F + F + 2F}{F + 2F} - 1 = \frac{4}{3} - 1 \approx 33\%
```

This is the same whatever $k$ is: every layer except possibly the last gets recomputed once. Segment size changes memory, not compute. (Some implementations skip recomputing the final segment, saving $k/L$ of the overhead.)

## 4. Implementation

### Build it from scratch

A residual MLP block stands in for a transformer layer. Its forward returns the output plus the cache backward needs: the input `x`, the pre-activation `h`, and the post-ReLU `a`.

```python
def layer_forward(x, w1, w2):
    """y = x + relu(x W1) W2   (a residual MLP block). Returns y and what backward needs."""
    h = x @ w1
    a = np.maximum(h, 0)
    return x + a @ w2, (x, h, a)

def layer_backward(gy, cache, w1, w2):
    x, h, a = cache
    gw2 = a.T @ gy
    gh = (gy @ w2.T) * (h > 0)
    gw1 = x.T @ gh
    gx = gy + gh @ w1.T          # residual path + MLP path
    return gx, (gw1, gw2)
```

*Predict:* which of `x`, `h`, `a` could backward rebuild from the others without re-running a matrix multiply?

The ordinary step keeps every layer's cache alive until backward reaches it:

```python
def backward_full(x, params, gout):
    caches, stored = [], 0
    for w1, w2 in params:
        x, cache = layer_forward(x, w1, w2)
        caches.append(cache)
        stored += sum(t.size for t in cache)       # every intermediate stays alive
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        gout, grads[i] = layer_backward(gout, caches[i], *params[i])
    return grads, stored, 0
```

*Predict:* doubling the batch doubles `stored`. What does doubling the number of layers do?

The checkpointed step keeps only the input of every `k`-th layer. In backward, it walks segments from last to first, re-runs each segment's forward from its saved input, and back-propagates through it.

```python
def backward_checkpointed(x, params, gout, k):
    L = len(params)
    boundaries = {}                                   # segment start index -> its input
    for i, (w1, w2) in enumerate(params):
        if i % k == 0:
            boundaries[i] = x                          # the only tensors kept from the forward pass
        x, _ = layer_forward(x, w1, w2)                # cache thrown away
    grads, recomputed, peak = [None] * L, 0, 0
    for start in reversed(range(0, L, k)):
        end = min(start + k, L)
        h, caches = boundaries[start], []
        for i in range(start, end):                    # re-run this segment's forward
            h, cache = layer_forward(h, *params[i])
            caches.append(cache)
            recomputed += 1
        live = sum(b.size for b in boundaries.values()) + sum(t.size for c in caches for t in c)
        peak = max(peak, live)
        for j in reversed(range(start, end)):
            gout, grads[j] = layer_backward(gout, caches[j - start], *params[j])
    return grads, peak, recomputed
```

*Predict:* the recomputed forward uses exactly the same weights and inputs. Will the gradients match the full version to the last bit, or only approximately?

### Use the library

In PyTorch, wrap each block's forward in `torch.utils.checkpoint.checkpoint`:

```python
from torch.utils.checkpoint import checkpoint

class Model(nn.Module):
    def forward(self, x):
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training else block(x)
        return x
```

Hugging Face models expose the same thing as `model.gradient_checkpointing_enable()` (or `gradient_checkpointing=True` in `TrainingArguments`). Megatron-Core offers `full` and `selective` recomputation, and PyTorch's FSDP2 and DeepSpeed can offload activations to CPU instead. `use_reentrant=False` is the recommended implementation; it supports keyword arguments and inputs that don't require gradients.

Watch out for randomness: dropout in the recomputed forward must use the same random mask. PyTorch's checkpoint saves and restores the RNG state by default (`preserve_rng_state=True`).

### Run it

```bash
uv run code/27-training-at-scale/gradient_checkpointing.py
```

Key lines:

```text
  full storage: 4,608 floats kept, 0 layers recomputed
  k = 1: peak 1,088 floats, 8 layer forwards recomputed, max grad diff 0.0e+00
  k = 2: peak 1,408 floats, 8 layer forwards recomputed, max grad diff 0.0e+00
  k = 8: peak 4,672 floats, 8 layer forwards recomputed, max grad diff 0.0e+00
  no checkpointing: 1 + 2 = 3 units; checkpointing: 1 + 1 + 2 = 4 units -> +33.3% per step
  no checkpointing: 32 x 34 x 2048 x 4096 = 9.127 GB
  checkpoint every  1 layers:  0.822 GB
  checkpoint every  8 layers:  2.349 GB
  at 8,192 tokens: none 36.51 GB, k = 1 3.29 GB
```

The gradient difference is exactly zero: recomputation is deterministic. With `k = 8` (one segment) the peak is slightly above full storage because the segment's input is stored on top of all the caches: checkpointing everything into one segment is all cost and no saving.

## 5. Engineering

**When to turn it on.** When activations, not weights or optimizer state, are what doesn't fit: long sequences, large micro-batches, or after you have already sharded the weights and optimizer (see [ZeRO and FSDP](05-zero-parameter-sharding.md)). Most fine-tuning recipes turn it on by default because sequence length is the usual limit.

**Spend the memory.** Checkpointing frees memory; use it for a bigger micro-batch or a longer context. A bigger micro-batch can win back part of the 33%, because larger matrix multiplies run the GPU more efficiently.

**Selective first.** If your framework supports selective recomputation, try it before full checkpointing: most of the memory saving for a few percent of compute. With FlashAttention the attention scores are already not stored, so the remaining activations are mostly linear-layer inputs and MLP intermediates.

**Offload when bandwidth is idle.** Activation offload to CPU costs no extra FLOPs but needs PCIe transfers to overlap with compute. It helps most when the GPU is memory-bound but the host link is underused.

**Interactions.** With pipeline parallelism, each stage stores activations for every micro-batch in flight, so checkpointing per stage is common. With tensor parallelism, recompute repeats the layer's communication too. With FP8 training, recomputation must reuse the same scaling factors as the original forward.

### Common mistakes

- Expecting checkpointing to reduce weight, gradient, or optimizer memory: it only touches activations.
- Checkpointing the whole model as one segment: you store the input plus everything else during backward, so you save nothing.
- Forgetting RNG state: if dropout masks differ between the forward and the recomputation, gradients are wrong. Keep `preserve_rng_state=True`.
- Leaving checkpointing on at inference or evaluation: there is no backward pass, so it only costs time. Guard it with `self.training`.
- Assuming the cost is free: plan for about a third more compute per step with full recomputation.

## 6. Knowledge check

<!-- quiz:gradient-checkpointing -->
**[Take the gradient checkpointing quiz](../../quizzes/gradient-checkpointing.md)**
<!-- /quiz -->

**Practice exercise.** A synthetic 13B-style model has $L = 40$ layers and $d = 5{,}120$. Using $A = 34$ and $b = 2$, compute activation memory at 4,096 tokens with no checkpointing and with checkpointing every layer.

<details>
<summary>Solution</summary>

$T d = 4{,}096 \times 5{,}120 = 20{,}971{,}520$.
No checkpointing: $40 \times 34 \times 20{,}971{,}520 = 1{,}360 \times 20{,}971{,}520 = 28{,}521{,}267{,}200$ bytes $\approx 28.5$ GB.
Every layer: $40 \times 2 \times 20{,}971{,}520 + 34 \times 20{,}971{,}520 = (80 + 34) \times 20{,}971{,}520 = 114 \times 20{,}971{,}520 = 2{,}390{,}753{,}280$ bytes $\approx 2.39$ GB. That is $1{,}360 / 114 \approx 11.9$ times less.

</details>

**Implementation challenge.** Add an `offload` variant to the script: in the forward pass, put each layer's cache in a separate list called `cpu` (standing in for host memory) and count the floats "transferred". In backward, take caches back from `cpu`. Report peak GPU floats and floats transferred, and compare with `k = 1` checkpointing.

<details>
<summary>Solution</summary>

```python
def backward_offload(x, params, gout):
    cpu, moved = [], 0
    for w1, w2 in params:
        x, cache = layer_forward(x, w1, w2)
        cpu.append(cache)                              # "copy to host"
        moved += sum(t.size for t in cache)
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        cache = cpu.pop()                              # "copy back"
        moved += sum(t.size for t in cache)
        gout, grads[i] = layer_backward(gout, cache, *params[i])
    return grads, moved
```

GPU peak is about one layer's cache (plus the in-flight tensor), lower than checkpointing, and no forward is recomputed. The cost is moving every cache twice: $2 \times 4{,}608 = 9{,}216$ floats here. In practice it pays off only if those copies overlap with compute.

</details>

## Summary

- Backward needs each layer's saved intermediates, so activation memory scales with layers × tokens × hidden size: about 9.13 GB for a 7B-style model at 2,048 tokens.
- Checkpointing keeps only segment inputs and recomputes each segment's forward during backward; gradients are identical.
- Checkpointing every layer brings that 9.13 GB down to 0.822 GB, for about 33% more compute per step whatever the segment size.
- The $\sqrt{L}$ rule assumes stored inputs cost as much as full activations; in transformers they are about 17 times cheaper, so per-block checkpointing wins.
- Selective recomputation, FlashAttention, and offloading get most of the saving for less compute.

**Next:** [Collective operations from scratch](03-collective-ops-from-scratch.md)

**Related:** [Backpropagation](../11-gradient-descent-backprop/02-backpropagation.md) · [Fine-tuning in practice](../15-llms/04-fine-tuning-in-practice.md) · [ZeRO and FSDP](05-zero-parameter-sharding.md)

*Go deeper, code-first: [AI Engineering from Scratch, gradient checkpointing](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/34-gradient-checkpointing).*

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Gradient Checkpointing and Activation Recomputation](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/34-gradient-checkpointing).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>What does gradient checkpointing save, and what does it cost?</strong></summary>

It saves activation memory only: the intermediates the forward pass keeps for backward. Instead of keeping them for every layer, you keep the input to each checkpointed block and, during backward, re-run that block's forward to rebuild them. For a 7B-style model at 2,048 tokens that takes activations from about 9.1 GB to about 0.8 GB, and the saving grows linearly with sequence length. The cost is compute: each block's forward runs twice. With backward costing about two forwards, a step goes from 3 units to 4, roughly 33% more, regardless of segment size. Weights, gradients and optimizer state are untouched, so if those don't fit, you need sharding or LoRA instead. Gradients are bit-for-bit identical as long as recomputation is deterministic, which is why PyTorch restores the RNG state for dropout.

</details>

<details>
<summary><strong>The classic advice is a checkpoint every square root of L layers. Why do frameworks checkpoint every block instead?</strong></summary>

The square-root rule minimizes memory when a stored checkpoint costs as much as one layer's full activations: you keep L/k checkpoints plus k layers during a segment's backward, and L/k + k is smallest at k = sqrt(L). In a transformer the two are very different. A block's input is one tensor of tokens × hidden size, about 2 bytes per value, while everything the block saves for backward (attention inputs, projections, MLP intermediates) is roughly 34 bytes per token per hidden unit. With stored inputs about 17 times cheaper, the k-layer term dominates and the optimum is k = 1. For 32 layers, k = 1 needs 98 units, k = 6 needs 216. Since compute is the same for any k (every block is recomputed once), checkpointing every block is the natural default.

</details>

<details>
<summary><strong>You turned on checkpointing and loss curves no longer match the run without it. What do you check?</strong></summary>

Checkpointing should give identical gradients, so a mismatch means the recomputed forward differs from the original. The most common cause is randomness: dropout or stochastic depth draws a different mask on recompute. I check that `preserve_rng_state` is on, or that no custom code reseeds the generator. Next, anything stateful in the forward: batch-norm running statistics updated twice, a counter or cache incremented in forward, or a module that behaves differently on the second call. Third, non-deterministic kernels: some fused or atomic GPU operations give slightly different results each run, which is usually harmless noise, not a trend. To isolate it, I run one step both ways with dropout off and compare gradients tensor by tensor; the first block with a mismatch is the culprit. With the reentrant implementation, also check that inputs require gradients, or some parameters silently get none.

</details>

<details>
<summary><strong>Quick calculation: how much activation memory does a 70B-style model need at 2,048 tokens?</strong></summary>

Use the rule of thumb of about 34 bytes per token per hidden unit per layer in 16-bit with fused attention. A 70B-style model has 80 layers and hidden size 8,192. Tokens times hidden size is 2,048 × 8,192, about 16.8 million. Without checkpointing: 80 × 34 × 16.8 million, about 45.6 GB per micro-batch. With checkpointing every block: stored inputs are 80 × 2 × 16.8 million, about 2.7 GB, plus one block's activations, 34 × 16.8 million, about 0.57 GB, so roughly 3.3 GB. That is about 14 times less, for one extra forward per step. At 8,192 tokens multiply by 4: about 182 GB without, 13 GB with. Either way it is small next to the roughly 1.1 TB of weights, gradients and Adam state, which is why checkpointing is combined with sharding.

</details>
