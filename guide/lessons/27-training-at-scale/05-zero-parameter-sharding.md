---
title: ZeRO and FSDP
summary: Shard optimizer state, gradients, and weights across data-parallel GPUs; compute per-GPU memory and communication for DDP and ZeRO stages 1 to 3, and pick a stage.
skill: deep-learning
minutes: 40
prerequisites: [scaling-distributed, collective-ops-from-scratch, fine-tuning-in-practice]
related: [gradient-checkpointing, gradient-clipping-amp, llm-inference]
adapted_from: [phases/19-capstone-projects/78-zero-parameter-sharding]
---

# ZeRO and FSDP

> **Mental model.** Eight people study from the same 1,000-page textbook, and each carries a full copy plus full notes. ZeRO says: everyone still reads the whole book, but each person keeps only 125 pages of the notes (stage 1), then only their pages of the homework too (stage 2), and finally only their 125 pages of the book itself, borrowing the rest the moment they need it (stage 3).

**You will learn to**
- Write per-GPU memory for DDP and ZeRO stages 1, 2, and 3 as a formula in parameters $P$ and GPUs $N$.
- Implement ZeRO-1 from scratch with reduce-scatter and all-gather, and show it produces the same weights as DDP.
- Explain why ZeRO-1 and ZeRO-2 send the same bytes as DDP while ZeRO-3 sends about 1.5 times as much.
- Choose a stage (and FSDP's equivalent setting) for a model, a GPU count, and a network.

**Why it matters.** Data parallelism wastes memory: $N$ GPUs hold $N$ identical copies of the weights, gradients, and optimizer state. ZeRO (Rajbhandari et al., 2020) removes the duplicates, and PyTorch FSDP is the same idea built into PyTorch. It is how most teams train or fully fine-tune models from a few billion to tens of billions of parameters, and the memory formulas are a staple of ML-systems interviews.

## 1. Intuition

Start from the [16 bytes per parameter](../15-llms/04-fine-tuning-in-practice.md) of mixed-precision AdamW: 2 for bf16 weights, 2 for bf16 gradients, 12 for optimizer state (an fp32 master copy plus Adam's $m$ and $v$). Under plain data parallelism (DDP) every GPU holds all 16.

**Stage 1: shard the optimizer state.** The optimizer state is the biggest piece (12 of 16 bytes) and is only used in the optimizer step, not in forward or backward. So split it: GPU $r$ keeps the Adam state for slice $r$ of the parameters only. After backward, instead of all-reducing the gradients, **reduce-scatter** them, so GPU $r$ receives only the summed gradient for slice $r$. It updates slice $r$ and then **all-gathers** the updated weights so every GPU again has the full model for the next forward.

**Stage 2: shard the gradients too.** GPU $r$ only ever needs gradient slice $r$, so it can free the rest as soon as the reduce-scatter is done. Same communication as stage 1.

**Stage 3: shard the weights.** Each GPU stores only its slice of the weights. Before computing a layer, the GPUs all-gather that layer's weights, use them, and free them. Backward gathers them again. Memory per GPU now falls as $1/N$ for everything, but weights cross the network twice per step instead of once. PyTorch calls this **FSDP** (fully sharded data parallel).

**What sharding does not touch.** Activations are not sharded by ZeRO: each GPU still processes its own micro-batch. Combine ZeRO with [gradient checkpointing](02-gradient-checkpointing.md) when activations are the problem.

## 2. Visualization

<!-- lab:zero-sharding -->
![Left: stacked horizontal bars of estimated memory per GPU for a synthetic 7B-style model on 8 GPUs with bf16 weights and AdamW. DDP is 108.6 GB, mostly optimizer state, past the 80 GB line. ZeRO-1 is 37.9 GB, ZeRO-2 26.1 GB, and ZeRO-3 14.3 GB. Right: memory per GPU on a log scale against number of GPUs from 1 to 256. DDP stays flat at about 109 GB. ZeRO-1 falls and levels off near 28 GB, ZeRO-2 levels off near 15 GB, and ZeRO-3 keeps falling to about 1.2 GB at 256 GPUs.](../../figures/zero-parameter-sharding.png)

*Synthetic 7B-style model (6.74 billion parameters), 2,048 tokens per GPU with gradient checkpointing. These are arithmetic estimates of training state; real jobs add temporary buffers, fragmentation, and framework overhead.*

*Interactive version: [open the lab on the website](https://sreshtalluri.github.io/ml-ai-learn/labs/zero-sharding/).*
<!-- /lab -->

**Try it** (predict first, then check):
1. On the default 7B-style model with 8 GPUs, predict which strategy first fits under 80 GB. Then switch the GPU memory to 24 GB and predict again.
2. Choose ZeRO-1 and slide the GPU count from 8 to 256. Predict where the bar stops shrinking and which colours make up the floor.
3. Pick the 70B-style model and ZeRO-3. Predict the fewest GPUs that fit 80 GB, then read it in the readout.
4. Switch precision from bf16 + AdamW to fp32 + AdamW. Predict whether the optimizer segment grows or shrinks, and why the total still grows.

## 3. The math

### Symbols

| Symbol | Meaning | Value used |
|---|---|---|
| $P$ | number of parameters | $6{,}738{,}149{,}376$ (7B-style) |
| $N$ | number of data-parallel GPUs | 8 |
| $\Psi_w, \Psi_g, \Psi_o$ | bytes per parameter for weights, gradients, optimizer state | 2, 2, 12 |
| $M$ | training-state memory per GPU | GB |
| $C$ | bytes each GPU sends per step | GB |

### Memory per GPU

```math
\begin{aligned}
M_{\text{DDP}} &= (\Psi_w + \Psi_g + \Psi_o)\,P = 16P \\
M_{\text{ZeRO-1}} &= (\Psi_w + \Psi_g)\,P + \frac{\Psi_o P}{N} = 4P + \frac{12P}{N} \\
M_{\text{ZeRO-2}} &= \Psi_w P + \frac{(\Psi_g + \Psi_o)P}{N} = 2P + \frac{14P}{N} \\
M_{\text{ZeRO-3}} &= \frac{(\Psi_w + \Psi_g + \Psi_o)P}{N} = \frac{16P}{N}
\end{aligned}
```

Activations (not sharded) come on top. As $N \to \infty$, ZeRO-1 tends to $4P$ and ZeRO-2 to $2P$: replicated pieces set a floor. Only ZeRO-3 keeps falling.

### Worked example: 7B-style model on 8 GPUs

$P = 6{,}738{,}149{,}376$, so $2P = 13.48$ GB (decimal GB, rounded).

1. **DDP:** $16P = 8 \times 13.48 = 107.81$ GB. Plus 0.822 GB activations: **108.63 GB**. Matches the [fine-tuning lesson](../15-llms/04-fine-tuning-in-practice.md).
2. **ZeRO-1:** weights $2P = 13.48$, gradients $13.48$, optimizer $12P/8 = 80.86/8 = 10.11$. Sum $13.48 + 13.48 + 10.11 = 37.06$, plus 0.822: **37.88 GB**.
3. **ZeRO-2:** weights $13.48$, gradients $13.48/8 = 1.68$, optimizer $10.11$. Sum $25.27$, plus 0.822: **26.09 GB**.
4. **ZeRO-3:** $107.81 / 8 = 13.48$, plus 0.822: **14.30 GB**.

So ZeRO-1 alone takes the job from "does not fit an 80 GB GPU" to "fits with room for a bigger micro-batch", a 65% drop.

### Communication per GPU per step

A ring all-reduce of $T$ bytes sends $2\frac{N-1}{N}T$ per GPU; reduce-scatter and all-gather each send $\frac{N-1}{N}T$ ([collective operations](03-collective-ops-from-scratch.md)). With bf16 weights and gradients, $T = 2P$ for either.

| Strategy | Collectives per step | Bytes sent per GPU | Relative to DDP |
|---|---|---|---|
| DDP | all-reduce gradients | $2\frac{N-1}{N} \cdot 2P$ | 1 |
| ZeRO-1, ZeRO-2 | reduce-scatter gradients + all-gather updated weights | $\frac{N-1}{N} \cdot 2P + \frac{N-1}{N} \cdot 2P$ | 1 |
| ZeRO-3 / FSDP | all-gather weights (forward) + all-gather weights (backward) + reduce-scatter gradients | $3 \cdot \frac{N-1}{N} \cdot 2P$ | 1.5 |

ZeRO-1 and ZeRO-2 do exactly what a ring all-reduce does internally, a reduce-scatter and an all-gather; the only change is that the all-gather carries updated weights instead of summed gradients. For the 7B-style model on 8 GPUs: DDP sends $1.75 \times 13.48 = 23.58$ GB per GPU per step, and ZeRO-3 sends $1.5 \times 23.58 = 35.38$ GB.

### Worked example: when does a 70B-style model fit?

$P = 68{,}975{,}329{,}280$ (from the [previous lesson](04-scaling-distributed.md)), so $16P = 1{,}103.61$ GB; activations at 2,048 tokens with checkpointing are 3.255 GB.

- ZeRO-3 on 8 GPUs: $1{,}103.61 / 8 = 137.95$, plus 3.255 = 141.21 GB. Does not fit.
- On 16 GPUs: $1{,}103.61 / 16 = 68.98$, plus 3.255 = **72.23 GB**. Fits 80 GB.
- ZeRO-1 can never fit: its floor is $4P = 275.9$ GB.

## 4. Implementation

### Build it from scratch

We simulate $N$ ranks in one process and train a tiny linear regression with Adam both ways. Adam is elementwise, so it can update any slice of the parameters independently: that is the property ZeRO relies on.

```python
def adam(p, g, m, v, t, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam step on any slice of parameters. Elementwise, so it works on a shard."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    p = p - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return p, m, v
```

*Predict:* would ZeRO-1 still be exact with an optimizer that normalizes by the norm of the *whole* gradient, like LAMB?

The two collectives. Reduce-scatter gives rank $r$ the average of chunk $r$ across ranks; all-gather hands every rank every chunk.

```python
def reduce_scatter_mean(grads, n):
    """Rank r receives the average over ranks of chunk r only."""
    chunks = [np.array_split(g, n) for g in grads]
    return [np.mean([chunks[src][r] for src in range(n)], axis=0) for r in range(n)]

def all_gather(shards):
    """Every rank receives every shard, concatenated in rank order."""
    full = np.concatenate(shards)
    return [full.copy() for _ in shards]
```

*Predict:* how many floats does each rank receive from `reduce_scatter_mean` for $P = 10$ and $N = 4$?

The ZeRO-1 step. Each rank allocates Adam state only for its slice. After the local backward, reduce-scatter, update the slice, all-gather.

```python
        if zero:
            g_shard = reduce_scatter_mean(grads, n_ranks)       # comm half 1
            new = []
            for r in range(n_ranks):
                w_shard = np.array_split(w[r], n_ranks)[r]
                p, m[r], v[r] = adam(w_shard, g_shard[r], m[r], v[r], t)
                new.append(p)
            w = all_gather(new)                                 # comm half 2
        else:
            g = np.mean(grads, axis=0)                          # all-reduce (average)
            for r in range(n_ranks):
                w[r], m[r], v[r] = adam(w[r], g, m[r], v[r], t)
```

*Predict:* after 50 steps, how different are the DDP and ZeRO-1 weights?

The memory formulas are a few lines. `k` counts how many of (optimizer, gradients, weights) the strategy shards:

```python
def per_gpu(c, strategy, n, precision="bf16 mixed + AdamW", tokens=2048):
    P = n_params(c)
    bw, bg, bo = PRECISION[precision]
    k = STRATEGIES.index(strategy)          # 0 = DDP, then how many of (optimizer, grads, weights) are sharded
    return dict(
        weights=bw * P / (n if k >= 3 else 1),
        grads=bg * P / (n if k >= 2 else 1),
        optimizer=bo * P / (n if k >= 1 else 1),
        activations=activation_bytes(c, tokens, 4 if precision.startswith("fp32") else 2),
    )
```

*Predict:* with fp32 weights and AdamW (no separate master copy, so 4 + 4 + 8 bytes), what is ZeRO-1's floor per parameter?

### Use the library

PyTorch FSDP2 shards per module; each `fully_shard` call becomes one all-gather unit:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.layers:
    fully_shard(block, mp_policy=mp)                       # ZeRO-3 per transformer block
fully_shard(model, mp_policy=mp, reshard_after_forward=True)
```

`reshard_after_forward=True` frees gathered weights after forward (ZeRO-3); `False` keeps them until backward finishes, close to ZeRO-2. The older `FullyShardedDataParallel` wrapper names these `ShardingStrategy.FULL_SHARD`, `SHARD_GRAD_OP`, and `HYBRID_SHARD` (shard within a node, replicate across nodes). DeepSpeed selects the stage in its config:

```json
{ "zero_optimization": { "stage": 2, "overlap_comm": true }, "bf16": { "enabled": true } }
```

DeepSpeed also offers ZeRO-Offload and ZeRO-Infinity, which move optimizer state (and optionally weights) to CPU memory or NVMe, trading speed for capacity.

### Run it

```bash
uv run code/27-training-at-scale/zero_sharding.py
```

Key lines:

```text
  max |w_DDP - w_ZeRO1| after 50 steps = 0.00e+00
  Adam state floats across all ranks: DDP 80, ZeRO-1 20  (P = 10, N = 4)
 N = 8 GPUs, bf16 mixed precision + AdamW (16 bytes/param)
  strategy  weights    grads    optim   activ  total GB  comm GB vs DDP
  DDP         13.48    13.48    80.86   0.822    108.63    23.58   1.00x
  ZeRO-1      13.48    13.48    10.11   0.822     37.88    23.58   1.00x
  ZeRO-2      13.48     1.68    10.11   0.822     26.09    23.58   1.00x
  ZeRO-3       1.68     1.68    10.11   0.822     14.30    35.38   1.50x
  ZeRO-3 on  16 GPUs:  72.23 GB per GPU  fits in 80 GB
```

The weights match DDP exactly, and the Adam state across all ranks shrinks from $N \times 2P$ floats to $2P$.

## 5. Engineering

**Picking a stage.**
- Fits with DDP? Use DDP.
- Fits with ZeRO-1 or ZeRO-2? Use them: same communication as DDP, so usually no slowdown. ZeRO-2 is a common default for fine-tuning 7B to 13B models on one 8-GPU server.
- Otherwise ZeRO-3 / FSDP. Budget for 1.5 times DDP's traffic, and keep it overlapped with compute (prefetch the next layer's all-gather while the current layer computes).

**Hybrid sharding.** Inter-node links are slower than NVLink. Sharding within a node (or a group of nodes) and replicating across groups keeps most traffic on fast links, at the cost of a higher memory floor. Choose the group size from the memory formula: the shard group must be big enough for $16P / N_{\text{group}}$ to fit.

**Wrap granularity.** In FSDP, each wrapped unit is gathered whole. Wrap per transformer block: wrapping the whole model as one unit gathers every weight at once and loses the memory benefit; wrapping tiny modules creates many small, latency-bound collectives.

**Peak memory is higher than the formula.** ZeRO-3 temporarily holds a full gathered layer (or two, with prefetch), communication buffers, and the allocator's fragmentation. Leave 10 to 20% headroom and measure with `torch.cuda.max_memory_allocated`.

**Checkpoints are sharded too.** Each rank saves its own slice of optimizer state (and weights, under ZeRO-3). Use the framework's distributed checkpoint format (`torch.distributed.checkpoint`, DeepSpeed's), which records which rank owns what and can reshard when you resume on a different number of GPUs.

**Gradient clipping needs a global norm.** Under ZeRO-2 and 3 no rank has the whole gradient. Each rank computes the sum of squares of its slice, an all-reduce adds them, and every rank applies the same factor. FSDP's `clip_grad_norm_` does this; a per-rank norm would give each slice a different factor.

### Common mistakes

- Expecting ZeRO-1 or ZeRO-2 memory to keep falling with more GPUs: the replicated weights and gradients are a $4P$ or $2P$ floor.
- Expecting ZeRO to shrink activations: it does not. Use checkpointing or smaller micro-batches.
- Writing ZeRO-1's memory as $2P + 2P + 8P/N$ (forgetting the fp32 master copy): it is $4P + 12P/N$.
- Saving rank 0's optimizer state only and calling it a checkpoint: the other ranks' slices are lost.
- Running ZeRO-3 across slow inter-node links with no prefetch or hybrid sharding: GPUs wait on all-gathers.
- Clipping by a local per-rank norm under sharded gradients.

## 6. Knowledge check

<!-- quiz:zero-parameter-sharding -->
**[Take the ZeRO and FSDP quiz](../../quizzes/zero-parameter-sharding.md)**
<!-- /quiz -->

**Practice exercise.** A synthetic 13B-style model has $P = 13 \times 10^9$ parameters, trained with bf16 + AdamW on $N = 16$ GPUs. Compute per-GPU training state (no activations) for DDP, ZeRO-1, ZeRO-2, and ZeRO-3, and say which fit 24 GB GPUs.

<details>
<summary>Solution</summary>

$P = 13 \times 10^9$, so $1P = 13$ GB.
- DDP: $16 \times 13 = 208$ GB.
- ZeRO-1: $4 \times 13 + 12 \times 13 / 16 = 52 + 9.75 = 61.75$ GB.
- ZeRO-2: $2 \times 13 + 14 \times 13 / 16 = 26 + 11.375 = 37.375$ GB.
- ZeRO-3: $208 / 16 = 13$ GB.

Only ZeRO-3 fits a 24 GB GPU, with about 11 GB left for activations and buffers.

</details>

**Implementation challenge.** Extend the script's `train` function with a `stage=2` mode in which each rank keeps only its own gradient slice after the reduce-scatter (free the rest), and count the gradient floats held per rank. Then add global-norm gradient clipping that works on sharded gradients.

<details>
<summary>Solution</summary>

```python
g_shard = reduce_scatter_mean(grads, n_ranks)
grads = None                                             # ZeRO-2: full gradients freed
held = [len(s) for s in g_shard]                         # P/N floats per rank
sq = sum(float(np.sum(s * s)) for s in g_shard)          # all-reduce of local sums of squares
scale = min(1.0, max_norm / (np.sqrt(sq) + 1e-6))
g_shard = [s * scale for s in g_shard]                   # same factor on every rank
```

The weights still match DDP (with the same clipping applied to DDP's full gradient), since clipping by the global norm multiplies every element by the same factor. Each rank holds $P/N$ gradient floats instead of $P$.

</details>

## Summary

- DDP stores $16P$ bytes on every GPU. ZeRO-1 shards the 12 bytes of optimizer state ($4P + 12P/N$), ZeRO-2 adds gradients ($2P + 14P/N$), ZeRO-3 / FSDP adds weights ($16P/N$).
- A 7B-style model on 8 GPUs: 108.63, 37.88, 26.09, and 14.30 GB per GPU including activations.
- ZeRO-1 and ZeRO-2 send the same bytes as DDP (reduce-scatter + all-gather = all-reduce); ZeRO-3 sends about 1.5 times as much because weights are gathered for forward and again for backward.
- Only ZeRO-3 keeps shrinking with more GPUs; a 70B-style model first fits 80 GB cards at 16 GPUs.
- Activations are not sharded; pair ZeRO with gradient checkpointing, sharded checkpoints, and global-norm clipping.

**Next:** [Self-attention](../14-transformers/01-self-attention.md)

**Related:** [Scaling and distributed training](04-scaling-distributed.md) · [Gradient checkpointing](02-gradient-checkpointing.md) · [LLM inference](../18-llm-inference/01-llm-inference.md)

*Go deeper, code-first: [AI Engineering from Scratch, ZeRO stage 1 over real processes](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/78-zero-parameter-sharding).*

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [ZeRO Optimizer State Sharding](https://aiengineeringfromscratch.com/lesson?path=phases/19-capstone-projects/78-zero-parameter-sharding).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Explain ZeRO stages 1, 2 and 3. What does each shard, and what does it cost?</strong></summary>

Plain data parallelism keeps a full copy of weights, gradients and optimizer state on every GPU: 16 bytes per parameter with mixed-precision Adam. ZeRO removes the duplicates in three steps. Stage 1 shards the optimizer state, the fp32 master copy and Adam moments, 12 of the 16 bytes, giving 4P + 12P/N per GPU. After backward it reduce-scatters gradients so each GPU gets only its slice, updates that slice, and all-gathers the new weights. Stage 2 also frees the gradients outside each GPU's slice: 2P + 14P/N. Both send the same bytes as DDP, because a ring all-reduce is a reduce-scatter plus an all-gather. Stage 3, which PyTorch calls FSDP, shards the weights too: 16P/N. Each layer's weights are gathered before forward and again before backward, so communication is about 1.5 times DDP.

</details>

<details>
<summary><strong>Why does ZeRO-1 cost no extra communication compared with DDP?</strong></summary>

Because DDP's all-reduce is already two halves. A ring all-reduce of the gradients is a reduce-scatter, after which each GPU holds the summed gradient for one slice, followed by an all-gather that shares those summed slices. Each half sends (N − 1)/N of the tensor per GPU. ZeRO-1 does the same reduce-scatter, then runs the optimizer step on that slice, using the optimizer state only that GPU keeps, and then all-gathers the updated weights instead of the summed gradients. With bf16 weights and bf16 gradients, both all-gathers move 2 bytes per parameter, so total traffic is identical. What changes is only what is stored. That is why ZeRO-1 is close to a free win whenever optimizer state is the memory problem, which it usually is: 12 of 16 bytes.

</details>

<details>
<summary><strong>Your FSDP job runs out of memory even though the formula says 30 GB per GPU on an 80 GB card. What do you check?</strong></summary>

The formula covers sharded training state and a rough activation estimate, not peaks. First, wrapping: if the model is wrapped as one FSDP unit, the whole model is gathered at once and the 16P/N saving disappears during forward and backward; wrap each transformer block. Second, activations: they are not sharded, so check micro-batch size, sequence length and whether gradient checkpointing is actually on. Third, temporary buffers: prefetching gathers the next layer while the current one runs, and mixed-precision casting keeps both bf16 and fp32 copies briefly. Fourth, layers that are not wrapped, such as a large embedding or output head that stays replicated. Fifth, fragmentation: compare memory allocated with memory reserved; setting expandable segments in the CUDA allocator config often helps. I would profile one step with the memory snapshot tool to see the exact peak.

</details>

<details>
<summary><strong>Quick calculation: how many 80 GB GPUs does a 70B model need under ZeRO-3, and could ZeRO-1 ever work?</strong></summary>

A 70B-style model has about 69 billion parameters. Mixed-precision AdamW is 16 bytes each, about 1,104 GB of training state. Under ZeRO-3 each GPU holds 1,104 / N, plus activations, about 3.3 GB at 2,048 tokens with checkpointing. With 8 GPUs that is 138 + 3.3, about 141 GB: too big. With 16 GPUs it is 69 + 3.3, about 72 GB, which fits an 80 GB card with a little headroom, though in practice I would use 32 for buffers and a larger micro-batch. ZeRO-1 can never fit: it shards only the optimizer state, so the replicated bf16 weights and gradients stay at 4 bytes per parameter, about 276 GB per GPU, however many GPUs you add. ZeRO-2's floor is 2 bytes per parameter, about 138 GB, also too big.

</details>
