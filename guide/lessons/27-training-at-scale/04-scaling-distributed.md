---
title: Scaling and distributed training
summary: Size training memory per GPU, and choose between data, tensor, pipeline, and sharded data parallelism (and their 3D combination) for a model and cluster.
skill: deep-learning
minutes: 45
prerequisites: [collective-ops-from-scratch, gradient-checkpointing, fine-tuning-in-practice]
related: [zero-parameter-sharding, transformer-architecture, llm-inference, serving-llms]
adapted_from: [phases/10-llms-from-scratch/05-scaling-distributed]
---

# Scaling and distributed training

> **Mental model.** A kitchen with more cooks. Data parallelism gives every cook the whole recipe and a different pile of ingredients. Tensor parallelism has several cooks chop the same onion together. Pipeline parallelism is an assembly line, each cook doing one stage. Sharding means no cook keeps the whole recipe book, only their pages, and they pass pages around as needed.

**You will learn to**
- Compute training memory per GPU from bytes per parameter, and decide whether a model fits.
- Explain data parallelism and show that averaging shard gradients equals the full-batch gradient.
- Split a matrix multiply by columns or by rows across GPUs, and explain why a transformer MLP needs only one all-reduce.
- Compute the pipeline bubble of a GPipe schedule and how micro-batches shrink it.
- Combine tensor, pipeline, and data parallelism (3D parallelism) for a large cluster.

**Why it matters.** A 7B model's training state is over 100 GB; a 70B model's is over 1 TB. No single GPU holds that. Every LLM you use was trained with some mix of these strategies, and "how would you train a 70B model on 64 GPUs?" is a standard ML-systems interview question. The answer starts with arithmetic.

## 1. Intuition

**Memory first.** Training keeps four things on the GPU: the weights, a gradient for each weight, the optimizer's state, and the activations from the forward pass. With mixed-precision AdamW the first three cost **16 bytes per parameter** (see [fine-tuning in practice](../15-llms/04-fine-tuning-in-practice.md)): 2 for bf16 weights, 2 for bf16 gradients, and 12 for the fp32 master copy and Adam's two moments. Activations depend on tokens and are controlled with [checkpointing](02-gradient-checkpointing.md).

**Data parallelism (DP).** Copy the whole model to every GPU and give each a different slice of the batch. After backward, [all-reduce](03-collective-ops-from-scratch.md) the gradients so every copy applies the same update. Throughput scales with the number of GPUs, but memory per GPU does not drop at all: each still holds all 16 bytes per parameter.

**Tensor parallelism (TP).** Split each weight matrix across GPUs so each does part of every matrix multiply. Memory per GPU drops, but the GPUs must exchange partial results inside every layer, so TP needs very fast links and usually stays within one server (8 GPUs on NVLink).

**Pipeline parallelism (PP).** Give each GPU a contiguous block of layers. Activations flow forward from stage to stage and gradients flow back. Only the boundary activations cross between GPUs, so PP works across servers. The catch is the **bubble**: at the start and end of each step, some stages have nothing to do. Splitting the batch into many micro-batches keeps the pipe fuller.

**Sharded data parallelism (ZeRO, FSDP).** Keep data parallelism's simplicity but stop storing duplicates: each GPU keeps only its slice of the optimizer state, gradients, and even weights, and gathers what it needs when it needs it. The [next lesson](05-zero-parameter-sharding.md) works through it.

**3D parallelism.** Large runs combine them: TP inside a server, PP across a few servers, and DP (often sharded) across the rest. Llama 3 405B was pre-trained with 8-way TP, 16-way PP, and data parallelism on up to 16,384 H100 GPUs (Llama 3 paper, 2024).

## 2. Visualization

![Left: a GPipe schedule for 4 pipeline stages and 4 micro-batches. Forward passes F1 to F4 (blue, 1 tick each) start one tick later on each stage, forming a staircase; backward passes B1 to B4 (orange, 2 ticks each) run in reverse stage order after all forwards finish. White gaps where a stage waits make up 43% of the 21-tick schedule. Right: bubble fraction against number of micro-batches from 1 to 64 for 2, 4, 8, and 16 stages. Every curve falls as micro-batches increase; with 4 stages it goes from 75% at 1 micro-batch to about 4% at 64; with 16 stages it is still about 19% at 64.](../../figures/scaling-distributed.png)

*Idealized schedule: forward 1 tick, backward 2 ticks per stage, no communication time. Synthetic.*

For memory per GPU under DDP and the ZeRO stages, use the lab in the [ZeRO lesson](05-zero-parameter-sharding.md).

**Try it** (predict first, then run `code/27-training-at-scale/scaling_distributed.py` and edit it):
1. Predict the bubble for 8 stages and 8 micro-batches, then add `(8, 8)` to the pipeline loop and check.
2. The script shows that unequal shards (60 examples on 8 GPUs) break the "average of gradients = full gradient" identity. Predict how to fix it with a weighted average, then implement it.
3. In section 4, predict the smallest TP × PP split of the 70B-style model that fits 80 GB before activations.
4. In `tp_mlp`, swap the order: row-split $W_1$ and column-split $W_2$. Predict whether the result is still exact without an extra all-reduce.

## 3. The math

### Symbols

| Symbol | Meaning | Value used |
|---|---|---|
| $P$ | number of parameters | $6.74 \times 10^9$ (7B-style) |
| $B_p$ | bytes per parameter of training state | 16 |
| $N$ | number of GPUs | |
| $t, p, d$ | tensor-, pipeline-, data-parallel degrees ($N = t \cdot p \cdot d$) | 8, 16, 128 |
| $m$ | micro-batches per step | |
| $X, W$ | layer input $[b, d_{\text{in}}]$ and weight $[d_{\text{in}}, d_{\text{out}}]$ | |

### Memory per GPU

Without sharding, and with weights split $t \cdot p$ ways by TP and PP:

```math
M_{\text{state}} = \frac{B_p \, P}{t \cdot p}, \qquad B_p = \underbrace{2}_{\text{bf16 } W} + \underbrace{2}_{\text{bf16 } \nabla W} + \underbrace{4 + 4 + 4}_{\text{fp32 master}, m, v} = 16
```

**Worked example, 7B-style.** $P = 6{,}738{,}149{,}376$ (the [fine-tuning lesson's](../15-llms/04-fine-tuning-in-practice.md) synthetic config).
1. $16 \times 6{,}738{,}149{,}376 = 107{,}810{,}390{,}016$ bytes $= 107.81$ GB.
2. Add 0.822 GB of activations at 2,048 tokens with checkpointing: $108.63$ GB. It does not fit an 80 GB GPU, and plain DP on 8 GPUs leaves it at 108.63 GB per GPU.
3. With $t = 8$: $107.81 / 8 = 13.48$ GB per GPU.

**Worked example, 70B-style** (synthetic: $d = 8{,}192$, 80 layers, MLP width 28,672, grouped-query attention with KV width 1,024, vocabulary 32,000).
1. Attention per layer: $q$ and $o$ are $8{,}192^2 = 67{,}108{,}864$ each; $k$ and $v$ are $8{,}192 \times 1{,}024 = 8{,}388{,}608$ each. Sum: $2 \times 67{,}108{,}864 + 2 \times 8{,}388{,}608 = 150{,}994{,}944$.
2. MLP per layer: $3 \times 8{,}192 \times 28{,}672 = 3 \times 234{,}881{,}024 = 704{,}643{,}072$.
3. Per layer: $150{,}994{,}944 + 704{,}643{,}072 = 855{,}638{,}016$. Times 80: $68{,}451{,}041{,}280$.
4. Embedding plus output head: $2 \times 32{,}000 \times 8{,}192 = 524{,}288{,}000$. Total $P = 68{,}975{,}329{,}280$.
5. State: $16 \times 68{,}975{,}329{,}280 = 1{,}103.61$ GB. With $t = 8$: $137.95$ GB, still too big. With $t = 8, p = 2$: $68.98$ GB, which fits 80 GB before activations.

### Data parallelism is exact

With equal shards of $n$ examples each, the mean loss over the full batch of $Nn$ examples is the mean of the shard means, so its gradient is the mean of the shard gradients:

```math
\nabla \frac{1}{Nn}\sum_{i=1}^{Nn} \ell_i = \frac{1}{N}\sum_{k=1}^{N} \nabla \Big(\frac{1}{n}\sum_{i \in \text{shard } k} \ell_i\Big)
```

If shards are unequal, a plain mean of shard gradients weights the small shards too much. The script shows the error: 0.067 for 60 examples on 8 GPUs (7 or 8 each).

### Tensor parallelism: columns and rows

Split $W$ by **columns** into $[W_1 \mid W_2]$: GPU $k$ computes $X W_k$, and the output is the concatenation $[X W_1 \mid X W_2]$. No sum is needed.

Split $W$ by **rows** and $X$ by the matching columns: GPU $k$ computes $X_k W_k$ and the output is the **sum** $X_1 W_1 + X_2 W_2$, which is an all-reduce.

A transformer MLP is $\text{act}(X W_{\text{up}}) W_{\text{down}}$. Column-split $W_{\text{up}}$: each GPU gets a slice of the hidden activation and applies the elementwise activation locally. Row-split $W_{\text{down}}$: the slices line up with the hidden slices, so each GPU multiplies its own piece and one all-reduce sums the result. That is Megatron-LM's layout: one all-reduce per MLP in the forward pass (and one per attention block, which splits by heads the same way).

**Worked example.** $X = [1, 2]$, $W = \begin{bmatrix} 1 & 0 & 2 & 1 \\ 0 & 1 & 1 & 3 \end{bmatrix}$. Full: $XW = [1, 2, 4, 7]$.
- Columns over 2 GPUs: GPU 1 has the first two columns, $X W_1 = [1 \cdot 1 + 2 \cdot 0,\ 1 \cdot 0 + 2 \cdot 1] = [1, 2]$; GPU 2 has $X W_2 = [1 \cdot 2 + 2 \cdot 1,\ 1 \cdot 1 + 2 \cdot 3] = [4, 7]$. Concatenate: $[1, 2, 4, 7]$. ✓
- Rows over 2 GPUs: GPU 1 has $x_1 = 1$ and row 1, giving $[1, 0, 2, 1]$; GPU 2 has $x_2 = 2$ and row 2, giving $[0, 2, 2, 6]$. Sum: $[1, 2, 4, 7]$. ✓

### Pipeline bubble

With $p$ stages and $m$ micro-batches, forward 1 tick and backward 2 ticks per stage, all forwards run first (GPipe). The forward phase takes $m + p - 1$ ticks and the backward phase $2(m + p - 1)$. Each stage is busy for $3m$ ticks.

```math
\text{bubble} = 1 - \frac{3m}{3(m + p - 1)} = \frac{p - 1}{m + p - 1}
```

**Worked example.** $p = 4$, $m = 4$: step time $3 \times (4 + 3) = 21$ ticks, busy $3 \times 4 = 12$, bubble $= 9/21 = 3/7 = 42.9\%$. With $m = 16$: $3/19 = 15.8\%$. With $m = 32$: $3/35 = 8.6\%$.

You will also see the bubble written as $(p - 1)/m$: that is the idle time *relative to useful time* rather than to total time. For $m = 16$, $3/16 = 18.75\%$ of useful time, which is the same $3/19$ of the total.

### 3D parallelism

$N = t \cdot p \cdot d$. With $t = 8$, $p = 16$, $d = 128$: $8 \times 16 = 128$, $128 \times 128 = 16{,}384$ GPUs. Each GPU holds $1/(t \cdot p) = 1/128$ of the weights and sees $1/128$ of each batch.

## 4. Implementation

### Build it from scratch

**Data parallelism.** Each "GPU" computes the mean-squared-error gradient on its shard; the all-reduce is a mean.

```python
def mse_grad(w, X, y):
    return 2 * X.T @ (X @ w - y) / len(y)

def data_parallel_grad(w, X, y, n_gpus):
    """Each GPU gets an equal shard, computes its local gradient; all-reduce averages them."""
    shards = zip(np.array_split(X, n_gpus), np.array_split(y, n_gpus))
    return np.mean([mse_grad(w, Xs, ys) for Xs, ys in shards], axis=0)
```

*Predict:* with 64 examples on 8 GPUs, how far is the result from `mse_grad(w, X, y)`?

**Tensor parallelism.** Column split concatenates; row split sums. The Megatron MLP chains them so the hidden activation never leaves its GPU.

```python
def column_parallel(x, W, n):
    """Each GPU holds a slice of W's output columns and computes a slice of y. Combine: concatenate."""
    return np.concatenate([x @ Ws for Ws in np.array_split(W, n, axis=1)], axis=-1)

def row_parallel(x, W, n):
    """Each GPU holds a slice of W's input rows and the matching slice of x. Combine: sum (an all-reduce)."""
    xs, Ws = np.array_split(x, n, axis=-1), np.array_split(W, n, axis=0)
    return sum(a @ b for a, b in zip(xs, Ws))

def tp_mlp(x, W1, W2, n):
    h_parts = [np.maximum(x @ W1s, 0) for W1s in np.array_split(W1, n, axis=1)]   # local, no comm
    return sum(h @ W2s for h, W2s in zip(h_parts, np.array_split(W2, n, axis=0)))  # all-reduce
```

*Predict:* why can ReLU be applied to each slice separately, but a layer norm over the hidden dimension could not?

**Pipeline bubble.** The schedule length and the idle fraction follow from counting ticks.

```python
def gpipe_time(p, m):
    """p stages, m micro-batches, forward = 1 tick, backward = 2 ticks per stage per micro-batch."""
    return (m + p - 1) * 1 + (m + p - 1) * 2

def bubble_fraction(p, m):
    busy = 3 * m          # each stage does m forwards and m backwards
    return 1 - busy / gpipe_time(p, m)           # = (p - 1) / (m + p - 1)
```

*Predict:* doubling the stages from 4 to 8 with $m = 32$: does the bubble double?

**Memory.** Bytes per parameter times parameters, divided by however many ways the weights are split.

```python
def train_memory(P, tp=1, pp=1, bytes_per_param=16):
    """16 bytes/param = bf16 weights 2 + bf16 grads 2 + fp32 master 4 + Adam m 4 + Adam v 4."""
    return bytes_per_param * P / (tp * pp)
```

*Predict:* switching to 8-bit Adam moments (1 byte each instead of 4) changes `bytes_per_param` to what?

### Use the library

```python
# Data parallel: one process per GPU, launched with  torchrun --nproc-per-node=8 train.py
model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank])

# Sharded data parallel (ZeRO-3 style) with PyTorch FSDP2
from torch.distributed.fsdp import fully_shard
for block in model.layers:
    fully_shard(block)
fully_shard(model)
```

Tensor and pipeline parallelism need model code that knows how to split itself: Megatron-LM / Megatron-Core, PyTorch's `torch.distributed.tensor.parallel` and `torch.distributed.pipelining`, or DeepSpeed's pipeline engine. Hugging Face Accelerate and the Trainer wrap DDP, FSDP, and DeepSpeed ZeRO behind a config file.

### Run it

```bash
uv run code/27-training-at-scale/scaling_distributed.py
```

Key lines:

```text
  8 GPUs: max |avg shard grad - full grad| = 4.4e-16
  unequal shards (60 examples on 8 GPUs): diff = 6.7e-02
  8 GPUs: column 0.0e+00, row 1.5e-14, column+row MLP 6.2e-15
  p = 4, m =  4: step =  21 ticks, bubble = 42.9%  [(p-1)/(m+p-1) = 3/7]
  p = 4, m = 16: step =  57 ticks, bubble = 15.8%  [(p-1)/(m+p-1) = 3/19]
  7B-style: P = 6,738,149,376; full state = 107.81 GB
  70B-style: P = 68,975,329,280; full state = 1103.61 GB
    TP 8 x PP 2:    68.98 GB per GPU  fits for 80 GB
  3D parallelism: TP 8 x PP 16 x DP 128 = 16,384 GPUs
```

## 5. Engineering

**Choosing a strategy.** A reasonable order of questions:
1. Does the full training state fit one GPU? Use DDP; it is the simplest and fastest.
2. Does it fit if optimizer state and gradients are sharded? Use ZeRO-1 or ZeRO-2 (same communication as DDP).
3. Otherwise shard everything with ZeRO-3 / FSDP. This covers models up to tens of billions of parameters on a few dozen GPUs.
4. For the largest models, or when per-GPU batch would get too small, add TP inside each server and PP across servers.

**Global batch size.** DP multiplies the batch: 64 GPUs × 16 sequences = 1,024 sequences per step. Large batches may need a higher learning rate and warm-up, and past some size more batch stops helping. Gradient accumulation lets you reach a target batch with fewer GPUs.

**Pipelines need micro-batches.** The bubble shrinks only when $m \gg p$, but more micro-batches in flight means more stored activations. Interleaved and 1F1B schedules (alternating one forward, one backward) reduce memory and the bubble.

**Mixture of experts.** MoE models such as DeepSeek-V3 (671B total parameters, 37B active per token; trained on 2,048 H800 GPUs per its report) add *expert parallelism*: different GPUs hold different experts and tokens are routed to them with all-to-all communication.

**Failures at scale.** On thousands of GPUs, hardware failures are routine. Large runs depend on frequent sharded checkpoints and automatic restart.

### Common mistakes

- Assuming DP reduces memory: it only increases throughput. Each GPU still holds 16 bytes per parameter.
- Forgetting the fp32 master copy and using 12 bytes per parameter for mixed-precision Adam: it is 16.
- Running TP across servers over a slow network: TP communicates in every layer and needs NVLink-class bandwidth.
- Using few micro-batches with many pipeline stages: with $p = 16$ and $m = 4$ the bubble is $15/19 = 79\%$.
- Averaging gradients over unequal shards without weighting, or not scaling the learning rate and warm-up for a much larger global batch.

## 6. Knowledge check

<!-- quiz:scaling-distributed -->
**[Take the scaling and distributed training quiz](../../quizzes/scaling-distributed.md)**
<!-- /quiz -->

**Practice exercise.** A synthetic 13B-style model has $13 \times 10^9$ parameters. With mixed-precision AdamW and no sharding, how much training state is there? What TP degree is the minimum (a power of 2, at most 8) to fit 80 GB per GPU before activations? What pipeline bubble would 4 stages with 12 micro-batches give?

<details>
<summary>Solution</summary>

State: $16 \times 13 \times 10^9 = 208$ GB. TP 2: $104$ GB, no. TP 4: $52$ GB, fits. Bubble: $(4 - 1)/(12 + 4 - 1) = 3/15 = 20\%$.

</details>

**Implementation challenge.** Make `data_parallel_grad` exact for unequal shards: weight each shard's gradient by its number of examples. Verify the difference from `mse_grad` drops to floating-point noise for 60 examples on 8 GPUs.

<details>
<summary>Solution</summary>

```python
def data_parallel_grad_weighted(w, X, y, n_gpus):
    parts = list(zip(np.array_split(X, n_gpus), np.array_split(y, n_gpus)))
    total = sum(len(ys) for _, ys in parts)
    return sum(len(ys) * mse_grad(w, Xs, ys) for Xs, ys in parts) / total
```

Each shard's mean gradient times its size is its *sum* of per-example gradients; adding those and dividing by the total count gives the full-batch mean exactly. In a real job you would all-reduce the per-shard sums and the counts. In practice most data loaders drop or pad the last uneven batch instead.

</details>

## Summary

- Mixed-precision AdamW costs 16 bytes per parameter before activations: 107.81 GB for a 7B-style model, 1.10 TB for a 70B-style one.
- Data parallelism scales throughput with one gradient all-reduce per step, but does not reduce memory per GPU.
- Tensor parallelism splits matrices (columns concatenate, rows sum); it is exact but communicates every layer, so it stays inside a server.
- Pipeline parallelism splits layers; its bubble is $(p - 1)/(m + p - 1)$, so it needs many micro-batches.
- Big runs combine all three: Llama 3 405B used TP 8 × PP 16 × DP on up to 16,384 GPUs. Sharded data parallelism (next lesson) removes DP's duplicate memory.

**Next:** [ZeRO and FSDP](05-zero-parameter-sharding.md)

**Related:** [Collective operations](03-collective-ops-from-scratch.md) · [Transformer architecture](../14-transformers/02-transformer-architecture.md) · [Serving LLMs](../18-llm-inference/02-serving-llms.md)

*Go deeper, code-first: [scaling and distributed training](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/05-scaling-distributed) · [DDP and FSDP from scratch](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/48-distributed-fsdp-ddp) · [pipeline parallelism](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/79-pipeline-parallel) (AI Engineering from Scratch).*

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Distributed Data Parallel and FSDP from Scratch](https://aiengineeringfromscratch.com/lesson?path=phases/19-capstone-projects/48-distributed-fsdp-ddp) · [Pipeline Parallel and Bubble Analysis](https://aiengineeringfromscratch.com/lesson?path=phases/19-capstone-projects/79-pipeline-parallel)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Scaling: Distributed Training, FSDP, DeepSpeed](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/05-scaling-distributed).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>How would you train a 70B model on 64 GPUs with 80 GB each?</strong></summary>

Start with memory. A 70B model with mixed-precision AdamW needs 16 bytes per parameter, about 1.1 TB, before activations. Plain data parallelism would need all of it on every GPU, so it is out. Across 64 GPUs the total is about 17 GB per GPU if fully sharded, which fits easily. So the simplest plan is ZeRO-3 / FSDP over all 64 GPUs with gradient checkpointing: about 17 GB of state plus a few GB of activations per GPU, and communication of about 1.5 times DDP. If the network between servers is slow, hybrid sharding helps: shard within groups of 16 GPUs (about 69 GB each, tight but it fits) and replicate across the 4 groups; sharding within one 8-GPU server would need 138 GB and does not fit. If throughput is still communication-bound, add 8-way tensor parallelism inside each server and run sharded data parallelism across the 8 servers. Pipeline parallelism is usually unnecessary at this size. I would verify with a short profiled run before committing.

</details>

<details>
<summary><strong>Compare data, tensor, and pipeline parallelism. What does each cost?</strong></summary>

Data parallelism replicates the model and splits the batch; it costs one gradient all-reduce per step, about twice the gradient size per GPU, which overlaps with backward. It does not reduce memory. Tensor parallelism splits each weight matrix, so memory per GPU drops by the TP degree, but it needs an all-reduce inside every transformer block in both forward and backward, on the critical path, so it requires NVLink-class bandwidth and stays within a server. Pipeline parallelism splits layers into stages; it sends only boundary activations, so it tolerates slower links, but it has a bubble of (p − 1)/(m + p − 1) of the step time and needs many micro-batches, which costs activation memory. In practice: TP inside the server, PP across a few servers, data parallelism (usually sharded) across everything else.

</details>

<details>
<summary><strong>You doubled pipeline stages from 4 to 8 to fit a bigger model, and throughput dropped 25%. Why?</strong></summary>

Most likely the bubble. With the same number of micro-batches the idle fraction is (p − 1)/(m + p − 1). With m = 16, 4 stages give 3/19, about 16% idle; 8 stages give 7/23, about 30%. That alone costs roughly 17% throughput, and more stages also mean more boundary transfers and more chances for uneven stage times: the slowest stage sets the pace for everyone, and splitting 80 layers into 8 stages often leaves the stage with the embedding or output head heavier. Fixes: raise micro-batches per step (with gradient accumulation if the global batch must stay fixed), use an interleaved or 1F1B schedule, balance layers across stages by measured time rather than count, or reduce PP and fit the model with more sharding or tensor parallelism instead.

</details>

<details>
<summary><strong>Quick calculation: what does a 7B model's training state cost per GPU on 8 GPUs with DDP versus 8-way tensor parallelism?</strong></summary>

A 7B-style model has about 6.74 billion parameters. Mixed-precision AdamW is 16 bytes each: 2 bf16 weights, 2 bf16 gradients, and 12 for the fp32 master copy and two Adam moments, so 107.8 GB. With DDP every GPU holds all of it, about 108 GB per GPU plus activations, which does not fit an 80 GB card. With 8-way tensor parallelism each GPU holds one eighth of every matrix and its gradients and optimizer state, so 107.8 / 8 = 13.5 GB per GPU, which fits with room for activations. But TP then all-reduces activations inside every block. ZeRO-3 over the same 8 GPUs gives the same 13.5 GB without changing model code, at about 1.5 times DDP's communication, so for a 7B model I'd usually start with FSDP rather than TP.

</details>
