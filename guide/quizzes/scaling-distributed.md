<!-- GENERATED from scaling-distributed.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Scaling and distributed training

Covers the lesson [Scaling and distributed training](../lessons/27-training-at-scale/04-scaling-distributed.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/scaling-distributed/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

With mixed-precision AdamW (bf16 weights and gradients, fp32 master copy and two fp32 Adam moments), how many bytes per parameter does training state take, before activations?

<details>
<summary>Answer</summary>

**16**

$2 + 2 + 4 + 4 + 4 = 16$.

</details>

## 2. Calculation (medium)

A model has $7 \times 10^9$ parameters. How many GB (decimal) of training state does it need with mixed-precision AdamW?

<details>
<summary>Answer</summary>

**112**

$16 \times 7 \times 10^9 = 112 \times 10^9$ bytes $= 112$ GB. (Leaving out the fp32 master copy would give 84 GB, a common mistake.)

</details>

## 3. Multiple choice (easy)

What does plain data parallelism (DDP) do to memory per GPU as you add GPUs?

- **A.** Divides it by the number of GPUs.
- **B.** Nothing; every GPU still holds the full weights, gradients, and optimizer state.
- **C.** Halves it.
- **D.** Doubles it, because of communication buffers.

<details>
<summary>Answer</summary>

**B.** Nothing; every GPU still holds the full weights, gradients, and optimizer state.

DDP replicates the model; it scales throughput, not capacity. Sharding (ZeRO, FSDP) is what divides memory.

- **A:** That is ZeRO-3 / FSDP.
- **C:** No part is split under DDP.
- **D:** Buffers add a little, not a doubling.

</details>

## 4. Match (medium)

Match each strategy to what it splits across GPUs.

| Concept | Options |
|---|---|
| Data parallelism | weights, gradients, and optimizer state (gathered when needed) |
| Tensor parallelism | the layers |
| Pipeline parallelism | each weight matrix |
| ZeRO-3 / FSDP | the batch |

<details>
<summary>Answer</summary>

- Data parallelism → the batch
- Tensor parallelism → each weight matrix
- Pipeline parallelism → the layers
- ZeRO-3 / FSDP → weights, gradients, and optimizer state (gathered when needed)

DP splits data, TP splits matrices inside a layer, PP splits the stack of layers, and FSDP shards all per-parameter state while keeping the data-parallel structure.

</details>

## 5. Calculation (medium)

A GPipe pipeline has $p = 4$ stages and $m = 12$ micro-batches. What fraction of the schedule time is bubble (idle)? (3 decimals)

<details>
<summary>Answer</summary>

**0.2** (within ±0.001)

$(p - 1)/(m + p - 1) = 3/15 = 0.200$.

</details>

## 6. Multiple choice (medium)

In a Megatron-style MLP, $W_{\text{up}}$ is split by columns and $W_{\text{down}}$ by rows. Why this order?

- **A.** It halves the number of parameters.
- **B.** Each GPU applies the activation to its own slice of the hidden layer, and the row split of $W_{\text{down}}$ lines up with those slices, so only one all-reduce is needed at the end.
- **C.** Row splits need no communication at all.
- **D.** Column splits are only allowed for the first layer.

<details>
<summary>Answer</summary>

**B.** Each GPU applies the activation to its own slice of the hidden layer, and the row split of $W_{\text{down}}$ lines up with those slices, so only one all-reduce is needed at the end.

Column split gives hidden slices (no communication); elementwise activation is local; the row split multiplies each slice by matching rows and sums the partial outputs with one all-reduce.

- **A:** The parameter count is unchanged; it is just split.
- **C:** Row splits produce partial sums that must be added, which is an all-reduce.
- **D:** Either split can be used for any matrix; the order matters for communication.

</details>

## 7. Multiple choice (hard)

A team runs 16-way tensor parallelism across two 8-GPU servers connected by Ethernet, and GPU utilization is poor. What is the most likely reason?

- **A.** Tensor parallelism is not exact, so it recomputes layers.
- **B.** Tensor parallelism all-reduces inside every block on the critical path; across a slow inter-node link these all-reduces dominate step time.
- **C.** The pipeline bubble is too large.
- **D.** The optimizer state is replicated.

<details>
<summary>Answer</summary>

**B.** Tensor parallelism all-reduces inside every block on the critical path; across a slow inter-node link these all-reduces dominate step time.

Keep TP within the NVLink domain (one server), and use pipeline or (sharded) data parallelism across servers.

- **A:** TP is mathematically exact.
- **C:** There is no pipeline here.
- **D:** Replicated optimizer state is a memory issue, not a utilization issue.

</details>

## 8. Calculation (easy)

A run uses 8-way tensor, 16-way pipeline, and 128-way data parallelism. How many GPUs?

<details>
<summary>Answer</summary>

**16384**

$8 \times 16 \times 128 = 16{,}384$. This is the shape reported for Llama 3 405B pre-training.

</details>

## 9. Multiple choice (hard)

Data-parallel training on 8 GPUs gives slightly different results from single-GPU training on the same 60-example batch. Shard sizes are 7 or 8. Why?

- **A.** Floating-point non-associativity alone.
- **B.** A plain mean of shard-mean gradients weights the smaller shards' examples more; with unequal shards it is not the full-batch mean.
- **C.** Tensor parallelism changed the matmul order.
- **D.** The learning rate must be multiplied by 8.

<details>
<summary>Answer</summary>

**B.** A plain mean of shard-mean gradients weights the smaller shards' examples more; with unequal shards it is not the full-batch mean.

Weight each shard gradient by its size (or all-reduce sums and counts). The lesson's script shows a difference of 0.067 here, far above rounding noise of about $10^{-16}$.

- **A:** Rounding differences are around $10^{-16}$, not $10^{-2}$.
- **C:** No tensor parallelism is involved.
- **D:** The global batch is the same 60 examples; no rescaling is needed.

</details>

## 10. Reflection (hard)

You must fully fine-tune a 70B model on 64 GPUs with 80 GB each. Sketch a parallelism plan and justify it with numbers.

<details>
<summary>Answer</summary>

**Model answer.** Training state is 16 bytes per parameter, about 1.1 TB, so DDP is impossible. ZeRO-3 / FSDP across all 64 GPUs gives about 17 GB per GPU plus a few GB of checkpointed activations, comfortably under 80 GB, with about 1.5 times DDP's communication. If inter-node bandwidth is the bottleneck, use hybrid sharding within groups of at least 16 GPUs (about 69 GB each) or add 8-way tensor parallelism inside each server with sharded data parallelism across servers. Turn on gradient checkpointing and bf16; profile a few steps before committing.

Start from bytes per parameter, then pick the least communication-heavy plan that fits.

</details>
