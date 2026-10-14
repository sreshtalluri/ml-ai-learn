<!-- GENERATED from collective-ops-from-scratch.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Collective operations from scratch

Covers the lesson [Collective operations from scratch](../lessons/27-training-at-scale/03-collective-ops-from-scratch.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/collective-ops-from-scratch/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

How many bytes does each rank send in a ring all-reduce of a $T$-byte tensor over $N$ ranks?

- **A.** $N \cdot T$
- **B.** $T \log_2 N$
- **C.** $T$
- **D.** $2T(N-1)/N$

<details>
<summary>Answer</summary>

**D.** $2T(N-1)/N$

Reduce-scatter and all-gather each take $N - 1$ steps sending a $T/N$ chunk: $2(N-1) \cdot T/N$. It approaches $2T$ and does not grow with $N$.

- **A:** That is roughly what a parameter server's link carries, not a ring rank.
- **B:** That is a simple tree's bandwidth term.
- **C:** Each rank must send its data out and pass on others' partial sums, so it is close to $2T$.

</details>

## 2. Multiple choice (medium)

After the reduce-scatter half of a ring all-reduce, what does each rank hold?

- **A.** The full summed tensor.
- **B.** The fully summed values of one chunk ($1/N$ of the tensor).
- **C.** Its own unmodified tensor.
- **D.** Partial sums of every chunk.

<details>
<summary>Answer</summary>

**B.** The fully summed values of one chunk ($1/N$ of the tensor).

Each chunk's partial sum travels around the ring for $N - 1$ steps, collecting every rank's contribution, and ends complete on one rank. All-gather then shares those finished chunks.

- **A:** That is the state after the all-gather half.
- **B:** Reduce-scatter changes every rank's chunks.
- **D:** After $N - 1$ steps one chunk per rank is complete; the others are stale partial sums no longer needed.

</details>

## 3. Calculation (medium)

You all-reduce 13.48 GB of bf16 gradients over $N = 8$ GPUs with a ring. How many GB does each GPU send? (2 decimals)

<details>
<summary>Answer</summary>

**23.59** (within ±0.02)

$2 \times 7/8 = 1.75$. $1.75 \times 13.48 = 23.59$ GB (23.58 with the unrounded tensor size).

</details>

## 4. Calculation (easy)

How many sequential communication steps does a ring all-reduce take on 16 ranks?

<details>
<summary>Answer</summary>

**30**

$2(N - 1) = 2 \times 15 = 30$: 15 for reduce-scatter, 15 for all-gather.

</details>

## 5. Multiple choice (medium)

When does a tree all-reduce beat a ring?

- **A.** Always, because it has fewer steps.
- **B.** For large tensors on a fast network.
- **C.** For small messages or high-latency links, where the per-message startup cost dominates and $2\log_2 N$ rounds beat $2(N-1)$ steps.
- **D.** Never; trees send more bytes.

<details>
<summary>Answer</summary>

**C.** For small messages or high-latency links, where the per-message startup cost dominates and $2\log_2 N$ rounds beat $2(N-1)$ steps.

In the $\alpha$-$\beta$ model the ring pays $2(N-1)\alpha$ in latency, the tree $2\log_2 N \cdot \alpha$. For large tensors the ring's smaller bandwidth term wins.

- **A:** Fewer steps matter only when latency dominates.
- **B:** That is where the ring's bandwidth advantage wins.
- **D:** A simple tree does send more bytes, but for tiny messages bytes hardly matter.

</details>

## 6. Multiple choice (easy)

Which composition equals an all-reduce?

- **A.** Broadcast then reduce.
- **B.** Reduce-scatter then all-gather.
- **C.** All-gather then broadcast.
- **D.** Two reduce-scatters.

<details>
<summary>Answer</summary>

**B.** Reduce-scatter then all-gather.

Reduce-scatter leaves each rank with one summed chunk; all-gather gives everyone every summed chunk. ZeRO exploits this split.

- **A:** Reduce to one rank then broadcast also works, but that is the parameter-server pattern, and the order given here is reversed.
- **C:** All-gather concatenates without summing.
- **D:** The second reduce-scatter has nothing left to sum.

</details>

## 7. Multiple choice (hard)

A multi-GPU job hangs at step 0. Rank 0 runs `dist.all_reduce(loss)` inside `if rank == 0:` to log the loss; the other ranks skip it. Why does it hang?

- **A.** NCCL does not support all-reduce on scalars.
- **B.** Collectives must be called by every rank in the same order; rank 0 waits forever for partners that never join.
- **C.** The loss must be on CPU.
- **D.** All-reduce needs at least 1 MB of data.

<details>
<summary>Answer</summary>

**B.** Collectives must be called by every rank in the same order; rank 0 waits forever for partners that never join.

A collective is a rendezvous. Call it on all ranks (then print only on rank 0), and set a timeout so mismatches turn into errors instead of hangs.

- **A:** Scalars are fine.
- **C:** NCCL works on GPU tensors.
- **D:** There is no minimum size.

</details>

## 8. Multiple choice (medium)

You implement data parallelism by hand with `all_reduce(grad, op=SUM)` and nothing else. Compared with single-GPU training at the same per-GPU batch, what happens?

- **A.** Nothing; SUM is the standard choice.
- **B.** The update is $N$ times larger, as if the learning rate were multiplied by $N$, unless you divide by $N$ (or use AVG).
- **C.** Gradients become $N$ times smaller.
- **D.** The all-reduce sends $N$ times more bytes.

<details>
<summary>Answer</summary>

**B.** The update is $N$ times larger, as if the learning rate were multiplied by $N$, unless you divide by $N$ (or use AVG).

Data parallelism wants the mean of shard gradients. DDP divides by world size for you; a hand-rolled SUM does not.

- **A:** SUM without dividing is a common bug, not the standard.
- **C:** It is the opposite.
- **D:** Bytes sent do not depend on the reduction operation.

</details>

## 9. Reflection (medium)

Why does PyTorch DDP group gradients into buckets (25 MB by default) instead of all-reducing each parameter tensor separately?

<details>
<summary>Answer</summary>

**Model answer.** A model has hundreds of parameter tensors; one all-reduce each would pay the per-message latency hundreds of times and keep messages small and latency-bound. Buckets make a few large messages where bandwidth dominates, and each bucket's all-reduce starts as soon as backward has filled it, overlapping communication with the rest of the backward pass.

Bucketing amortizes latency and enables overlap.

</details>
