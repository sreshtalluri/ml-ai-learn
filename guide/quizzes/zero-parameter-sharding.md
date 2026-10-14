<!-- GENERATED from zero-parameter-sharding.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: ZeRO and FSDP

Covers the lesson [ZeRO and FSDP](../lessons/27-training-at-scale/05-zero-parameter-sharding.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/zero-parameter-sharding/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

Why does ZeRO shard the optimizer state first?

- **A.** It is the largest per-parameter term (12 of 16 bytes) and is only used in the optimizer step, not in forward or backward.
- **B.** DeepSpeed requires it.
- **C.** The optimizer state is the most important for accuracy.
- **D.** Weights cannot be sharded.

<details>
<summary>Answer</summary>

**A.** It is the largest per-parameter term (12 of 16 bytes) and is only used in the optimizer step, not in forward or backward.

Sharding the biggest term that nobody else needs gives the largest saving for the least extra communication: none.

- **B:** It is a design choice backed by the memory arithmetic, not a framework rule.
- **C:** Sharding does not change the values or the accuracy.
- **D:** ZeRO-3 shards weights too.

</details>

## 2. Multiple choice (medium)

What are the communication operations in one ZeRO-1 step?

- **A.** Nothing.
- **B.** One broadcast.
- **C.** One all-reduce of the gradients, as in DDP.
- **D.** One reduce-scatter of gradients, then one all-gather of the updated weights.

<details>
<summary>Answer</summary>

**D.** One reduce-scatter of gradients, then one all-gather of the updated weights.

Each rank receives only its slice's summed gradient, updates that slice with its local Adam state, and all-gathers the new weights.

- **A:** Gradients must still be combined across ranks.
- **B:** A single broadcast cannot combine gradients.
- **C:** An all-reduce gives every rank the whole gradient, which ZeRO-1 does not need.

</details>

## 3. Calculation (medium)

With mixed-precision AdamW on $N = 8$ GPUs, ZeRO-1 stores $4P + 12P/N$ bytes per GPU. By what percentage does that reduce memory compared with DDP's $16P$? (1 decimal)

<details>
<summary>Answer</summary>

**65.6** (within ±0.1)

$4 + 12/8 = 5.5$ bytes per parameter. $(16 - 5.5)/16 = 10.5/16 = 65.6\%$.

</details>

## 4. Multiple choice (medium)

Why do ZeRO-1 and ZeRO-2 send the same number of bytes as DDP?

- **A.** They compress gradients.
- **B.** A ring all-reduce is itself a reduce-scatter plus an all-gather; ZeRO does the same two operations, only the all-gather carries updated weights instead of summed gradients.
- **C.** They skip the all-gather.
- **D.** They do not; they send twice as much.

<details>
<summary>Answer</summary>

**B.** A ring all-reduce is itself a reduce-scatter plus an all-gather; ZeRO does the same two operations, only the all-gather carries updated weights instead of summed gradients.

With bf16 weights and gradients both halves move $\frac{N-1}{N} \cdot 2P$ bytes, so the total equals DDP's $2\frac{N-1}{N} \cdot 2P$.

- **A:** No compression is involved.
- **C:** Without the all-gather ranks would not have the full model for the next forward.
- **D:** Traffic is identical.

</details>

## 5. Calculation (medium)

ZeRO-3 all-gathers weights for the forward pass, again for the backward pass, and reduce-scatters gradients. With bf16 weights and gradients, how many times DDP's bytes does it send per step?

<details>
<summary>Answer</summary>

**1.5**

Three operations of $\frac{N-1}{N} \cdot 2P$ each, versus DDP's two (one all-reduce). $3/2 = 1.5$.

</details>

## 6. Calculation (hard)

A 70B-style model has $P = 69 \times 10^9$ parameters. Under ZeRO-3 with mixed-precision AdamW, plus 3.3 GB of activations per GPU, what is the fewest GPUs (a power of 2) that keeps each GPU under 80 GB?

<details>
<summary>Answer</summary>

**16**

$16P = 1{,}104$ GB. On 8 GPUs: $138 + 3.3 = 141.3$ GB, too big. On 16: $69 + 3.3 = 72.3$ GB, fits.

</details>

## 7. Multiple choice (hard)

You keep adding GPUs to a ZeRO-1 job. What does memory per GPU approach?

- **A.** Zero.
- **B.** $4P$ bytes, the replicated bf16 weights and gradients.
- **C.** $12P$ bytes.
- **D.** $16P/N$ bytes.

<details>
<summary>Answer</summary>

**B.** $4P$ bytes, the replicated bf16 weights and gradients.

$4P + 12P/N \to 4P$. Only sharding the weights and gradients (ZeRO-3) removes that floor; ZeRO-2's floor is $2P$.

- **A:** Replicated pieces never shrink.
- **C:** That is the optimizer state, which is the part that does get divided.
- **D:** That is ZeRO-3's formula.

</details>

## 8. Multiple choice (hard)

An FSDP job runs out of memory although the formula predicts 30 GB per GPU on 80 GB cards. The whole model was wrapped in one `fully_shard` call. What is the likely cause?

- **A.** FSDP does not shard optimizer state.
- **B.** With one wrap unit, FSDP gathers every weight at once for forward and backward, so the full model is materialized on each GPU; wrap each transformer block separately.
- **C.** Gradient checkpointing doubles weight memory.
- **D.** bf16 weights take 4 bytes.

<details>
<summary>Answer</summary>

**B.** With one wrap unit, FSDP gathers every weight at once for forward and backward, so the full model is materialized on each GPU; wrap each transformer block separately.

Each wrapped unit is gathered whole. Per-block wrapping keeps only one or two blocks' full weights alive at a time.

- **A:** FSDP shards optimizer state along with weights and gradients.
- **C:** Checkpointing affects activations, not weights.
- **D:** bf16 is 2 bytes.

</details>

## 9. Multiple choice (medium)

Under ZeRO-2, how must global-norm gradient clipping be computed?

- **A.** Each rank clips its own gradient slice by its own norm.
- **B.** Each rank computes the sum of squares of its slice, an all-reduce adds them, and every rank scales by the same factor from the global norm.
- **C.** Clipping is impossible with sharded gradients.
- **D.** Only rank 0 clips.

<details>
<summary>Answer</summary>

**B.** Each rank computes the sum of squares of its slice, an all-reduce adds them, and every rank scales by the same factor from the global norm.

A per-rank norm would apply different factors to different slices and change the update's direction.

- **A:** That is per-shard clipping, which changes direction.
- **C:** It is possible with one small all-reduce of a scalar.
- **D:** Other ranks' slices would never be clipped.

</details>

## 10. Reflection (hard)

You fine-tune a 13B model on one server with 8 x 80 GB GPUs. Which ZeRO stage would you start with, and why?

<details>
<summary>Answer</summary>

**Model answer.** Training state is about 16 x 13 = 208 GB. DDP needs 208 GB per GPU and does not fit. ZeRO-1 gives 4P + 12P/8 = 52 + 19.5 = 71.5 GB, too tight with activations. ZeRO-2 gives 2P + 14P/8 = 26 + 22.75 = 48.75 GB, which fits with room for activations at DDP's communication cost. So start with ZeRO-2 (FSDP's SHARD_GRAD_OP) plus gradient checkpointing, and only move to ZeRO-3 if longer sequences need more headroom.

Pick the lowest stage that fits; ZeRO-1 and 2 cost no extra communication.

</details>
