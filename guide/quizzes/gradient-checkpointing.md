<!-- GENERATED from gradient-checkpointing.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Gradient checkpointing

Covers the lesson [Gradient checkpointing](../lessons/27-training-at-scale/02-gradient-checkpointing.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/gradient-checkpointing/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What does gradient checkpointing reduce?

- **A.** Memory for weights.
- **B.** Memory for optimizer states.
- **C.** Memory for activations saved for the backward pass.
- **D.** Memory for gradients.

<details>
<summary>Answer</summary>

**C.** Memory for activations saved for the backward pass.

It keeps only segment inputs and recomputes the intermediates during backward. Weights, gradients, and optimizer state are unchanged.

- **A:** Weights are needed in full for forward and backward; sharding (ZeRO-3) reduces them.
- **B:** Optimizer state is reduced by sharding or 8-bit optimizers.
- **D:** Gradients are reduced by ZeRO-2 sharding, not by checkpointing.

</details>

## 2. Calculation (medium)

A backward pass costs about twice a forward pass. With full checkpointing (every layer's forward recomputed once), what is the extra compute per step, as a percentage? (1 decimal)

<details>
<summary>Answer</summary>

**33.3** (within ±0.1)

Without: $1 + 2 = 3$ units. With: $1 + 1 + 2 = 4$ units. $4/3 - 1 = 33.3\%$.

</details>

## 3. Calculation (hard)

A synthetic model has $L = 32$ layers, hidden size $d = 4096$, and processes 2,048 tokens. Using 34 bytes per token per hidden unit per layer without checkpointing, how many GB (decimal) of activations does it keep? (2 decimals)

<details>
<summary>Answer</summary>

**9.13** (within ±0.01)

$32 \times 34 \times 2{,}048 \times 4{,}096 = 1{,}088 \times 8{,}388{,}608 = 9{,}126{,}805{,}504$ bytes $\approx 9.13$ GB.

</details>

## 4. Calculation (hard)

Same model, checkpointing every layer: keep each layer's input at 2 bytes per value, plus one layer's full activations (34 bytes per token per hidden unit) during backward. How many GB? (3 decimals)

<details>
<summary>Answer</summary>

**0.822** (within ±0.001)

$T d = 8{,}388{,}608$. Inputs: $32 \times 2 \times 8{,}388{,}608 = 536{,}870{,}912$. One layer: $34 \times 8{,}388{,}608 = 285{,}212{,}672$. Sum $822{,}083{,}584$ bytes $\approx 0.822$ GB.

</details>

## 5. Multiple choice (hard)

The classic rule says checkpoint every $\sqrt{L}$ layers. Why do transformer frameworks usually checkpoint every block instead?

- **A.** Because $\sqrt{L}$ is not an integer for most models.
- **B.** The rule assumes a stored checkpoint costs as much as a layer's full activations; a block's input is about 17 times smaller than everything the block saves, so small segments minimize memory.
- **C.** Because recompute cost grows with segment size.
- **D.** Because per-block checkpointing needs no recomputation.

<details>
<summary>Answer</summary>

**B.** The rule assumes a stored checkpoint costs as much as a layer's full activations; a block's input is about 17 times smaller than everything the block saves, so small segments minimize memory.

Memory is $\lceil L/k \rceil \cdot b + k \cdot A$ (in units of $Td$). With $b = 2$ and $A = 34$, $k = 1$ gives 98 units for $L = 32$, while $k = 6$ gives 216.

- **A:** Rounding would be trivial; it is not the reason.
- **C:** Each layer is recomputed once whatever the segment size, so compute does not depend on k.
- **D:** Every checkpointed block is recomputed during backward.

</details>

## 6. Multiple choice (hard)

After enabling checkpointing, your loss curve no longer matches the run without it, even with the same seed. Which cause is most likely?

- **A.** Checkpointing always changes gradients slightly by design.
- **B.** Dropout draws a different random mask in the recomputed forward because the RNG state was not preserved.
- **C.** The optimizer state is now stored in 16-bit.
- **D.** Checkpointing halves the effective batch size.

<details>
<summary>Answer</summary>

**B.** Dropout draws a different random mask in the recomputed forward because the RNG state was not preserved.

Recomputation must reproduce the original forward exactly. PyTorch saves and restores RNG state by default (`preserve_rng_state=True`); custom code that reseeds or disables it breaks this.

- **A:** With deterministic recomputation, gradients are identical (the lesson's script shows a difference of exactly 0).
- **C:** Checkpointing does not touch the optimizer.
- **D:** Batch size is unchanged.

</details>

## 7. Multiple choice (medium)

You wrap the entire 32-layer model in a single checkpoint call. What happens to peak memory?

- **A.** It drops by 32 times.
- **B.** It drops to one layer's activations.
- **C.** It barely changes (slightly higher), because during backward the whole segment's activations are rebuilt at once, plus the stored input.
- **D.** Training fails because one segment is not allowed.

<details>
<summary>Answer</summary>

**C.** It barely changes (slightly higher), because during backward the whole segment's activations are rebuilt at once, plus the stored input.

The recompute rebuilds all 32 layers' intermediates before backward through them, so peak memory matches no checkpointing, and you pay the extra forward too.

- **A:** Savings come from having many segments, only one of which is live.
- **B:** That is per-layer checkpointing, not a single segment.
- **D:** It runs; it is just useless.

</details>

## 8. Multiple choice (medium)

Fine-tuning runs out of memory only when you raise the sequence length from 2K to 16K tokens. Weights, gradients, and optimizer state already fit with room to spare. What is the best first change?

- **A.** Switch from ZeRO-2 to ZeRO-3.
- **B.** Turn on gradient checkpointing (selective if available) and keep FlashAttention on.
- **C.** Use 8-bit Adam.
- **D.** Increase the number of data-parallel GPUs.

<details>
<summary>Answer</summary>

**B.** Turn on gradient checkpointing (selective if available) and keep FlashAttention on.

Activations scale with tokens; the fixed per-parameter memory already fits. Checkpointing targets exactly the part that grew.

- **A:** ZeRO shards per-parameter state, not activations.
- **C:** 8-bit Adam shrinks optimizer state, which already fits.
- **D:** Each GPU still processes its own 16K-token sequences; activations per GPU do not drop.

</details>

## 9. Reflection (medium)

When would you offload activations to CPU memory instead of recomputing them?

<details>
<summary>Answer</summary>

**Model answer.** When the GPU is short on memory but has spare compute time hidden by overlap and the host link (PCIe or NVLink-C2C) has idle bandwidth, so the copies to and from CPU can overlap with compute. Offload costs no extra FLOPs, but every activation crosses the link twice, so it only pays when transfers are hidden; otherwise recomputation is usually faster.

Offload trades transfer bandwidth for recompute FLOPs.

</details>
