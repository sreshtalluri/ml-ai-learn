<!-- GENERATED from gradient-clipping-amp.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Gradient clipping and mixed precision

Covers the lesson [Gradient clipping and mixed precision](../lessons/27-training-at-scale/01-gradient-clipping-amp.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/gradient-clipping-amp/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What problem does clipping gradients by their global L2 norm solve?

- **A.** One unusual batch can produce a gradient hundreds of times larger than normal; clipping caps the size of that update so it cannot throw the weights far off.
- **B.** It stops the vocabulary from drifting during fine-tuning.
- **C.** It makes every training step faster.
- **D.** It removes the need for a learning-rate schedule.

<details>
<summary>Answer</summary>

**A.** One unusual batch can produce a gradient hundreds of times larger than normal; clipping caps the size of that update so it cannot throw the weights far off.

Clipping rescales the whole gradient when its norm exceeds a threshold. Normal steps are untouched; spikes are capped while keeping their direction.

- **B:** Clipping acts on gradient magnitudes, not on the tokenizer or vocabulary.
- **C:** Computing the norm adds a little work; it never makes steps faster.
- **D:** Warm-up and decay still matter; clipping only limits outlier steps.

</details>

## 2. Calculation (medium)

Gradients are $g_1 = [6, 8]$ and $g_2 = [0, 0]$. With global-norm clipping at $c = 2$, what is the first entry of the clipped $g_1$?

<details>
<summary>Answer</summary>

**1.2** (within ±0.001)

Norm $= \sqrt{36 + 64} = 10$. Factor $= \min(1, 2/10) = 0.2$. Clipped $g_1 = [6 \times 0.2, 8 \times 0.2] = [1.2, 1.6]$.

</details>

## 3. Multiple choice (medium)

Why clip by the global norm instead of clipping each value to $[-c, c]$?

- **A.** Value clipping is not supported by PyTorch.
- **B.** Global-norm clipping multiplies every gradient by the same factor, so the update keeps its direction; value clipping shrinks components unequally and changes the direction.
- **C.** Global-norm clipping is cheaper to compute.
- **D.** Value clipping only works with SGD.

<details>
<summary>Answer</summary>

**B.** Global-norm clipping multiplies every gradient by the same factor, so the update keeps its direction; value clipping shrinks components unequally and changes the direction.

With $[3, 4]$ and $[12]$, global clipping to 1 keeps the 3 : 4 : 12 ratio. Clipping each value to 1 gives $[1, 1, 1]$, a different direction.

- **A:** PyTorch has `clip_grad_value_`; it exists but changes direction.
- **C:** Global-norm clipping needs a reduction over all gradients, so it is slightly more work.
- **D:** Both can be used with any optimizer.

</details>

## 4. Arrange in order (medium)

Put one fp16 training step with a GradScaler and clipping in the correct order.

- scaler.update()
- scaler.step(optimizer)
- clip_grad_norm_(model.parameters(), 1.0)
- scaler.unscale_(optimizer)
- scaler.scale(loss).backward()

<details>
<summary>Answer</summary>

1. scaler.scale(loss).backward()
2. scaler.unscale_(optimizer)
3. clip_grad_norm_(model.parameters(), 1.0)
4. scaler.step(optimizer)
5. scaler.update()

Backward on the scaled loss, unscale so gradients are true size, clip those, step (skipped automatically on inf/NaN), then update the scale on every step.

</details>

## 5. Calculation (hard)

fp16's largest finite value is 65,504. With a loss scale of $S = 2^{17} = 131{,}072$, above what true gradient magnitude does a scaled gradient overflow? (3 decimals)

<details>
<summary>Answer</summary>

**0.5** (within ±0.001)

Overflow when $|g| \cdot S > 65{,}504$, so $|g| > 65{,}504 / 131{,}072 = 0.49976 \approx 0.500$.

</details>

## 6. Multiple choice (hard)

An fp16 run logs "skipped step" on every step after step 300, and the logged loss scale never changes. What is the most likely bug?

- **A.** The learning rate is too low.
- **B.** `scaler.update()` is only called on successful steps, so the scale never halves after an overflow.
- **C.** The model is too small for fp16.
- **D.** Gradient clipping threshold is too high.

<details>
<summary>Answer</summary>

**B.** `scaler.update()` is only called on successful steps, so the scale never halves after an overflow.

The scaler halves its factor inside `update()` when it has seen inf/NaN. If update is skipped on bad steps, the factor stays too high and every following step overflows again.

- **A:** A low learning rate makes training slow, not every step non-finite.
- **C:** Model size does not determine whether the scale adapts.
- **D:** Clipping runs after the overflow check; it cannot cause or fix overflow.

</details>

## 7. Multiple choice (medium)

Why does bf16 training usually not need a loss scaler while fp16 does?

- **A.** bf16 has more mantissa bits than fp16.
- **B.** bf16 has 8 exponent bits, the same range as fp32, so small gradients do not underflow and large values rarely overflow.
- **C.** bf16 computes gradients in fp32 automatically.
- **D.** bf16 only works with SGD, which does not need scaling.

<details>
<summary>Answer</summary>

**B.** bf16 has 8 exponent bits, the same range as fp32, so small gradients do not underflow and large values rarely overflow.

fp16 has 5 exponent bits (max 65,504, normals down to about $6 \times 10^{-5}$). bf16 trades mantissa (7 bits) for range (8 exponent bits).

- **A:** It is the opposite; bf16 has 7 mantissa bits, fp16 has 10.
- **C:** Autocast chooses per-operation precision, but bf16 tensors stay bf16.
- **D:** bf16 works with any optimizer, including AdamW.

</details>

## 8. Multiple choice (medium)

Why does mixed-precision training keep an fp32 master copy of the weights (the 4 bytes in "16 bytes per parameter")?

- **A.** To make checkpoint files smaller.
- **B.** Small updates vanish in 16-bit; for example 1 + 0.0001 rounds back to 1 in fp16, so updates must accumulate in fp32.
- **C.** Because GPUs cannot read 16-bit weights.
- **D.** To store the Adam moments.

<details>
<summary>Answer</summary>

**B.** Small updates vanish in 16-bit; for example 1 + 0.0001 rounds back to 1 in fp16, so updates must accumulate in fp32.

fp16's spacing near 1 is about 0.00098 and bf16's is about 0.0078. Typical updates are far smaller and would round away.

- **A:** It makes checkpoints bigger, not smaller.
- **C:** Tensor cores are built for 16-bit matrix multiplies.
- **D:** The Adam moments are separate, 4 more bytes each.

</details>

## 9. Multiple choice (hard)

A bf16 fine-tuning run (no loss scaler) produces a NaN loss on one particular batch every epoch. Which first step is most useful?

- **A.** Switch to fp16 with a GradScaler.
- **B.** Lower the clip threshold from 1.0 to 0.1.
- **C.** Check the loss with `torch.isfinite` before backward, log that batch's IDs, and inspect it (for example an empty sequence giving 0/0 or an out-of-range label).
- **D.** Double the learning rate to escape the bad region.

<details>
<summary>Answer</summary>

**C.** Check the loss with `torch.isfinite` before backward, log that batch's IDs, and inspect it (for example an empty sequence giving 0/0 or an out-of-range label).

A NaN tied to one batch points to the data or to a forward-pass overflow, not to gradient size. Skip and log the batch, then fix the data or the offending operation.

- **A:** fp16 has a narrower range than bf16, so it would make overflow more likely.
- **B:** Clipping acts on gradients after backward; a NaN loss already poisons them.
- **D:** A higher learning rate makes instability worse.

</details>

## 10. Reflection (medium)

The logged pre-clip gradient norm is above your threshold of 1.0 on 95% of steps. What does that mean, and what would you do?

<details>
<summary>Answer</summary>

**Model answer.** Clipping is no longer a safety belt for rare spikes; it is rescaling almost every update, which acts like a smaller and varying learning rate. I would check whether the learning rate is too high or warm-up too short, look for bad data or numerical issues raising the norm, and compare runs with a lower learning rate or a higher threshold. Only rare steps should clip.

Frequent clipping hides a learning-rate or data problem.

</details>
