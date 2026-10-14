<!-- GENERATED from numerical-stability.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Numerical stability

Covers the lesson [Numerical stability](../lessons/00-foundations/06-numerical-stability.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/numerical-stability/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What is the approximate largest finite value in float32?

- **A.** $1.8 \times 10^{308}$
- **B.** $3.4 \times 10^{38}$
- **C.** $10^{10}$
- **D.** 65,504

<details>
<summary>Answer</summary>

**B.** $3.4 \times 10^{38}$

float32 has 8 exponent bits, giving a max of about $3.4 \times 10^{38}$.

- **A:** That is float64's max (11 exponent bits).
- **B:** Correct.
- **C:** No standard float format tops out there.
- **D:** That is float16's max (5 exponent bits).

</details>

## 2. Multiple choice (easy)

Why does `0.1 + 0.2 == 0.3` return False in Python?

- **A.** The CPU's adder has a bug
- **B.** 0.1 and 0.2 have no exact binary representation, so each is stored rounded and the rounded sum differs from rounded 0.3
- **C.** Python rounds decimals to integers
- **D.** 0.3 is not a valid float

<details>
<summary>Answer</summary>

**B.** 0.1 and 0.2 have no exact binary representation, so each is stored rounded and the rounded sum differs from rounded 0.3

0.1 is a repeating fraction in base 2, like 1/3 in base 10. The sum is 0.30000000000000004. Compare floats with a tolerance, such as `math.isclose`.

- **A:** The hardware follows IEEE 754 exactly; this is the specified behaviour.
- **B:** Correct.
- **C:** Python floats keep about 16 significant digits.
- **D:** 0.3 is stored fine, just also rounded, and to a different neighbour than the sum.

</details>

## 3. Multiple choice (medium)

In a stable softmax, why subtract max(z) from the logits before exponentiating?

- **A.** It normalizes the logits to zero mean
- **B.** It converts the logits to float16
- **C.** It makes the output more uniform
- **D.** The largest exponent becomes $e^0 = 1$, so nothing overflows, and the output is unchanged because softmax is shift-invariant

<details>
<summary>Answer</summary>

**D.** The largest exponent becomes $e^0 = 1$, so nothing overflows, and the output is unchanged because softmax is shift-invariant

$\text{softmax}(z - c) = \text{softmax}(z)$ for any constant $c$. With $c = \max z$ every term is at most 1 and the denominator is at least 1.

- **A:** Subtracting the max doesn't center the logits; subtracting the mean would, and it would not guarantee no overflow.
- **B:** It changes no types.
- **C:** The probabilities are identical to the exact softmax.
- **D:** Correct.

</details>

## 4. Calculation (medium)

Compute $\log(e^{3} + e^{1} + e^{-2})$ using the max trick. Give 3 decimals.

<details>
<summary>Answer</summary>

**3.133** (within ±0.002)

$m = 3$. $e^{0} + e^{-2} + e^{-5} = 1 + 0.1353 + 0.0067 = 1.1420$. $\ln 1.1420 = 0.1328$. LSE $= 3.1328$.

</details>

## 5. Multiple choice (medium)

You use $h = 10^{-15}$ in a centered-difference gradient check in float64. What happens?

- **A.** The estimate becomes more accurate
- **B.** The gradient becomes exactly 0
- **C.** $f(x+h)$ and $f(x-h)$ agree in almost every digit, so their difference is mostly rounding noise and the estimate is badly wrong
- **D.** The function evaluates faster

<details>
<summary>Answer</summary>

**C.** $f(x+h)$ and $f(x-h)$ agree in almost every digit, so their difference is mostly rounding noise and the estimate is badly wrong

Error behaves like $h^2 + \varepsilon/h$. At $h = 10^{-15}$ the $\varepsilon/h$ term is about 0.2. The sweet spot is near $\varepsilon^{1/3} \approx 6 \times 10^{-6}$.

- **A:** Only the truncation error shrinks; rounding error grows as $1/h$ and dominates.
- **B:** It can be 0 by accident, but generally it's noise of either sign.
- **C:** Correct.
- **D:** Speed doesn't depend on $h$.

</details>

## 6. Multiple choice (medium)

Why is bfloat16 usually preferred over float16 for training?

- **A.** bf16 uses less memory
- **B.** bf16 has float32's exponent range, so large activations don't overflow and small gradients don't flush to zero, usually without loss scaling
- **C.** bf16 has more fraction bits than fp16
- **D.** bf16 is supported by every GPU

<details>
<summary>Answer</summary>

**B.** bf16 has float32's exponent range, so large activations don't overflow and small gradients don't flush to zero, usually without loss scaling

bf16 has 8 exponent bits (max about $3.4 \times 10^{38}$); fp16 has 5 (max 65,504, smallest subnormal about $6 \times 10^{-8}$).

- **A:** Both are 16 bits.
- **B:** Correct.
- **C:** bf16 has 7 fraction bits; fp16 has 10. bf16 is less precise.
- **D:** Older GPUs (before NVIDIA Ampere, for example) lack fast bf16 support, which is when fp16 with a scaler is used.

</details>

## 7. Multiple choice (hard)

Your hand-written loss is `-torch.log(torch.softmax(logits, -1)[range(n), y]).mean()`. Training runs fine for a while, then the loss is `inf` and then `NaN`. What is the most likely cause and fix?

- **A.** The learning rate is too low; raise it
- **B.** A logit gap grew large enough that the true class's probability underflowed to 0, and log(0) is −inf; use F.cross_entropy on the logits
- **C.** Softmax overflowed because the max wasn't subtracted; torch.softmax doesn't do it
- **D.** The labels are one-hot encoded

<details>
<summary>Answer</summary>

**B.** A logit gap grew large enough that the true class's probability underflowed to 0, and log(0) is −inf; use F.cross_entropy on the logits

torch.softmax is already shift-stable, but a probability of $e^{-200}$ still rounds to 0 in float32. Computing log-softmax as $z_y - \text{LSE}(z)$ never forms the probability.

- **A:** A low LR doesn't produce inf; it just learns slowly.
- **B:** Correct.
- **C:** torch.softmax subtracts the max internally; overflow isn't the issue, underflow before the log is.
- **D:** The indexing with y implies integer labels, and wrong label format would fail immediately, not after a while.

</details>

## 8. Calculation (hard)

fp16's smallest positive subnormal is about $5.96 \times 10^{-8}$. A gradient of $2 \times 10^{-9}$ must be at least that size after scaling. What is the smallest power-of-two loss scale that achieves this?

<details>
<summary>Answer</summary>

**32**

Need $S \ge 5.96 \times 10^{-8} / 2 \times 10^{-9} = 29.8$. The smallest power of two above 29.8 is 32. (Real scalers start much larger, e.g. 65,536, and back off on overflow.)

</details>

## 9. Multiple choice (medium)

You need the variance of a float32 feature whose values are around 10,000 with spread around 1. Which approach do you use?

- **A.** $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ in float32, since it needs only one pass
- **B.** Subtract the mean first (two-pass or Welford's online algorithm), or accumulate in float64
- **C.** Cast to float16 to save memory
- **D.** Add 1e-8 to the result

<details>
<summary>Answer</summary>

**B.** Subtract the mean first (two-pass or Welford's online algorithm), or accumulate in float64

Both terms of the one-pass formula are about $10^8$, where float32 spacing is 8, so their difference is noise; the script gets −8 for a true 1.25. Centering first removes the large common part.

- **A:** This is catastrophic cancellation; it can even return a negative variance.
- **B:** Correct.
- **C:** float16 can't even represent the squares (max 65,504).
- **D:** An epsilon doesn't fix an error of ±8.

</details>
