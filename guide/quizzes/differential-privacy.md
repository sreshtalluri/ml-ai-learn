<!-- GENERATED from differential-privacy.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Differential privacy

Covers the lesson [Differential privacy](../lessons/21-safety-security/03-differential-privacy.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/differential-privacy/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What does (ε, δ)-differential privacy guarantee?

- **A.** The output never contains any personal data
- **B.** For datasets differing in one record, the probability of any output changes by at most a factor $e^\varepsilon$, except with probability δ
- **C.** Training data is encrypted at rest
- **D.** The model cannot learn any fact from the data

<details>
<summary>Answer</summary>

**B.** For datasets differing in one record, the probability of any output changes by at most a factor $e^\varepsilon$, except with probability δ

DP bounds how much the output can depend on any one record, so observers learn little about whether you were in the data.

- **A:** DP bounds influence; the output can still describe patterns in the data.
- **B:** Correct.
- **C:** Encryption protects storage, not what a released output reveals.
- **D:** DP allows learning population-level patterns; that is its purpose.

</details>

## 2. Calculation (easy)

You release a count (sensitivity 1) with the Laplace mechanism at ε = 0.25. What is the noise scale b?

<details>
<summary>Answer</summary>

**4**

$b = \Delta / \varepsilon = 1 / 0.25 = 4$.

</details>

## 3. Arrange in order (easy)

Put the steps of one DP-SGD update in order.

- Update the parameters and record the step's privacy cost
- Sum the clipped gradients and add Gaussian noise with standard deviation σC
- Clip each per-example gradient to norm C
- Compute each example's gradient separately

<details>
<summary>Answer</summary>

1. Compute each example's gradient separately
2. Clip each per-example gradient to norm C
3. Sum the clipped gradients and add Gaussian noise with standard deviation σC
4. Update the parameters and record the step's privacy cost

Per-example gradients, clip, noise the sum, step and account.

</details>

## 4. Multiple choice (medium)

Why does DP-SGD clip each example's gradient rather than the batch gradient?

- **A.** To speed up training
- **B.** To bound how much any one example can change the sum, which sets the sensitivity the noise must cover
- **C.** To make the loss convex
- **D.** Clipping the batch gradient gives the same guarantee

<details>
<summary>Answer</summary>

**B.** To bound how much any one example can change the sum, which sets the sensitivity the noise must cover

Per-example clipping caps one example's contribution at C, so noise proportional to C hides it. A batch clip lets one outlier still dominate.

- **A:** Per-example gradients are slower, not faster.
- **B:** Correct.
- **C:** Clipping does not change the shape of the loss.
- **D:** Batch clipping does not bound any single example's influence.

</details>

## 5. Calculation (hard)

Full-batch DP gradient descent runs $T = 100$ steps with $\sigma = 16$ and $\delta = 10^{-5}$. Using $\varepsilon = \min_\alpha \left[\frac{T\alpha}{2\sigma^2} + \frac{\ln(1/\delta)}{\alpha-1}\right]$, compute ε to one decimal place. (Hint: the minimum is at $\alpha - 1 = \sqrt{\ln(10^5) \cdot 2\sigma^2 / T}$.)

<details>
<summary>Answer</summary>

**3.2** (within ±0.05)

$\frac{T}{2\sigma^2} = \frac{100}{512} = 0.1953$. $\alpha - 1 = \sqrt{11.513 / 0.1953} = \sqrt{58.95} = 7.678$, so $\alpha = 8.678$. $\varepsilon = 0.1953 \times 8.678 + 11.513 / 7.678 = 1.695 + 1.499 = 3.19$.

</details>

## 6. Multiple choice (medium)

At the same σ, C, and number of steps, why does DP training lose less accuracy on 20,000 examples than on 200?

- **A.** ε is smaller with more data in full-batch training
- **B.** The noise is added to the gradient sum and does not grow with n, so after dividing by n it is 100 times smaller relative to the signal
- **C.** Larger datasets need no clipping
- **D.** The accountant ignores large datasets

<details>
<summary>Answer</summary>

**B.** The noise is added to the gradient sum and does not grow with n, so after dividing by n it is 100 times smaller relative to the signal

The noise standard deviation in the averaged gradient is $\sigma C / n$.

- **A:** In the full-batch accountant, ε depends on σ, T, and δ, not on n.
- **B:** Correct.
- **C:** Clipping is still required to bound sensitivity.
- **D:** The accountant applies to every run regardless of size.

</details>

## 7. Multiple choice (hard)

A DP fine-tuned model (example-level, ε = 3) still reproduces one customer's API key. Which is the most plausible explanation?

- **A.** DP guarantees fail randomly
- **B.** The key appears in many of that customer's records, so example-level DP does not bound what the model learns about it; user-level DP and secret scrubbing are needed
- **C.** ε = 3 means nothing is protected
- **D.** The model was too small

<details>
<summary>Answer</summary>

**B.** The key appears in many of that customer's records, so example-level DP does not bound what the model learns about it; user-level DP and secret scrubbing are needed

DP protects each unit of privacy. When one person contributes many records, protect the person (user-level DP) and scrub secrets before training.

- **A:** The failure probability δ is tiny and doesn't explain a reproducible leak.
- **B:** Correct.
- **C:** ε = 3 is a meaningful bound per record; the issue is the unit of privacy.
- **D:** Smaller models also memorize repeated strings.

</details>

## 8. Multiple choice (easy)

You trained a model with DP at ε = 2. Publishing its generated text costs how much additional privacy budget?

- **A.** None, because post-processing a DP output is free
- **B.** ε = 2 for each generated sample
- **C.** It depends on the temperature
- **D.** It voids the guarantee entirely

<details>
<summary>Answer</summary>

**A.** None, because post-processing a DP output is free

Anything computed from a DP output without touching the data again is just as private.

- **A:** Correct.
- **B:** The budget was spent during training; sampling from the model doesn't touch the data again.
- **C:** Sampling settings are post-processing too.
- **D:** Post-processing cannot weaken a DP guarantee.

</details>

## 9. Multiple choice (medium)

A RAG assistant retrieves customer records at query time and the model was trained with DP. Which risk does DP training NOT address?

- **A.** The model memorizing a record from its fine-tuning data
- **B.** A user tricking the assistant into revealing another customer's retrieved record
- **C.** Membership inference on the fine-tuning set
- **D.** Verbatim regurgitation of a fine-tuning example

<details>
<summary>Answer</summary>

**B.** A user tricking the assistant into revealing another customer's retrieved record

Data supplied at inference time never went through DP training; protect it with access control and output checks.

- **A:** That is exactly what DP training bounds.
- **B:** Correct.
- **C:** DP bounds membership inference on the training set.
- **D:** DP bounds this for training records.

</details>
