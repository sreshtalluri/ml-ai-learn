<!-- GENERATED from probability-and-statistics.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Probability and statistics

Covers the lesson [Probability and statistics](../lessons/00-foundations/03-probability-and-statistics.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/probability-and-statistics/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

2% of transactions are fraud. A detector flags 90% of fraud and 10% of legitimate transactions. What is $P(\text{fraud} \mid \text{flagged})$? (3 decimals)

<details>
<summary>Answer</summary>

**0.155** (within ±0.002)

$\frac{0.9 \times 0.02}{0.9 \times 0.02 + 0.1 \times 0.98} = \frac{0.018}{0.018 + 0.098} = \frac{0.018}{0.116} = 0.155$.

</details>

## 2. Calculation (easy)

A bet pays 10 with probability 0.2 and loses 3 with probability 0.8. What is its expected value?

<details>
<summary>Answer</summary>

**-0.4** (within ±0.0001)

$10 \times 0.2 + (-3) \times 0.8 = 2 - 2.4 = -0.4$.

</details>

## 3. Calculation (medium)

What is the variance of a Bernoulli variable with $p = 0.3$?

<details>
<summary>Answer</summary>

**0.21** (within ±0.0001)

$p(1-p) = 0.3 \times 0.7 = 0.21$.

</details>

## 4. Multiple choice (medium)

"95% of people with the disease test positive." Which quantity is that?

- **A.** P(disease | positive)
- **B.** P(positive | disease)
- **C.** P(disease)
- **D.** P(positive)

<details>
<summary>Answer</summary>

**B.** P(positive | disease)

It conditions on having the disease. Confusing it with P(disease | positive) is the prosecutor's fallacy.

</details>

## 5. Multiple choice (medium)

Users who enable dark mode churn 30% less. A PM wants to force dark mode on everyone to cut churn. What's the best response?

- **A.** Ship it; the correlation is strong.
- **B.** The correlation may come from engaged users being more likely to explore settings; run an A/B test to measure the causal effect.
- **C.** Correlation always implies causation for large samples.
- **D.** Remove dark mode.

<details>
<summary>Answer</summary>

**B.** The correlation may come from engaged users being more likely to explore settings; run an A/B test to measure the causal effect.

Engagement is a plausible confounder. Only an intervention (randomized experiment) isolates the causal effect.

</details>

## 6. Multiple choice (hard)

Model A scores 0.872 and model B 0.865 accuracy on the same 500-example test set. What should you conclude?

- **A.** A is clearly better.
- **B.** The difference (0.007) is well within sampling noise (standard error about 0.015); compare with a paired test or a larger set before deciding.
- **C.** B is better because it is simpler.
- **D.** The test set must be broken.

<details>
<summary>Answer</summary>

**B.** The difference (0.007) is well within sampling noise (standard error about 0.015); compare with a paired test or a larger set before deciding.

$\sqrt{0.87 \times 0.13 / 500} \approx 0.015$. A 0.7-point gap is not evidence of a real difference. Paired comparisons (on which examples each gets right) are more sensitive.

</details>
