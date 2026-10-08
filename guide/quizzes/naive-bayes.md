<!-- GENERATED from naive-bayes.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Naive Bayes

Covers the lesson [Naive Bayes](../lessons/05-instance-and-probabilistic/02-naive-bayes.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/naive-bayes/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

In spam, a word appears 4 times out of 96 total words. The vocabulary has 200 words. What is its Laplace-smoothed likelihood $(4 + 1)/(96 + 200)$? (4 decimals)

<details>
<summary>Answer</summary>

**0.0169** (within ±0.0002)

$5 / 296 = 0.0169$.

</details>

## 2. Calculation (medium)

Unnormalized scores are spam $6 \times 10^{-5}$ and ham $2 \times 10^{-5}$. What is $P(\text{spam} \mid x)$?

<details>
<summary>Answer</summary>

**0.75** (within ±0.0001)

$6 / (6 + 2) = 0.75$.

</details>

## 3. Multiple choice (medium)

Without smoothing, what happens if an email contains a word that never appeared in any training spam?

- **A.** The word is ignored.
- **B.** P(spam | email) becomes exactly 0, no matter how spammy the other words are.
- **C.** The model raises an error.
- **D.** The prior takes over.

<details>
<summary>Answer</summary>

**B.** P(spam | email) becomes exactly 0, no matter how spammy the other words are.

One zero factor makes the whole product zero.

</details>

## 4. Match (medium)

Match each feature type to a Naive Bayes variant.

| Concept | Options |
|---|---|
| Word counts in documents | Gaussian NB |
| Whether each word is present or absent | Bernoulli NB |
| Continuous sensor readings | Multinomial NB |

<details>
<summary>Answer</summary>

- Word counts in documents → Multinomial NB
- Whether each word is present or absent → Bernoulli NB
- Continuous sensor readings → Gaussian NB

The variant defines the per-feature likelihood distribution.

</details>

## 5. Multiple choice (medium)

Why do implementations add log-probabilities instead of multiplying probabilities?

- **A.** Logs make the model more accurate.
- **B.** Products of thousands of small probabilities underflow to 0 in floating point; sums of logs don't.
- **C.** Logs remove the need for smoothing.
- **D.** Logs make probabilities sum to 1.

<details>
<summary>Answer</summary>

**B.** Products of thousands of small probabilities underflow to 0 in floating point; sums of logs don't.

The argmax is unchanged because log is monotonic.

</details>

## 6. Reflection (hard)

A Naive Bayes spam filter outputs probabilities like 0.99999 and 0.00001 for most emails, even borderline ones. Explain why.

<details>
<summary>Answer</summary>

**Model answer.** The independence assumption treats correlated words as separate pieces of evidence, so related words ("free", "money", "offer") multiply the same signal several times. The product becomes extreme, pushing posteriors toward 0 or 1. The class ranking can still be good, but the probabilities are overconfident; calibrate them on held-out data if they are used as probabilities.

Double-counted evidence produces overconfident posteriors.

</details>
