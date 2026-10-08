<!-- GENERATED from logistic-regression.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Logistic regression

Covers the lesson [Logistic regression](../lessons/04-classification/01-logistic-regression.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/logistic-regression/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

A logistic model has logit $z = 0$. What probability does it predict?

<details>
<summary>Answer</summary>

**0.5** (within ±0.0001)

$\sigma(0) = 1/(1 + e^0) = 1/2$.

</details>

## 2. Calculation (medium)

The model predicts $\hat{p} = 0.8$ for an example whose true label is 0. What is the binary cross-entropy (natural log, 3 decimals)?

<details>
<summary>Answer</summary>

**1.609** (within ±0.002)

$-\ln(1 - 0.8) = -\ln 0.2 = 1.609$.

</details>

## 3. Calculation (medium)

For $\hat{p} = 0.7$, $y = 1$, and feature value $x = 4$, what is $\partial L/\partial w$?

<details>
<summary>Answer</summary>

**-1.2** (within ±0.0001)

$(\hat{p} - y)x = (0.7 - 1)(4) = -1.2$.

</details>

## 4. Multiple choice (medium)

A feature's weight is $\ln 2 \approx 0.693$. What does a one-unit increase in that feature do?

- **A.** Adds 0.693 to the probability.
- **B.** Doubles the odds p/(1 − p).
- **C.** Doubles the probability.
- **D.** Nothing; weights aren't interpretable.

<details>
<summary>Answer</summary>

**B.** Doubles the odds p/(1 − p).

The logit is the log-odds, so adding $\ln 2$ multiplies the odds by $e^{\ln 2} = 2$. The probability change depends on where you start.

</details>

## 5. Multiple choice (medium)

Why does plain logistic regression fail on XOR-shaped data?

- **A.** It needs more training steps.
- **B.** Its decision boundary $w \cdot x + b = 0$ is a straight line, which can't separate XOR.
- **C.** Cross-entropy is undefined for XOR.
- **D.** It only works on balanced data.

<details>
<summary>Answer</summary>

**B.** Its decision boundary $w \cdot x + b = 0$ is a straight line, which can't separate XOR.

Add an interaction feature ($x_1 x_2$) or use a nonlinear model.

</details>

## 6. Calculation (hard)

Logits are $[\ln 4, 0, 0]$. What probability does softmax assign to the first class?

<details>
<summary>Answer</summary>

**0.6667** (within ±0.001)

Exponentials $[4, 1, 1]$, sum 6, so $4/6 = 0.667$.

</details>
