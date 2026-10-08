<!-- GENERATED from support-vector-machines.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Support vector machines

Covers the lesson [Support vector machines](../lessons/05-instance-and-probabilistic/03-support-vector-machines.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/support-vector-machines/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

A linear SVM has $w = (6, 8)$. What is the margin width $2/\lVert w\rVert$?

<details>
<summary>Answer</summary>

**0.2** (within ±0.0001)

$\lVert w\rVert = \sqrt{36 + 64} = 10$; $2/10 = 0.2$.

</details>

## 2. Calculation (medium)

For $y = -1$ and $f(x) = 0.3$, what is the hinge loss $\max(0, 1 - y f(x))$?

<details>
<summary>Answer</summary>

**1.3** (within ±0.0001)

$y f(x) = -0.3$; $1 - (-0.3) = 1.3$. The point is on the wrong side.

</details>

## 3. Calculation (medium)

RBF kernel with $\gamma = 1$ between points at squared distance 2. What is $K$? (4 decimals)

<details>
<summary>Answer</summary>

**0.1353** (within ±0.0005)

$e^{-2} = 0.1353$.

</details>

## 4. Multiple choice (medium)

You increase C from 0.1 to 1000. What typically happens?

- **A.** The margin widens and the model becomes more regularized.
- **B.** The margin narrows as the model tries harder to classify every training point, risking overfitting.
- **C.** Nothing changes for a linear kernel.
- **D.** The number of support vectors always increases.

<details>
<summary>Answer</summary>

**B.** The margin narrows as the model tries harder to classify every training point, risking overfitting.

C is the price of each margin violation. A high price forces a narrower, more complex fit.

</details>

## 5. Multiple choice (easy)

A training point is correctly classified and well outside the margin. If you delete it and retrain, what happens to the boundary?

- **A.** It changes a lot.
- **B.** It doesn't change; only support vectors determine the solution.
- **C.** It flips.
- **D.** The margin doubles.

<details>
<summary>Answer</summary>

**B.** It doesn't change; only support vectors determine the solution.

Its hinge loss is zero, so it has no influence on the optimum.

</details>

## 6. Select all that apply (hard)

When is an SVM a reasonable choice? Select all that apply.

- **A.** Linear SVM on 200,000 TF-IDF features for text classification
- **B.** RBF SVM on 5,000 examples with a curved boundary
- **C.** RBF SVM on 50 million rows
- **D.** When you need well-calibrated probabilities out of the box

<details>
<summary>Answer</summary>

**A, B**

Kernel SVMs scale poorly to huge n, and SVM scores need extra calibration to be probabilities.

</details>
