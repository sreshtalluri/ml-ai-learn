<!-- GENERATED from python-toolkit.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: The Python toolkit

Covers the lesson [The Python toolkit](../lessons/00-foundations/04-python-toolkit.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/python-toolkit/) grades these interactively and tracks a review queue.

## 1. Fill in (medium)

What is the shape of `A + B` when `A.shape == (8, 1)` and `B.shape == (6,)`? Answer like `(a, b)`.

<details>
<summary>Answer</summary>

**(8, 6)** or **(8,6)** or **8,6** or **[8, 6]**

Align from the right; 1 stretches to 6, and the missing leading dimension of B becomes 1 and stretches to 8.

</details>

## 2. Fill in (easy)

For `X` of shape `(200, 5)`, what is the shape of `X.mean(axis=0)`? Answer like `(n,)`.

<details>
<summary>Answer</summary>

**(5,)** or **(5)** or **5** or **[5]**

Averaging over rows (axis 0) leaves one value per column.

</details>

## 3. Calculation (medium)

A column has values [2, 4, 6]. What is the standardized value of 6, using the population standard deviation? (4 decimals)

<details>
<summary>Answer</summary>

**1.2247** (within ±0.001)

Mean 4; deviations −2, 0, 2; variance $8/3$; std $1.633$. $(6-4)/1.633 = 1.2247$.

</details>

## 4. Multiple choice (medium)

Where should missing values be imputed (for example with the median)?

- **A.** On the full dataset before splitting, so all splits are consistent.
- **B.** Using statistics computed on the training split only, then applied to validation and test.
- **C.** Separately on each split with its own median.
- **D.** Never; drop every row with a missing value.

<details>
<summary>Answer</summary>

**B.** Using statistics computed on the training split only, then applied to validation and test.

Statistics from validation or test data leak information. Fit imputers inside the training pipeline.

</details>

## 5. Select all that apply (easy)

Which practices make an experiment reproducible? Select all that apply.

- **A.** Fixing random seeds
- **B.** Pinning package versions
- **C.** Recording the dataset version
- **D.** Re-running until the metric looks good

<details>
<summary>Answer</summary>

**A, B, C**

Seeds, versions, and data snapshots let anyone reproduce the run. Re-running until it looks good is cherry-picking.

</details>

## 6. Reflection (hard)

Your MSE computed with NumPy is far larger than scikit-learn's `mean_squared_error` on the same predictions. Predictions have shape `(n, 1)` and targets `(n,)`. Explain.

<details>
<summary>Answer</summary>

**Model answer.** `y_pred - y` broadcasts `(n, 1)` against `(n,)` into an `(n, n)` matrix of every prediction minus every target, so the mean is over n² meaningless differences. Flatten predictions with `.ravel()` (or reshape targets) before subtracting.

The classic silent broadcasting bug.

</details>
