<!-- GENERATED from pca.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: PCA, t-SNE, and UMAP

Covers the lesson [PCA, t-SNE, and UMAP](../lessons/08-dimensionality-reduction/01-pca.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/pca/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

Eigenvalues of a covariance matrix are 6, 3, and 1. What fraction of variance do the first two components explain?

<details>
<summary>Answer</summary>

**0.9** (within ±0.0001)

$(6 + 3)/(6 + 3 + 1) = 0.9$.

</details>

## 2. Calculation (hard)

For covariance $\begin{bmatrix}3 & 1\\1 & 3\end{bmatrix}$, what is the largest eigenvalue?

<details>
<summary>Answer</summary>

**4** (within ±0.0001)

Trace 6, determinant 8: $\lambda = (6 \pm \sqrt{36 - 32})/2 = (6 \pm 2)/2$, so 4 and 2. PC1 points along $(1, 1)/\sqrt{2}$.

</details>

## 3. Fill in (easy)

Center the points $(1, 4)$, $(3, 8)$. What is the centered version of the first point? Answer like `(a, b)`.

<details>
<summary>Answer</summary>

**(-1, -2)** or **(-1,-2)** or **-1,-2** or **-1, -2**

The mean is $(2, 6)$, so $(1 - 2, 4 - 6) = (-1, -2)$.

</details>

## 4. Multiple choice (medium)

Features are income in dollars and age in years. You run PCA without standardizing. What will PC1 mostly be?

- **A.** An even mix of income and age.
- **B.** Essentially the income axis, because its variance is millions of times larger.
- **C.** Essentially the age axis.
- **D.** Undefined.

<details>
<summary>Answer</summary>

**B.** Essentially the income axis, because its variance is millions of times larger.

PCA maximizes variance, and raw income variance dwarfs age variance. Standardize when units differ.

</details>

## 5. Select all that apply (hard)

Which conclusions from a t-SNE plot are justified? Select all that apply.

- **A.** Points that are close together are likely neighbors in the original space.
- **B.** A cluster twice as large on the plot has twice the spread in the original space.
- **C.** Two clusters far apart on the plot are more different than two nearby clusters.
- **D.** Visible islands are hypotheses to test, not proven categories.

<details>
<summary>Answer</summary>

**A, D**

t-SNE preserves local neighborhoods; sizes and inter-cluster distances are not reliable.

</details>

## 6. Reflection (medium)

Give one reason to apply PCA before training a model, and one risk.

<details>
<summary>Answer</summary>

**Model answer.** Reason: compress many correlated features into fewer components, reducing overfitting and training time (and removing multicollinearity). Risk: PCA keeps the highest-variance directions, which may not be the ones that predict the target, so it can throw away useful low-variance signal; components are also harder to interpret.

Variance is not the same as task usefulness.

</details>
