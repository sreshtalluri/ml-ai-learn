<!-- GENERATED from vectors-and-matrices.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Vectors and matrices

Covers the lesson [Vectors and matrices](../lessons/00-foundations/01-vectors-and-matrices.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/vectors-and-matrices/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

Compute $[2, -1, 4] \cdot [3, 5, 0.5]$.

<details>
<summary>Answer</summary>

**3** (within ±0.0001)

$2 \times 3 + (-1) \times 5 + 4 \times 0.5 = 6 - 5 + 2 = 3$.

</details>

## 2. Fill in (easy)

$X$ has shape `[32, 128]` and $W$ has shape `[128, 10]`. What is the shape of $XW$? Answer like `[a, b]`.

<details>
<summary>Answer</summary>

**[32, 10]** or **[32,10]** or **32,10** or **(32, 10)**

The inner dimension 128 matches and disappears; the outer dimensions remain.

</details>

## 3. Calculation (medium)

$A = \begin{bmatrix}1 & 2\\ 0 & 3\end{bmatrix}$, $B = \begin{bmatrix}4 & 1\\ 2 & 5\end{bmatrix}$. What is the entry in row 2, column 2 of $AB$?

<details>
<summary>Answer</summary>

**15**

Row 2 of $A$ is $[0, 3]$, column 2 of $B$ is $[1, 5]$: $0 \times 1 + 3 \times 5 = 15$.

</details>

## 4. Calculation (medium)

What is the cosine similarity between $[1, 0]$ and $[1, 1]$? (4 decimals)

<details>
<summary>Answer</summary>

**0.7071** (within ±0.001)

Dot product 1; lengths 1 and $\sqrt{2}$; $1/\sqrt{2} = 0.7071$ (a 45° angle).

</details>

## 5. Multiple choice (hard)

`y_hat` has shape `[100, 1]` and `y` has shape `[100]`. You compute `((y_hat - y) ** 2).mean()` in NumPy. What happens?

- **A.** An error is raised.
- **B.** The correct MSE is returned.
- **C.** Broadcasting produces a `[100, 100]` matrix, so the "MSE" is wrong but no error is raised.
- **D.** The result is a `[100]` vector.

<details>
<summary>Answer</summary>

**C.** Broadcasting produces a `[100, 100]` matrix, so the "MSE" is wrong but no error is raised.

`[100, 1] - [100]` broadcasts to `[100, 100]`: every prediction minus every target. The mean of that is a meaningless number. Reshape one side first.

- **A:** NumPy broadcasts silently; that is the danger.
- **B:** It looks plausible but averages 10,000 cross pairs.
- **C:** Correct.
- **D:** `.mean()` reduces to a scalar, after the wrong broadcast.

</details>

## 6. Select all that apply (medium)

Which statements are true? Select all that apply.

- **A.** If $A$ is $[2, 3]$ and $B$ is $[3, 2]$, both $AB$ and $BA$ exist but have different shapes.
- **B.** Matrix multiplication is commutative.
- **C.** `A * B` in NumPy multiplies element-wise.
- **D.** A dot product of perpendicular vectors is 0.

<details>
<summary>Answer</summary>

**A, C, D**

$AB$ is $[2, 2]$ and $BA$ is $[3, 3]$. Matrix products are generally not commutative. `*` is element-wise; `@` is the matrix product.

</details>
