<!-- GENERATED from calculus-for-ml.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Calculus for ML

Covers the lesson [Calculus for ML](../lessons/00-foundations/02-calculus-for-ml.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/calculus-for-ml/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

For $f(t) = 3t^2 - 2t + 7$, what is $f'(2)$?

<details>
<summary>Answer</summary>

**10**

$f'(t) = 6t - 2$, so $f'(2) = 12 - 2 = 10$.

</details>

## 2. Calculation (medium)

For $L(x, y) = x^2 y + 3y$, what is $\partial L / \partial y$ at $(2, 5)$?

<details>
<summary>Answer</summary>

**7**

Holding $x$ fixed, $\partial L/\partial y = x^2 + 3 = 4 + 3 = 7$.

</details>

## 3. Calculation (easy)

At what value of $t$ does $f(t) = t^2 - 6t + 1$ reach its minimum?

<details>
<summary>Answer</summary>

**3**

$f'(t) = 2t - 6 = 0$ gives $t = 3$.

</details>

## 4. Calculation (medium)

$L = z^2$ and $z = 3w + 1$. What is $dL/dw$ at $w = 1$?

<details>
<summary>Answer</summary>

**24**

$dL/dz = 2z = 2(4) = 8$; $dz/dw = 3$; product $= 24$.

</details>

## 5. Calculation (medium)

The sigmoid's derivative is $\sigma(z)(1 - \sigma(z))$. What is its maximum possible value?

<details>
<summary>Answer</summary>

**0.25** (within ±0.0001)

It peaks at $z = 0$ where $\sigma = 0.5$: $0.5 \times 0.5 = 0.25$. Stacking many sigmoids multiplies factors of at most 0.25, which is one cause of vanishing gradients.

</details>

## 6. Multiple choice (medium)

The gradient of a loss at the current parameters is $(3, -2)$. Which direction lowers the loss fastest (locally)?

- **A.** (3, -2)
- **B.** (-3, 2)
- **C.** (0, 0)
- **D.** (2, 3)

<details>
<summary>Answer</summary>

**B.** (-3, 2)

The gradient points uphill; the steepest descent direction is its negative.

- **A:** That is the steepest ascent direction.
- **B:** Correct.
- **C:** Not moving doesn't lower the loss.
- **D:** This direction is perpendicular to the gradient, so to first order it doesn't change the loss.

</details>
