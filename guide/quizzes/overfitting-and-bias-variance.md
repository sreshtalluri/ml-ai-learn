<!-- GENERATED from overfitting-and-bias-variance.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Overfitting and the bias-variance trade-off

Covers the lesson [Overfitting and the bias-variance trade-off](../lessons/02-ml-workflow/02-overfitting-and-bias-variance.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/overfitting-and-bias-variance/) grades these interactively and tracks a review queue.

## 1. Match (easy)

Match each pair of errors to its diagnosis.

| Concept | Options |
|---|---|
| Train 0.30, validation 0.31 (target ≈ 0.05) | Good fit |
| Train 0.01, validation 0.40 | Overfitting (high variance) |
| Train 0.06, validation 0.07 | Underfitting (high bias) |

<details>
<summary>Answer</summary>

- Train 0.30, validation 0.31 (target ≈ 0.05) → Underfitting (high bias)
- Train 0.01, validation 0.40 → Overfitting (high variance)
- Train 0.06, validation 0.07 → Good fit

Look at both the level and the gap.

</details>

## 2. Calculation (medium)

Predictions at a point across four training sets are 2, 4, 6, 8. What is the variance (divide by 4)?

<details>
<summary>Answer</summary>

**5**

Mean 5; squared deviations 9, 1, 1, 9; average 20/4 = 5.

</details>

## 3. Calculation (medium)

The same predictions (2, 4, 6, 8) with truth 7. What is bias²?

<details>
<summary>Answer</summary>

**4**

Mean prediction 5; $(7 - 5)^2 = 4$.

</details>

## 4. Select all that apply (medium)

A model overfits. Which remedies are appropriate? Select all that apply.

- **A.** Collect more training data
- **B.** Increase regularization strength
- **C.** Increase polynomial degree
- **D.** Use early stopping

<details>
<summary>Answer</summary>

**A, B, D**

More data, regularization, and early stopping reduce variance. Higher degree increases it.

</details>

## 5. Multiple choice (hard)

Training and validation error have converged at the same high value as the training set grew. What will doubling the data likely do?

- **A.** Fix the problem.
- **B.** Very little; the model is bias-limited, so improve features or model capacity.
- **C.** Cause overfitting.
- **D.** Increase training error to the validation error.

<details>
<summary>Answer</summary>

**B.** Very little; the model is bias-limited, so improve features or model capacity.

Converged, high curves mean the model can't represent the pattern. More of the same data won't change that.

</details>

## 6. Reflection (medium)

Explain why a model with lower training loss can have worse test performance.

<details>
<summary>Answer</summary>

**Model answer.** Training loss rewards fitting the specific training examples, including their noise. A more flexible model can keep lowering training loss by memorizing noise, which doesn't repeat in new data, so its predictions on unseen examples get worse. That is overfitting: variance rises faster than bias falls.

Generalization is measured on unseen data; training loss can keep falling while validation loss rises.

</details>
