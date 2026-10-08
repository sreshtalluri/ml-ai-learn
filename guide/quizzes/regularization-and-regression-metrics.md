<!-- GENERATED from regularization-and-regression-metrics.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Regularization and regression metrics

Covers the lesson [Regularization and regression metrics](../lessons/03-regression/02-regularization-and-regression-metrics.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/regularization-and-regression-metrics/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

One feature with $s_{xx} = 4$ and $s_{xy} = 6$. What is the ridge weight $s_{xy}/(s_{xx} + \lambda)$ for $\lambda = 2$?

<details>
<summary>Answer</summary>

**1** (within ±0.0001)

$6 / (4 + 2) = 1$. The OLS weight would be $6/4 = 1.5$.

</details>

## 2. Calculation (medium)

Same data, lasso with $\lambda = 4$. What is $\max(|s_{xy}| - \lambda/2, 0)/s_{xx}$?

<details>
<summary>Answer</summary>

**1** (within ±0.0001)

$(6 - 2)/4 = 1$. With $\lambda \ge 12$ the weight would be exactly 0.

</details>

## 3. Calculation (easy)

Errors are $[3, -4, 0, 5]$. What is the RMSE? (3 decimals)

<details>
<summary>Answer</summary>

**3.536** (within ±0.002)

Squares $9, 16, 0, 25$ sum to 50; mean 12.5; $\sqrt{12.5} = 3.536$. (MAE would be 3.)

</details>

## 4. Match (medium)

Match each situation to the best-suited metric.

| Concept | Options |
|---|---|
| Delivery-time errors where every minute late costs the same | MAPE |
| Load forecasting where one huge miss causes a blackout | R² |
| Comparing against simply predicting the average | RMSE |
| Sales forecasts for products of very different sizes, never near zero | MAE |

<details>
<summary>Answer</summary>

- Delivery-time errors where every minute late costs the same → MAE
- Load forecasting where one huge miss causes a blackout → RMSE
- Comparing against simply predicting the average → R²
- Sales forecasts for products of very different sizes, never near zero → MAPE

Match the metric's penalty shape to the cost of errors.

</details>

## 5. Multiple choice (medium)

Why must features be standardized before ridge or lasso?

- **A.** It makes training faster but doesn't change the result.
- **B.** The penalty compares weight sizes, and a weight's size depends on its feature's units.
- **C.** Lasso cannot handle negative values.
- **D.** Standardization removes multicollinearity.

<details>
<summary>Answer</summary>

**B.** The penalty compares weight sizes, and a weight's size depends on its feature's units.

Without scaling, a feature in small units needs a big weight and gets penalized more, regardless of importance.

</details>

## 6. Multiple choice (hard)

Two features are nearly identical copies. What do ridge and lasso typically do?

- **A.** Both zero out both features.
- **B.** Ridge splits the weight between them; lasso tends to keep one and zero the other, somewhat arbitrarily.
- **C.** Lasso splits the weight; ridge keeps one.
- **D.** Both keep exactly the same weights as OLS.

<details>
<summary>Answer</summary>

**B.** Ridge splits the weight between them; lasso tends to keep one and zero the other, somewhat arbitrarily.

The squared penalty is smallest when weight is shared; the absolute penalty is indifferent to how weight is split and its corner at zero favors sparse solutions.

</details>
