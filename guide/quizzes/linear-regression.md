<!-- GENERATED from linear-regression.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Linear regression

Covers the lesson [Linear regression](../lessons/03-regression/01-linear-regression.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/linear-regression/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

A model predicts $\hat{y} = [3, 5, 4]$ for targets $y = [2, 5, 7]$. What is the mean squared error?

<details>
<summary>Answer</summary>

**3.3333** (within ±0.01)

Residuals $y - \hat{y}$ are $-1, 0, 3$. Squared: $1, 0, 9$. MSE $= (1 + 0 + 9)/3 = 10/3 \approx 3.333$.

</details>

## 2. Calculation (medium)

For $\hat{y} = wx + b$ with data $x = [1, 2]$, $y = [3, 5]$, and current $w = 1$, $b = 0$, compute $\partial\,\text{MSE}/\partial w$.

<details>
<summary>Answer</summary>

**-8** (within ±0.01)

Predictions are $1, 2$, so errors $\hat{y} - y$ are $-2, -3$.
$\frac{2}{n}\sum(\hat{y}_i - y_i)x_i = \frac{2}{2}[(-2)(1) + (-3)(2)] = 1 \times (-8) = -8$.
The gradient is negative, so gradient descent will increase $w$.

</details>

## 3. Calculation (medium)

Continuing the previous question: the bias gradient is $\partial\,\text{MSE}/\partial b = -5$. With learning rate $\eta = 0.1$, what is the new value of $b$ after one step (starting from $b = 0$)?

<details>
<summary>Answer</summary>

**0.5** (within ±0.001)

$b \leftarrow b - \eta \cdot \partial\text{MSE}/\partial b = 0 - 0.1 \times (-5) = 0.5$.

</details>

## 4. Multiple choice (medium)

You add one point far above the others and refit with MSE. What happens to the line, and why?

- **A.** Nothing changes; one point out of hundreds has no effect.
- **B.** The line tilts toward the outlier, because its large residual is squared and dominates the loss.
- **C.** The line moves away from the outlier to protect the other points.
- **D.** The slope becomes exactly zero.

<details>
<summary>Answer</summary>

**B.** The line tilts toward the outlier, because its large residual is squared and dominates the loss.

Squaring makes a residual of 10 cost 100, as much as a hundred residuals of 1. Minimizing MSE therefore pulls the fit toward the outlier. MAE or Huber loss are more robust choices when outliers are expected.

- **A:** A single large residual can dominate a sum of squares, even among many points.
- **B:** Correct. The squared term grows quadratically with distance.
- **C:** Least squares has no mechanism that pushes away from points; every residual pulls the line toward its point.
- **D:** The slope changes, but there is no reason for it to become zero.

</details>

## 5. Multiple choice (hard)

Your from-scratch gradient descent prints a loss of 52, then 410, then 3,300, then `inf`. Features are raw square footage (values around 2,000). What is the most likely fix?

- **A.** Train for more steps so it converges.
- **B.** Standardize the feature or lower the learning rate; the steps are overshooting.
- **C.** Remove the bias term.
- **D.** Switch the loss from MSE to MAE.

<details>
<summary>Answer</summary>

**B.** Standardize the feature or lower the learning rate; the steps are overshooting.

A loss that grows geometrically means each update overshoots the minimum by more than it started. The gradient with respect to $w$ scales with $x$, so raw values around 2,000 make the steps enormous. Standardizing $x$ (mean 0, std 1) or shrinking $\eta$ fixes it.

- **A:** More steps of a diverging process only diverge further.
- **B:** Correct. Scale and learning rate together set the step size.
- **C:** The bias is not the cause; the weight gradient scales with the large feature values.
- **D:** MAE changes robustness to outliers but does not fix a step size that is too large.

</details>

## 6. Select all that apply (medium)

A residual plot (residual vs. prediction) shows a clear U-shape, and the spread of residuals grows with the prediction. Which conclusions are justified? Select all that apply.

- **A.** The relationship is probably nonlinear; consider polynomial features or a tree model.
- **B.** Error variance is not constant; predicting log(y) may help.
- **C.** The model is overfitting.
- **D.** The data has leakage.

<details>
<summary>Answer</summary>

**A, B**

A U-shape means systematic error that depends on the prediction, so a straight line is missing curvature. A fan shape means heteroscedasticity. Neither pattern is evidence of overfitting (compare train vs. validation error for that) or leakage (look for features unavailable at prediction time).

- **A:** Correct. Curvature in residuals means a missing nonlinear term.
- **B:** Correct. Growing spread is heteroscedasticity; a log transform often stabilizes it.
- **C:** Overfitting is diagnosed by a gap between training and validation error, not by residual shape.
- **D:** Nothing in a residual plot indicates leakage.

</details>

## 7. Match (easy)

Match each term to its role.

| Concept | Options |
|---|---|
| MSE during training | Parameter learned from data |
| RMSE in a report | Hyperparameter chosen before training |
| Learning rate | Metric in the target's units for humans |
| Slope w | Loss the optimizer minimizes |

<details>
<summary>Answer</summary>

- MSE during training → Loss the optimizer minimizes
- RMSE in a report → Metric in the target's units for humans
- Learning rate → Hyperparameter chosen before training
- Slope w → Parameter learned from data

Loss is what training optimizes; a metric is how people judge the result (they can coincide, but serve different purposes). Parameters are learned; hyperparameters are set outside training.

</details>

## 8. Arrange in order (easy)

Put one iteration of gradient descent for linear regression in order.

- Update parameters by stepping against the gradients
- Compute gradients $\partial L/\partial w$ and $\partial L/\partial b$
- Compute residuals and the loss
- Compute predictions $\hat{y} = wx + b$

<details>
<summary>Answer</summary>

1. Compute predictions $\hat{y} = wx + b$
2. Compute residuals and the loss
3. Compute gradients $\partial L/\partial w$ and $\partial L/\partial b$
4. Update parameters by stepping against the gradients

Forward pass, measure, differentiate, update. This is the same loop that trains neural networks, with backpropagation computing the gradients.

</details>

## 9. Reflection (hard)

A house-price model has a large positive weight on "has a swimming pool". A product manager wants to tell sellers that adding a pool raises the price by that amount. What would you say?

<details>
<summary>Answer</summary>

**Model answer.** The weight measures association in the training data while holding the other included features fixed. It is not a causal effect. Houses with pools differ in ways the model may not capture (neighborhood, lot size, climate), so the weight absorbs those differences. Estimating the effect of adding a pool needs causal methods (comparable before/after sales, controlled comparisons), not a regression coefficient.

Correlation versus causation: a predictive model can be accurate while its coefficients have no interventional meaning.

</details>
