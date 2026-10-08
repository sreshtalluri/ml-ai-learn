<!-- GENERATED from gradient-descent.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Gradient descent

Covers the lesson [Gradient descent](../lessons/11-gradient-descent-backprop/01-gradient-descent.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/gradient-descent/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

For $f(\theta) = (\theta - 3)^2$, starting at $\theta = 0$ with learning rate $\eta = 0.25$, what is $\theta$ after one gradient-descent step?

<details>
<summary>Answer</summary>

**1.5** (within ±0.001)

$f'(\theta) = 2(\theta - 3) = 2(0 - 3) = -6$. Update: $0 - 0.25 \times (-6) = 1.5$.

</details>

## 2. Fill in (medium)

$f(x, y) = x^2 + 3y^2$. At $(1, 1)$ with $\eta = 0.1$, what is the new point after one step? Answer as `(x, y)`.

<details>
<summary>Answer</summary>

**(0.8, 0.4)** or **(0.8,0.4)** or **0.8,0.4** or **0.8, 0.4**

$\nabla f = (2x, 6y) = (2, 6)$. New point $= (1 - 0.2,\; 1 - 0.6) = (0.8, 0.4)$.

</details>

## 3. Calculation (hard)

For $f(x, y) = \tfrac12(x^2 + 50y^2)$, what is the largest learning rate for which plain gradient descent converges (the strict upper limit)?

<details>
<summary>Answer</summary>

**0.04** (within ±0.0001)

Curvatures are 1 and 50. Each step multiplies $y$ by $1 - 50\eta$, which needs $|1 - 50\eta| < 1$, so $\eta < 2/50 = 0.04$. The steepest direction sets the limit.

</details>

## 4. Match (medium)

Match each training-loss curve to its most likely cause.

| Concept | Options |
|---|---|
| Loss falls, then shoots up to NaN | Overfitting, not an optimizer problem |
| Loss barely moves for thousands of steps | Constant learning rate too large near the minimum |
| Loss bounces around a level and never settles | Learning rate too low or no gradient flowing |
| Training loss falls, validation loss rises | Learning rate too high or exploding gradients |

<details>
<summary>Answer</summary>

- Loss falls, then shoots up to NaN → Learning rate too high or exploding gradients
- Loss barely moves for thousands of steps → Learning rate too low or no gradient flowing
- Loss bounces around a level and never settles → Constant learning rate too large near the minimum
- Training loss falls, validation loss rises → Overfitting, not an optimizer problem

Reading curves is a core debugging skill. Note the last one: the optimizer is doing its job; the model is memorizing.

</details>

## 5. Multiple choice (medium)

Why is mini-batch gradient descent the default rather than full-batch?

- **A.** It computes the exact gradient more cheaply.
- **B.** Many cheap, slightly noisy steps make more progress per unit of compute, and batches parallelize well on GPUs.
- **C.** Full-batch gradient descent cannot reach a minimum.
- **D.** Mini-batches remove the need for a learning rate.

<details>
<summary>Answer</summary>

**B.** Many cheap, slightly noisy steps make more progress per unit of compute, and batches parallelize well on GPUs.

A mini-batch gradient is an unbiased but noisy estimate. You get many updates per epoch instead of one, and hardware processes a batch in parallel.

- **A:** It estimates, rather than computes exactly.
- **B:** Correct.
- **C:** Full-batch can converge; it is just slow on large datasets.
- **D:** Every variant needs a learning rate.

</details>

## 6. Multiple choice (hard)

Parameter A has gradients around 100, parameter B around 0.01. With plain SGD at one learning rate, A's steps are 10,000× larger. What does Adam do differently?

- **A.** It clips both gradients to 1.
- **B.** It divides each parameter's averaged gradient by the root of its averaged squared gradient, so step sizes are comparable.
- **C.** It uses the second derivative (Hessian) to pick the exact step.
- **D.** It skips updates for parameters with small gradients.

<details>
<summary>Answer</summary>

**B.** It divides each parameter's averaged gradient by the root of its averaged squared gradient, so step sizes are comparable.

Adam's update is roughly $\eta\, \hat m / \sqrt{\hat s}$, which is about $\eta$ in magnitude for steady gradients, whatever their scale.

- **A:** That is gradient clipping, a different technique.
- **B:** Correct.
- **C:** Adam is a first-order method; it never computes the Hessian.
- **D:** Every parameter is updated every step.

</details>

## 7. Multiple choice (medium)

A PyTorch training loop omits `optimizer.zero_grad()`. Loss behaves erratically and diverges. Why?

- **A.** PyTorch adds new gradients to the stored ones, so each step uses the sum of all past gradients.
- **B.** The optimizer stops updating weights.
- **C.** The forward pass reuses the previous batch.
- **D.** Nothing; `zero_grad` is optional.

<details>
<summary>Answer</summary>

**A.** PyTorch adds new gradients to the stored ones, so each step uses the sum of all past gradients.

`.backward()` accumulates into `.grad`. Without clearing, the effective gradient grows every step, acting like an exploding learning rate.

- **A:** Correct.
- **B:** The optimizer still steps; it steps with the wrong gradients.
- **C:** The data loader is unaffected.
- **D:** It is required unless you deliberately accumulate gradients across batches.

</details>

## 8. Reflection (medium)

Explain to a new engineer why standardizing input features usually lets a model train faster with gradient descent.

<details>
<summary>Answer</summary>

**Model answer.** Features on very different scales make the loss surface a long, narrow valley: the loss is far more sensitive to some weights than others. Gradient descent must use a learning rate small enough for the steepest direction, so it crawls along the gentle ones. Standardizing makes the curvature more similar in every direction (a rounder bowl), so a single larger learning rate works for all weights.

This is the conditioning argument from the worked example, where the steepest curvature sets the speed limit $\eta < 2/\lambda_{\max}$.

</details>
