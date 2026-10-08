<!-- GENERATED from backpropagation.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Backpropagation

Covers the lesson [Backpropagation](../lessons/11-gradient-descent-backprop/02-backpropagation.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/backpropagation/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

A sigmoid neuron with cross-entropy loss predicts $\hat{y} = 0.8$ for a label $y = 0$. What is $\partial L/\partial z$?

<details>
<summary>Answer</summary>

**0.8** (within ±0.0001)

$\hat{y} - y = 0.8 - 0 = 0.8$.

</details>

## 2. Calculation (medium)

Same neuron, input $x_j = -3$ on that weight. What is $\partial L/\partial w_j$?

<details>
<summary>Answer</summary>

**-2.4** (within ±0.0001)

$(\partial L/\partial z) \times x_j = 0.8 \times (-3) = -2.4$.

</details>

## 3. Calculation (medium)

With $w_j = 0.5$ and learning rate $0.1$, what is $w_j$ after the update?

<details>
<summary>Answer</summary>

**0.74** (within ±0.0001)

$0.5 - 0.1 \times (-2.4) = 0.74$.

</details>

## 4. Calculation (medium)

Along a path, local derivatives are 0.5, 2, and 0.25, and the loss gradient at the end is 4. What is the gradient at the start of the path?

<details>
<summary>Answer</summary>

**1** (within ±0.0001)

$4 \times 0.25 \times 2 \times 0.5 = 1$. Backpropagation multiplies local derivatives.

</details>

## 5. Select all that apply (medium)

Which techniques help with vanishing gradients in deep networks? Select all that apply.

- **A.** ReLU activations in hidden layers
- **B.** Residual (skip) connections
- **C.** Replacing ReLU with sigmoid everywhere
- **D.** Normalization layers and careful initialization

<details>
<summary>Answer</summary>

**A, B, D**

Sigmoid's derivative is at most 0.25, which shrinks gradients layer after layer.

</details>

## 6. Reflection (hard)

The weights of the first three layers of your network never change during training, while later layers train normally. Name two likely causes and how to check.

<details>
<summary>Answer</summary>

**Model answer.** (1) Vanishing gradients: saturating activations or poor initialization shrink the gradient to near zero by the time it reaches early layers. Check by logging per-layer gradient norms. (2) A broken graph: an operation detaches the early layers (`.detach()`, converting to NumPy, `.item()`), or their parameters were frozen or not passed to the optimizer. Check `param.grad is None` and the optimizer's parameter list.

Per-layer gradient norms and checking that parameters are registered and attached are the standard first diagnostics.

</details>
