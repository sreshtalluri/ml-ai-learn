<!-- GENERATED from training-and-regularization.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Training and regularization

Covers the lesson [Training and regularization](../lessons/12-training-regularization/01-training-and-regularization.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/training-and-regularization/) grades these interactively and tracks a review queue.

## 1. Fill in (medium)

Activations `[2, 6, 4]`, dropout $p = 0.5$, mask `[0, 1, 1]`. What is the training-time output? Answer like `[a, b, c]`.

<details>
<summary>Answer</summary>

**[0, 12, 8]** or **[0,12,8]** or **0,12,8** or **0, 12, 8**

Kept units are scaled by $1/(1 - 0.5) = 2$.

</details>

## 2. Calculation (medium)

Layer-normalize $[2, 4, 6]$ ($\gamma = 1$, $\beta = 0$, ignore $\epsilon$). What is the normalized value of 6? (3 decimals)

<details>
<summary>Answer</summary>

**1.225** (within ±0.002)

Mean 4; variance $(4 + 0 + 4)/3 = 2.667$; std 1.633; $(6 - 4)/1.633 = 1.225$.

</details>

## 3. Multiple choice (easy)

Validation loss is lowest at epoch 12 and rises afterwards while training loss keeps falling. Which weights should you deploy?

- **A.** The final epoch's
- **B.** Epoch 12's (the best validation checkpoint)
- **C.** Epoch 1's
- **D.** An average of all epochs

<details>
<summary>Answer</summary>

**B.** Epoch 12's (the best validation checkpoint)

That is early stopping: keep the checkpoint with the best validation performance.

</details>

## 4. Match (medium)

Match each technique to what it does.

| Concept | Options |
|---|---|
| Dropout | Starts with small steps to stabilize early training |
| Weight decay | Rescales each example's activations to zero mean, unit variance |
| Layer normalization | Shrinks weights toward zero each step |
| Learning-rate warmup | Randomly zeroes activations during training |

<details>
<summary>Answer</summary>

- Dropout → Randomly zeroes activations during training
- Weight decay → Shrinks weights toward zero each step
- Layer normalization → Rescales each example's activations to zero mean, unit variance
- Learning-rate warmup → Starts with small steps to stabilize early training

Different mechanisms, shared goal of stable training and better generalization.

</details>

## 5. Multiple choice (medium)

A model with dropout gives slightly different predictions every time you call it on the same input in production. What's wrong?

- **A.** The data loader shuffles.
- **B.** The model is still in training mode, so dropout is active; call `model.eval()`.
- **C.** Dropout should be increased.
- **D.** Floating-point noise.

<details>
<summary>Answer</summary>

**B.** The model is still in training mode, so dropout is active; call `model.eval()`.

Dropout must be disabled at inference.

</details>

## 6. Reflection (hard)

Why is random search often more efficient than grid search when only one or two of several hyperparameters really matter?

<details>
<summary>Answer</summary>

**Model answer.** A grid repeats the same few values of each hyperparameter many times. With 9 trials on a 3×3 grid you only test 3 values of the important one. Random search samples a new value of every hyperparameter in every trial, so 9 trials test 9 distinct values of the important one, making it more likely to land near its best value. It's a coverage argument, not a guarantee for every run.

Bergstra and Bengio's observation; Bayesian optimization goes further by modeling the score.

</details>
