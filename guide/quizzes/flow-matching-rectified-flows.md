<!-- GENERATED from flow-matching-rectified-flows.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Flow matching and rectified flows

Covers the lesson [Flow matching and rectified flows](../lessons/13-deep-architectures/06-flow-matching-rectified-flows.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/flow-matching-rectified-flows/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

With $x_t = (1 - t)x_0 + t x_1$ ($x_0$ data, $x_1$ noise), what does a conditional flow-matching network regress onto?

- **A.** The noise $x_1$
- **B.** The velocity $x_1 - x_0$
- **C.** The data $x_0$
- **D.** The log-likelihood of $x_t$

<details>
<summary>Answer</summary>

**B.** The velocity $x_1 - x_0$

The time derivative of the straight-line interpolant is the constant $x_1 - x_0$.

- **A:** That's the epsilon-prediction target of DDPM; it carries the same information but is not the flow-matching target.
- **C:** That's the x0-prediction target.
- **D:** Flow matching never computes a likelihood during training.

</details>

## 2. Calculation (easy)

$x_0 = 2.0$, $x_1 = -0.5$, $t = 0.25$. What is $x_t = (1 - t)x_0 + t x_1$?

<details>
<summary>Answer</summary>

**1.375** (within ±0.001)

$0.75 \times 2.0 + 0.25 \times (-0.5) = 1.5 - 0.125 = 1.375$. The target velocity is $-0.5 - 2.0 = -2.5$.

</details>

## 3. Calculation (medium)

At $t = 0.6$, $x_t = 0.8$, the model predicts $v = -1.5$. What is its implied clean-data prediction $\hat{x}_0 = x_t - t\,v$?

<details>
<summary>Answer</summary>

**1.7** (within ±0.001)

$0.8 - 0.6 \times (-1.5) = 0.8 + 0.9 = 1.7$.

</details>

## 4. Calculation (hard)

A flow-matching model trained with independent (random) pairs reaches the ideal velocity field. The data is a 50/50 mixture of modes at 1 and 5. Starting from noise $x_1 = 0.7$ at $t = 1$, where does a single Euler step of size 1 land?

<details>
<summary>Answer</summary>

**3** (within ±0.001)

At $t = 1$, $v^*(x, 1) = x - \mathbb{E}[x_0] = 0.7 - 3 = -2.3$. One step gives $0.7 - (-2.3) = 3$, the data mean, which lies between the modes.

</details>

## 5. Multiple choice (medium)

A flow-matching model's training loss plateaus well above zero, yet 32-step samples match the data. Why?

- **A.** The model is underfitting and needs more layers
- **B.** Many pairs pass through the same $(x_t, t)$ with different velocities, so the best prediction is their average and the leftover variance can't be removed
- **C.** The learning rate is too high
- **D.** The noise is not Gaussian

<details>
<summary>Answer</summary>

**B.** Many pairs pass through the same $(x_t, t)$ with different velocities, so the best prediction is their average and the leftover variance can't be removed

Squared loss converges to the conditional mean; its floor is the conditional variance of $x_1 - x_0$ given $x_t$.

- **A:** Good samples show the field is learned; the floor is irreducible.
- **C:** A too-high learning rate shows up as noisy or diverging loss, not a stable floor with good samples.
- **D:** Gaussian noise doesn't remove crossing lines.

</details>

## 6. Multiple choice (medium)

What does one round of reflow (rectified flow) do?

- **A.** Trains with a larger learning rate
- **B.** Generates (noise, sample) pairs by solving the current model's ODE, then retrains on those coupled pairs so paths become straighter
- **C.** Adds a discriminator
- **D.** Doubles the number of sampling steps

<details>
<summary>Answer</summary>

**B.** Generates (noise, sample) pairs by solving the current model's ODE, then retrains on those coupled pairs so paths become straighter

Coupled pairs come from a non-crossing flow, so their straight lines rarely cross, and few-step sampling works.

- **A:** Learning rate doesn't change path geometry.
- **C:** That's adversarial training, a different technique.
- **D:** The goal is fewer steps, not more.

</details>

## 7. Multiple choice (hard)

Your flow-matching image model looks great at 30 steps but blurry and averaged at 1 step. What is the best explanation and fix?

- **A.** The decoder is broken; retrain the autoencoder
- **B.** With random pairs, paths curve and one big step heads to the conditional mean; use reflow, a better solver, or few-step distillation
- **C.** Guidance scale is too low
- **D.** The model needs more training on t near 0

<details>
<summary>Answer</summary>

**B.** With random pairs, paths curve and one big step heads to the conditional mean; use reflow, a better solver, or few-step distillation

One Euler step from noise approximates the data mean. Straighter paths or distillation fix it.

- **A:** The decoder works at 30 steps, so it isn't the problem.
- **C:** Guidance changes prompt adherence, not the curvature error of large steps.
- **D:** The problem is at large steps from t near 1, not accuracy near data.

</details>

## 8. Multiple choice (medium)

How does rectified flow relate to Gaussian diffusion $x_t = \alpha_t x_0 + \sigma_t \epsilon$?

- **A.** They are unrelated families of models
- **B.** Rectified flow is the special case $\alpha_t = 1 - t$, $\sigma_t = t$, with a velocity-prediction output; noise, data, and velocity predictions can be converted into each other
- **C.** Rectified flow requires adversarial training
- **D.** Diffusion is deterministic and rectified flow is stochastic

<details>
<summary>Answer</summary>

**B.** Rectified flow is the special case $\alpha_t = 1 - t$, $\sigma_t = t$, with a velocity-prediction output; noise, data, and velocity predictions can be converted into each other

$v = \epsilon - x_0$, $\hat{x}_0 = x_t - t v$, $\hat\epsilon = x_t + (1 - t)v$.

- **A:** Both define Gaussian paths between data and noise.
- **C:** It's trained with plain regression.
- **D:** It's usually the other way round; flow matching is typically sampled with a deterministic ODE.

</details>

## 9. Multiple choice (medium)

Why did SD3 sample training times from a logit-normal distribution instead of uniformly?

- **A.** To make training deterministic
- **B.** To spend more training on middle times, where predicting the velocity is hardest, and less near $t = 0$ and $t = 1$ where it is easy
- **C.** Because uniform sampling is not differentiable
- **D.** To avoid needing noise at all

<details>
<summary>Answer</summary>

**B.** To spend more training on middle times, where predicting the velocity is hardest, and less near $t = 0$ and $t = 1$ where it is easy

Near the ends the target is nearly predictable from $x_t$ alone; the middle is where the model must decide what the image becomes.

- **A:** Any time distribution is random.
- **C:** Time is sampled, not differentiated through.
- **D:** The interpolant still needs noise.

</details>

## 10. Multiple choice (hard)

You load a checkpoint trained with noise at $t = 0$ and data at $t = 1$ into a sampler that assumes data at $t = 0$. What goes wrong?

- **A.** Nothing; conventions don't matter
- **B.** The sampler feeds the wrong times and steps against the velocity's sign, so samples are garbage; map $t \to 1 - t$ and negate $v$
- **C.** Samples are correct but slower
- **D.** Only the guidance scale needs to change

<details>
<summary>Answer</summary>

**B.** The sampler feeds the wrong times and steps against the velocity's sign, so samples are garbage; map $t \to 1 - t$ and negate $v$

The math is the same up to $t \to 1 - t$ and a sign flip on $v$, but mixing them breaks sampling.

- **A:** The direction of time is baked into the network's inputs and outputs.
- **C:** Wrong times and signs change the result, not just speed.
- **D:** Guidance can't fix reversed time.

</details>
