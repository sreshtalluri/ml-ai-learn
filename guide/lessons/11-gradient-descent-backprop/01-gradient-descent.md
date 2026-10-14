---
title: Gradient descent
summary: Minimize a loss by repeatedly stepping against the gradient, compute steps by hand, and understand learning rates, mini-batches, momentum, and Adam.
skill: deep-learning
minutes: 40
prerequisites: [calculus-for-ml, linear-regression]
related: [backpropagation, training-and-regularization, neural-network-forward-pass]
adapted_from: [phases/01-math-foundations/08-optimization, phases/03-deep-learning-core/06-optimizers]
---

# Gradient descent

> **Mental model.** You are on a foggy hillside and want to reach the valley floor. You can't see far, but you can feel which way the ground slopes under your feet. Take a step downhill, feel again, repeat. The slope is the gradient; the length of your stride is the learning rate.

**You will learn to**
- Write the update rule $\theta \leftarrow \theta - \eta \nabla_\theta L$ and explain each symbol.
- Compute gradient-descent steps by hand on a two-parameter loss.
- Predict when a learning rate converges, zig-zags, or diverges.
- Distinguish batch, stochastic, and mini-batch gradient descent.
- Explain what momentum and Adam change, and when they help.

**Why it matters.** Almost every model in this course that has no closed-form solution, from logistic regression to GPT-scale transformers, is trained by some variant of gradient descent. Most "my model won't train" problems come down to the learning rate, the scale of the inputs, or the shape of the loss surface.

## 1. Intuition

Training means finding parameter values that make the loss small. The loss is a function of the parameters: change a weight, and the loss changes. For one parameter you could plot the loss as a curve and read off the bottom. With millions of parameters you cannot plot anything, but you *can* compute the local slope in every direction at once: the **gradient**.

The gradient points uphill, in the direction where the loss grows fastest. So you move the other way, by a small amount, and recompute. That's the whole algorithm.

Three things decide how well it works:

- **The step size (learning rate).** Too small and progress is painfully slow. Too large and you jump over the valley, land higher on the other side, and can bounce out entirely.
- **The shape of the valley.** A round bowl is easy. A long narrow valley is hard: the steep walls want tiny steps, the gentle floor wants big ones, and one learning rate can't satisfy both.
- **How you estimate the slope.** Using all the data is accurate but slow. Using a small random batch is noisy but much cheaper per step.

## 2. Visualization

<!-- lab:gradient-descent -->
![Left: optimizer paths on the contours of a narrow elliptical bowl. Gradient descent at learning rate 0.1 drops straight to the valley floor and walks along it; at 0.19 it zig-zags across the valley; momentum overshoots and spirals in; Adam takes even, diagonal steps. Right: loss per step on a log scale, with a dashed red curve for learning rate 0.21 rising instead of falling.](../../figures/gradient-descent.png)

*Loss $f(\theta_1, \theta_2) = \tfrac12(\theta_1^2 + 10\,\theta_2^2)$, starting at $(-3, 1.6)$. The valley is 10 times steeper across ($\theta_2$) than along ($\theta_1$).*

*Interactive version: click to choose a starting point, change the learning rate, add mini-batch noise, and race gradient descent, momentum, and Adam on three surfaces. [Open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/gradient-descent/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Predict whether η = 0.19 converges on the narrow bowl. Step it and watch the zig-zag.
2. Press **Compare all three** and step 30 times. Which optimizer is closest to the minimum, and is that what you expected?

What to notice:
- At $\eta = 0.19$ the steep direction overshoots every step: $\theta_2$ flips sign and shrinks slowly, the classic zig-zag.
- At $\eta = 0.21$ each overshoot is *larger* than the last. The loss grows forever: divergence.
- Momentum builds speed along the valley floor but overshoots and oscillates; Adam takes steps of roughly equal size in each coordinate regardless of steepness.
- On this simple quadratic, plain gradient descent with a well-chosen learning rate is the fastest of the four. Fancier optimizers earn their keep on messier surfaces and when you can't tune the learning rate carefully.

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $\theta$ | all trainable parameters, stacked into one vector | $[P]$ |
| $L(\theta)$ | loss: one number measuring how wrong the model is | scalar |
| $\nabla_\theta L$ | gradient: the vector of partial derivatives $\partial L / \partial \theta_j$ | $[P]$ |
| $\eta$ | learning rate (a **hyperparameter**) | scalar |
| $t$ | step number | integer |
| $B$ | a mini-batch of training examples | set |
| $v, m, s$ | optimizer state (velocity, first and second moment estimates) | $[P]$ each |
| $\beta, \beta_2$ | decay rates for that state, typically 0.9 and 0.999 | scalars |

### The update rule

```math
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta L(\theta_t)
```

Why subtract? The first-order Taylor approximation says $L(\theta + \Delta) \approx L(\theta) + \nabla L \cdot \Delta$. Choosing $\Delta = -\eta \nabla L$ gives $L(\theta) - \eta \lVert \nabla L \rVert^2$, which is smaller than $L(\theta)$ for a small enough $\eta > 0$.

### Batch, stochastic, mini-batch

The training loss is an average over $n$ examples, $L = \frac{1}{n}\sum_i \ell_i$, so its gradient is the average of per-example gradients. In practice we estimate it from a random mini-batch $B$:

```math
\nabla_\theta L \approx \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \ell_i(\theta)
```

- **Batch** gradient descent: $|B| = n$. Exact gradient, one expensive step per pass over the data.
- **Stochastic** gradient descent (SGD): $|B| = 1$. Very noisy, very cheap.
- **Mini-batch**: $|B|$ between about 32 and a few thousand. The default, because GPUs process a batch in parallel and the noise even helps escape sharp, poorly generalizing minima. One pass over the dataset is an **epoch**.

### Momentum and Adam

**Momentum** keeps a running velocity, so consistent directions accelerate and alternating directions cancel:

```math
v_t = \beta v_{t-1} + \nabla L(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta\, v_t
```

**Adam** keeps a running mean of the gradient ($m$) and of its square ($s$), corrects both for starting at zero, and divides, so each parameter gets its own step size:

```math
m_t = \beta m_{t-1} + (1-\beta) g_t, \quad
s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2, \quad
\theta_{t+1} = \theta_t - \eta \frac{m_t / (1 - \beta^t)}{\sqrt{s_t / (1-\beta_2^t)} + \epsilon}
```

Here $g_t$ is the gradient at step $t$, squares are element-wise, and $\epsilon \approx 10^{-8}$ prevents division by zero.

### Worked example

Loss $f(\theta_1, \theta_2) = \tfrac12(\theta_1^2 + 10\,\theta_2^2)$. Partial derivatives: $\partial f/\partial\theta_1 = \theta_1$ and $\partial f/\partial\theta_2 = 10\,\theta_2$. Start at $\theta_0 = (-3, 1.6)$, where $f = \tfrac12(9 + 10 \times 2.56) = \tfrac12(34.6) = 17.3$.

**Gradient at the start:** $\nabla f = (-3,\; 10 \times 1.6) = (-3,\; 16)$.

**Learning rate 0.1:**

```math
\theta_1 = (-3,\; 1.6) - 0.1 \times (-3,\; 16) = (-3 + 0.3,\; 1.6 - 1.6) = (-2.7,\; 0)
```

The loss drops to $\tfrac12(7.29 + 0) = 3.645$. The steep coordinate hit zero in one step, because $1 - \eta \times 10 = 0$. From now on $\theta_1$ shrinks by a factor $1 - 0.1 = 0.9$ per step: $-2.43, -2.187, \dots$

**Learning rate 0.19:** the steep coordinate is multiplied each step by $1 - 0.19 \times 10 = -0.9$: $1.6 \to -1.44 \to 1.296 \to -1.166$. It flips sign every step but shrinks: zig-zag convergence. The loss goes $17.3 \to 13.32 \to 10.34 \to 8.07$.

**Learning rate 0.21:** the factor is $1 - 2.1 = -1.1$, so $1.6 \to -1.76 \to 1.936 \to -2.13$. The loss goes $17.3 \to 18.30 \to 20.49 \to 23.77$, up every step. **Diverged.**

**The general rule:** along a direction with curvature $\lambda$ (here 1 and 10), each step multiplies the error by $1 - \eta\lambda$. Convergence needs $|1 - \eta\lambda| < 1$, so $\eta < 2/\lambda_{\max} = 2/10 = 0.2$. The steepest direction sets the speed limit for every direction.

**Momentum ($\eta = 0.02$, $\beta = 0.9$), two steps:**
$v_1 = (-3, 16)$, so $\theta_1 = (-3, 1.6) - 0.02(-3, 16) = (-2.94, 1.28)$.
New gradient $(-2.94, 12.8)$, so $v_2 = 0.9(-3, 16) + (-2.94, 12.8) = (-5.64, 27.2)$ and $\theta_2 = (-2.94 + 0.1128,\; 1.28 - 0.544) = (-2.8272,\; 0.736)$.
The velocity in $\theta_1$ grew from 3 to 5.64 because the gradient kept pointing the same way.

**Adam ($\eta = 0.1$), first step:** $m_1 = 0.1\,g$ and $s_1 = 0.001\,g^2$; after bias correction $\hat m = g$ and $\hat s = g^2$, so the step is $\eta\, g/|g| = 0.1 \times \text{sign}(g)$: $\theta_1 = (-2.9, 1.5)$. Both coordinates move by exactly 0.1, even though one gradient is 5 times larger. That per-parameter normalization is Adam's main idea.

## 4. Implementation

The full script is [`code/11-gradient-descent-backprop/optimizers_scratch.py`](../../code/11-gradient-descent-backprop/optimizers_scratch.py) (NumPy only). Each optimizer is a small class with one method, `step(params, grads)`, which is the same interface PyTorch's optimizers expose. The gradient comes from outside; the optimizer only decides how to use it.

### Build it from scratch

**Step 1: plain gradient descent.** No memory between steps. This is the update rule from section 3 and nothing else.

```python
class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, p, g):
        return p - self.lr * g
```

*Predict:* on the bowl from the worked example, which `lr` makes the steep coordinate land exactly on 0 in one step?

**Step 2: momentum.** Keep a velocity `v` that adds each new gradient to a decayed copy of the old one. Where gradients keep pointing the same way, `v` grows; where they flip sign (a zig-zag), they cancel. This uses the course's convention $v_t = \beta v_{t-1} + g_t$.

```python
class Momentum:
    def __init__(self, lr, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, p, g):
        self.v = g if self.v is None else self.beta * self.v + g
        return p - self.lr * self.v
```

*Predict:* if the gradient were the same constant $g$ forever, what would `v` approach? (Sum the geometric series.)

**Step 3: Adam.** Two running averages: `m` of the gradient (momentum) and `s` of its square (its typical size). Dividing one by the square root of the other gives every parameter a step of about `lr`, whatever its gradient scale. Both averages start at 0, so early on they are too small; dividing by $1 - \beta^t$ corrects that.

```python
    def step(self, p, g):
        if self.m is None:
            self.m, self.s = np.zeros_like(p), np.zeros_like(p)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g          # average gradient
        self.s = self.b2 * self.s + (1 - self.b2) * g ** 2     # average squared gradient
        m_hat = self.m / (1 - self.b1 ** self.t)               # undo the bias toward 0
        s_hat = self.s / (1 - self.b2 ** self.t)
        return p - self.lr * (m_hat / (np.sqrt(s_hat) + self.eps) + self.wd * p)
```

*Predict:* without the two bias-correction lines, is the first step bigger or smaller than `lr`? The script prints the answer.

**Step 4: AdamW.** The last line above already has a `self.wd * p` term. It shrinks weights directly, outside Adam's per-parameter scaling. That is *decoupled* weight decay, which is what AdamW means. Adding an L2 penalty to the loss instead would push the decay through `g`, so Adam would rescale it, and parameters with large gradients would barely be regularized.

```python
def AdamW(lr, weight_decay=0.01, **kw):
    return Adam(lr, weight_decay=weight_decay, **kw)
```

*Predict:* with zero gradient, `lr = 0.1`, and `weight_decay = 0.1`, what is a weight of 1.0 after 10 steps?

**Step 5: the training loop.** Every optimizer plugs into the same loop: compute the gradient, take a step, record where you are.

```python
def run(opt, grad, start, steps):
    p, path = np.array(start, float), [np.array(start, float)]
    for _ in range(steps):
        p = opt.step(p, grad(p))
        path.append(p.copy())
    return np.array(path)
```

*Predict:* on the Rosenbrock valley (a long, curved, flat-bottomed valley), which of the three reaches the minimum at (1, 1) within 3,000 steps?

### Use the library

PyTorch splits the same two jobs: autograd fills `.grad` for every parameter, then the optimizer reads it and updates in place. `torch.optim.AdamW` is step 4. One convention differs: PyTorch's SGD momentum uses the same $v_t = \beta v_{t-1} + g_t$ as ours, while some textbooks use $(1-\beta) g_t$, which changes the effective learning rate by a factor of 10 at $\beta = 0.9$.

```python
import torch

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
# or torch.optim.SGD(..., momentum=0.9)
for x_batch, y_batch in loader:           # mini-batches
    optimizer.zero_grad()                 # clear old gradients
    loss = torch.nn.functional.mse_loss(model(x_batch), y_batch)
    loss.backward()                       # compute gradients (backpropagation)
    optimizer.step()                      # θ ← θ − η · (update rule)
```

### Run it

```bash
uv run code/11-gradient-descent-backprop/optimizers_scratch.py
```

Key lines of the output:

```text
        GD lr=0.1: step 1 [-2.7  0. ], step 2 [-2.43  0.  ], f after 2 = 2.9525
 momentum lr=0.02: step 1 [-2.94  1.28], step 2 [-2.8272  0.736 ], f after 2 = 6.7050
      Adam lr=0.1: step 1 [-2.9  1.5], step 2 [-2.8001  1.4002], f after 2 = 13.7234
lr=0.21: f after 50 steps = 1.76e+05
raw m/sqrt(s) = 3.1623; corrected = 1.0000
     SGD: end [-1.2487  1.567 ], loss 5.06e+00
momentum: end [0.7811 0.6092], loss 4.80e-02
    Adam: end [1. 1.], loss 3.15e-11
after 10 steps: 0.9044 (= 0.99^10 = 0.9044)
```

The first three lines are the worked example in section 3. On Rosenbrock, the learning rate that keeps plain SGD stable is so small that it barely moves along the valley floor; momentum gets close, and Adam's per-parameter scaling reaches the minimum. The figure script for the bowl paths is [`code/11-gradient-descent-backprop/gradient_descent.py`](../../code/11-gradient-descent-backprop/gradient_descent.py).

## 5. Engineering

**Picking a learning rate.** Try values on a log scale (1e-1, 3e-2, 1e-2, ...). Pick the largest one that trains smoothly, then reduce it during training with a schedule (step decay, cosine decay). Large models often use a short **warmup** where the learning rate ramps up from near zero. A learning-rate range test (increase it each batch, plot the loss) finds the cliff quickly.

**Read the loss curve.**
- Flat from the start: learning rate too small, a bug (no gradient flowing), or dead activations.
- Decreasing then exploding into NaN: learning rate too large, or exploding gradients. Lower $\eta$, add gradient clipping, check for unscaled inputs.
- Very noisy: batch too small or learning rate too high; average over more steps before judging.

**Conditioning matters.** Unscaled features create narrow valleys (curvature differs wildly by direction), which force a tiny learning rate. Standardizing inputs and using normalization layers makes the surface rounder. Adam partially compensates by giving each parameter its own step size.

**Which optimizer?** Adam (or AdamW, which handles weight decay correctly) is the default for transformers and most deep networks because it works with little tuning. SGD with momentum is still competitive for many vision models and sometimes generalizes better. For convex problems with few parameters (linear and logistic regression), second-order or quasi-Newton methods such as L-BFGS are often faster.

**Cost.** Each step costs one forward and one backward pass over the batch, roughly proportional to (batch size × parameters). Adam stores two extra numbers per parameter, so its state alone takes twice the memory of the weights, which matters at LLM scale.

> [!WARNING]
> **Failure modes.** Divergence (loss becomes NaN), getting stuck on plateaus or saddle points, oscillating around the minimum with a constant learning rate, and convergence to a poor local minimum on non-convex losses. In deep networks, vanishing or exploding gradients can make some layers learn far faster than others (covered in [backpropagation](02-backpropagation.md)).

### Common mistakes

- Judging a learning rate after a handful of noisy steps.
- Changing the batch size without adjusting the learning rate (larger batches usually allow a larger $\eta$).
- Forgetting `optimizer.zero_grad()`, so gradients accumulate across steps.
- Comparing optimizers each at its default learning rate instead of each at its own tuned one.
- Treating a lower training loss as success without checking validation loss.

## 6. Knowledge check

<!-- quiz:gradient-descent -->
**[Take the gradient descent quiz](../../quizzes/gradient-descent.md)**: compute updates, find the stability limit, and diagnose loss curves.
<!-- /quiz -->

**Practice exercise.** For $f(\theta) = 2\theta^2$, starting at $\theta = 1$: compute three gradient-descent steps with $\eta = 0.1$, then find the largest learning rate that still converges.

<details>
<summary>Solution</summary>

$f'(\theta) = 4\theta$, so each step multiplies $\theta$ by $1 - 0.4 = 0.6$: $1 \to 0.6 \to 0.36 \to 0.216$.
Curvature is $f''(\theta) = 4$, so convergence requires $|1 - 4\eta| < 1$, that is $\eta < 0.5$. At $\eta = 0.5$ the factor is $-1$ and $\theta$ oscillates between 1 and −1 forever.
</details>

**Implementation challenge.** Implement mini-batch SGD for the linear regression from [Module 3](../03-regression/01-linear-regression.md) on a synthetic dataset of 10,000 points. Plot the training loss per step for batch sizes 1, 32, and 10,000 with the same number of *examples processed*, and explain the differences.

## Summary

- Gradient descent repeats $\theta \leftarrow \theta - \eta \nabla L$: compute the slope, step downhill.
- Along a direction of curvature $\lambda$, the error is multiplied by $1 - \eta\lambda$ each step, so $\eta < 2/\lambda_{\max}$ for convergence; steep directions set the limit.
- Mini-batches trade gradient accuracy for many cheaper steps and are the standard.
- Momentum accumulates velocity along consistent directions; Adam scales each parameter's step by its gradient history.
- Read the loss curve: flat, exploding, and noisy curves each point to different fixes.

**Next:** [Backpropagation](02-backpropagation.md)

**Related:** [Linear regression](../03-regression/01-linear-regression.md) · [Training and regularization](../12-training-regularization/01-training-and-regularization.md) · [Forward pass](../10-neural-networks/01-neural-network-forward-pass.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Optimization](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/08-optimization), [Optimizers](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/06-optimizers).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Why does a learning rate that is only slightly too large make the loss explode instead of just converging slowly?</strong></summary>

Because the error along each direction is multiplied by the same factor every step, so the behavior is geometric. Near a minimum, along a direction with curvature $\lambda$, one update multiplies the distance to the minimum by $1 - \eta\lambda$. If $|1 - \eta\lambda| < 1$ the error shrinks; if it exceeds 1 it grows every step. That gives the stability limit $\eta < 2/\lambda_{\max}$, set by the steepest direction. In the course's bowl $\tfrac12(\theta_1^2 + 10\theta_2^2)$, $\lambda_{\max} = 10$, so the limit is 0.2. At $\eta = 0.19$ the factor on $\theta_2$ is $-0.9$: it flips sign and slowly shrinks, a zig-zag. At $\eta = 0.21$ the factor is $-1.1$, so the loss goes $17.3 \to 18.3 \to 20.5 \to 23.8$ and diverges. A 10% change in $\eta$ is the difference between converging and NaN, which is why you tune it on a log scale.

</details>

<details>
<summary><strong>When would you choose SGD with momentum over Adam, or the other way around?</strong></summary>

Use AdamW as the default for transformers and most new architectures; consider SGD with momentum for well-understood vision models where you can tune the schedule. Adam divides each parameter's step by a running RMS of its gradient, so every parameter moves at a similar scale regardless of gradient magnitude. That makes it robust to badly conditioned problems and to layers with very different gradient sizes, with little tuning. SGD with momentum uses one global step size; it needs more careful learning-rate and schedule tuning but sometimes generalizes slightly better on image classification. Memory is a real trade-off at scale: Adam keeps two extra numbers per parameter. For a 7B model with fp32 optimizer states, that is $7 \times 10^9 \times 2 \times 4$ bytes $= 56$ GB on top of weights and gradients. Always use AdamW rather than Adam plus an L2 term if you want true weight decay.

</details>

<details>
<summary><strong>Training loss falls nicely for a few hundred steps, then suddenly becomes NaN. How do you debug it?</strong></summary>

Treat it as either an exploding update or a bad number, and find which. First log the global gradient norm and the loss per step: a norm that climbs or spikes right before the NaN means the step was too large. Fix with a lower peak learning rate, a warmup period, and gradient clipping (for example, max norm 1.0). If the norm was fine, look for numerical problems: computing `log(sigmoid(z))` by hand instead of a fused loss, division by a tiny variance in a custom normalization, fp16 overflow without loss scaling (switch to bf16 or enable a gradient scaler), or a batch containing inf or NaN features. Rerun with a fixed seed, save the batch index where it fails, and check that batch's inputs and labels. `torch.autograd.set_detect_anomaly(True)` will point at the first operation producing NaN, at a speed cost.

</details>

<details>
<summary><strong>You increase the batch size from 256 to 2,048. What should you do with the learning rate, and why?</strong></summary>

Usually raise it, roughly in proportion for SGD, with warmup, and then verify. A mini-batch gradient is an average of $|B|$ per-example gradients, so its noise variance falls as $1/|B|$. With an 8x larger batch you get a cleaner gradient but 8x fewer steps per epoch. A common heuristic for SGD is linear scaling: multiply $\eta$ by 8 so each epoch makes about the same total progress, and add a warmup because the large step is unstable early. For Adam, gains are often closer to square-root scaling ($\eta \times \sqrt{8} \approx 2.8$). Both heuristics break down past a "critical batch size" where the gradient is already accurate and bigger batches stop reducing the number of steps needed. Keep the stability limit in mind too: a larger $\eta$ still has to satisfy $\eta < 2/\lambda_{\max}$, so run a short learning-rate sweep at the new batch size rather than trusting the rule blindly.

</details>
