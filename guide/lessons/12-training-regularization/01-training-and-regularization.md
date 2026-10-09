---
title: Training and regularization
summary: Read training curves, stop at the right time, regularize with weight decay, dropout, normalization, and augmentation, schedule the learning rate, and tune hyperparameters without fooling yourself.
skill: deep-learning
minutes: 40
prerequisites: [backpropagation, overfitting-and-bias-variance]
related: [gradient-descent, regularization-and-regression-metrics, convolutional-networks]
---

# Training and regularization

> **Mental model.** A large network can memorize its training set. Regularization makes memorizing harder than learning the general pattern, and validation loss tells you whether it is working. Training loss tells you how well the model fits; validation loss tells you how well it generalizes.

**You will learn to**
- Diagnose training from training and validation curves and choose an early-stopping point.
- Apply weight decay (L2), L1, dropout, batch and layer normalization, and data augmentation, and say what each does.
- Choose a learning-rate schedule, including warmup.
- Run grid search, random search, and Bayesian optimization, with cross-validation where appropriate.

**Why it matters.** Most of the gap between a model that "works on my machine" and one that generalizes is training discipline. These are also the knobs you will tune most often in practice.

## 1. Intuition

Plot loss after every epoch on the training and validation sets. Early on, both fall. Eventually training loss keeps falling while validation loss stops improving and starts rising: the model has begun fitting noise in the training set. The best model is the one at the validation minimum, which is what **early stopping** keeps.

Regularization techniques push in the same direction in different ways:

- **Weight decay (L2):** penalizes large weights, favoring smoother functions.
- **Dropout:** randomly zeroes a fraction of activations during training, so the network can't rely on any single unit and learns redundant, robust features. It is switched off at inference.
- **Normalization (batch norm, layer norm):** rescales activations to a stable range, which speeds training and has a mild regularizing effect.
- **Data augmentation:** creates label-preserving variations (flipped or cropped images, noise, paraphrases) so the model sees more diverse examples.

**The learning rate** changes over training: often a short warmup from near zero (stabilizes early updates, especially for transformers), then a decay so the model settles into a minimum instead of bouncing around it.

## 2. Visualization

![Three panels. Training and validation log loss over 200 epochs: without regularization, validation loss bottoms out at epoch 6 and then climbs to about 1.05; with L2 regularization it rises much less. Learning-rate schedules: constant, step decay, and linear warmup followed by cosine decay. Grid versus random search over two hyperparameters with 9 trials each, showing random search covering 9 distinct learning rates versus grid's 3.](../../figures/training-and-regularization.png)

*Synthetic classification task, MLP with two hidden layers of 256 units. Search panel uses a synthetic validation surface on which only the learning rate matters much.*

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $L_{\text{data}}$ | the training loss (e.g. cross-entropy) |
| $\lambda$ | regularization strength |
| $p$ | dropout probability (fraction of units zeroed) |
| $\mu_B, \sigma_B^2$ | mean and variance of a batch (batch norm) |
| $\gamma, \beta$ | learnable scale and shift in normalization layers |
| $\eta_t$ | learning rate at step $t$ |

### Weight decay and L1

```math
L = L_{\text{data}} + \lambda\sum_j w_j^2 \quad (\text{L2}) \qquad L = L_{\text{data}} + \lambda\sum_j |w_j| \quad (\text{L1})
```

With plain SGD, the L2 term adds $2\lambda w$ to the gradient, so each step shrinks weights by a constant factor ("decay"). With Adam this equivalence breaks, which is why **AdamW** applies decay directly to the weights.

### Dropout

During training, each activation is kept with probability $1 - p$ and scaled by $\frac{1}{1-p}$ so its expected value is unchanged:

```math
\tilde{a}_i = \frac{m_i}{1 - p}\,a_i, \qquad m_i \sim \text{Bernoulli}(1 - p)
```

At inference all units are kept and no scaling is applied.

### Normalization

```math
\hat{a} = \frac{a - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad \text{output} = \gamma\,\hat{a} + \beta
```

Batch norm computes $\mu, \sigma^2$ per feature across the batch (and uses running averages at inference). Layer norm computes them per example across features, which doesn't depend on batch size, so it is the standard in transformers.

### Cosine schedule with warmup

```math
\eta_t = \begin{cases} \eta_{\max}\,\dfrac{t}{T_w} & t < T_w \\[4pt] \dfrac{\eta_{\max}}{2}\left(1 + \cos\dfrac{\pi(t - T_w)}{T - T_w}\right) & t \ge T_w \end{cases}
```

### Worked example 1: dropout scaling

A layer's activations are $[2, 4, 6, 8]$ with $p = 0.5$. A sampled mask is $[1, 0, 1, 0]$. Kept units are scaled by $1/(1 - 0.5) = 2$: output $[4, 0, 12, 0]$. The mean of the original activations is 5; the expected value of each scaled output equals its original value, so on average the next layer sees the same scale at training and inference.

### Worked example 2: layer norm

Activations $a = [1, 2, 3, 6]$. Mean $\mu = 3$; variance $\sigma^2 = \frac{4 + 1 + 0 + 9}{4} = 3.5$; standard deviation $1.871$. Normalized: $[-1.069, -0.535, 0, 1.604]$. With $\gamma = 1$, $\beta = 0$ that is the output.

### Worked example 3: reading the curves

Without regularization, validation loss reached its minimum (0.364) at **epoch 6** and climbed to 1.047 by epoch 200, while training loss kept falling. Early stopping with patience would have kept the epoch-6 weights. With L2 strength 1.0, the best validation loss was similar (0.362), but the final value was 0.495: regularization made the model far less sensitive to training too long.

## 4. Implementation

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(40, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p=0.3),
    nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p=0.3),
    nn.Linear(256, 1),
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * len(train_loader))

best, patience, bad = float("inf"), 10, 0
for epoch in range(epochs):
    model.train()                                  # dropout ON
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(xb).squeeze(-1), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()                                   # dropout OFF
    with torch.no_grad():
        val = evaluate(model, val_loader)
    if val < best:
        best, bad = val, 0
        torch.save(model.state_dict(), "best.pt")  # keep the best checkpoint
    else:
        bad += 1
        if bad >= patience:
            break                                  # early stopping
```

Hyperparameter search with scikit-learn:

```python
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator, {"alpha": loguniform(1e-5, 1), "learning_rate_init": loguniform(1e-4, 1e-1)},
                            n_iter=30, cv=5, random_state=0)
```

Runnable script (training curves, schedules, search comparison): [`code/11-gradient-descent-backprop/backprop_training.py`](../../code/11-gradient-descent-backprop/backprop_training.py).

## 5. Engineering

**Hyperparameter search.** Grid search tries every combination on a fixed grid; random search samples combinations at random. When only a few hyperparameters really matter, random search explores more distinct values of each one for the same budget: in the figure's 9 trials, grid tried 3 learning rates and random tried 9. (On this particular run grid search still scored slightly higher, 0.894 versus 0.891, because its grid happened to land near the optimum. The advantage is in coverage, not a guarantee every time.) Bayesian optimization (Optuna, scikit-optimize) models the validation score and proposes promising settings, which is better for expensive training runs. Search learning rates and regularization strengths on a log scale.

**Cross-validation** is affordable for classical models and small networks; for large deep models, a single fixed validation split is the norm.

**What to tune first:** learning rate, then model size and regularization (weight decay, dropout), then batch size and schedule.

**Reproducibility:** seed everything, log configurations and metrics (an experiment tracker helps), and save the best checkpoint, not the last.

> [!WARNING]
> **Failure modes.** Selecting hyperparameters on the test set; forgetting `model.eval()` so dropout and batch norm behave like training at inference; batch norm with tiny batches; augmentation that changes the label (flipping a "6" into a "9"); stopping on a noisy validation curve without patience.

### Common mistakes

- Using L2 penalty with Adam and expecting weight decay (use AdamW).
- Evaluating a model while it is still in training mode.
- Reporting the last epoch's validation score instead of the best checkpoint's (or reporting the best checkpoint's validation score as if it were a test score).

## 6. Knowledge check

<!-- quiz:training-and-regularization -->
**[Take the training and regularization quiz](../../quizzes/training-and-regularization.md)**
<!-- /quiz -->

**Practice exercise.** Activations $[3, 1, 4, 2]$ pass through dropout with $p = 0.25$ and mask $[1, 1, 0, 1]$. What is the output? Then compute layer norm of $[3, 1, 4, 2]$ (with $\gamma = 1$, $\beta = 0$).

<details>
<summary>Solution</summary>

Scale is $1/0.75 = 1.333$: output $[4, 1.333, 0, 2.667]$.
Layer norm: mean 2.5, variance $\frac{0.25 + 2.25 + 2.25 + 0.25}{4} = 1.25$, std 1.118. Output $[0.447, -1.342, 1.342, -0.447]$.
</details>

**Implementation challenge.** Train the same MLP on a small dataset four ways (no regularization, weight decay, dropout, both), plot all training and validation curves on one figure, and report the best validation loss and the epoch where it occurred for each.

## Summary

- Watch training and validation loss together; keep the checkpoint at the validation minimum (early stopping).
- Weight decay, dropout, normalization, and augmentation each make memorization harder.
- Warmup plus decay schedules stabilize early training and help the model settle.
- Tune on validation data, prefer random or Bayesian search on log scales, and never tune on the test set.

**Next:** [Convolutional networks](../13-deep-architectures/01-convolutional-networks.md)

**Related:** [Gradient descent](../11-gradient-descent-backprop/01-gradient-descent.md) · [Regularization for linear models](../03-regression/02-regularization-and-regression-metrics.md)

## Interview angle

<details>
<summary><strong>How does dropout work, and why do you scale the kept activations by 1/(1−p)?</strong></summary>

During training, dropout zeroes each activation independently with probability $p$ and multiplies the survivors by $1/(1-p)$; at inference it does nothing. The scaling keeps the expected activation unchanged, so the next layer sees the same scale in training and inference: $E[\tilde{a}] = (1-p) \cdot a/(1-p) = a$. In the course example, $[2, 4, 6, 8]$ with $p = 0.5$ and mask $[1, 0, 1, 0]$ becomes $[4, 0, 12, 0]$. Because any unit can disappear, the network can't rely on a single co-adapted feature and learns redundant ones. One view is that it trains an ensemble of thinned subnetworks that share weights and averages them at test time. Practical points: forgetting `model.eval()` leaves dropout on at inference and gives noisy predictions, and large pretrained transformers often use little or no dropout because their data is large relative to their capacity.

</details>

<details>
<summary><strong>Batch norm versus layer norm: why do transformers use layer norm?</strong></summary>

Batch norm normalizes each feature using the mean and variance across the batch; layer norm normalizes each example across its own features. Batch norm's dependence on the batch causes problems for transformers: statistics are noisy with small per-device batches, sequences have variable lengths and padding, and inference must use running averages, which creates a train/inference mismatch and makes autoregressive decoding one token at a time awkward. Layer norm computes everything per example, so it behaves the same at any batch size and in training and inference. For $[1, 2, 3, 6]$: mean 3, variance 3.5, output $[-1.069, -0.535, 0, 1.604]$. Batch norm is still a strong choice for CNNs with large batches, where its batch noise also acts as a mild regularizer. Many modern LLMs use RMSNorm, which drops the mean subtraction and is slightly cheaper.

</details>

<details>
<summary><strong>Training loss keeps falling, but validation loss bottoms out at epoch 6 and climbs after that. What do you do?</strong></summary>

That is classic overfitting, but confirm it before treating it. Check that validation data comes from the same distribution as training and that no leakage or preprocessing difference explains the gap. Also check accuracy alongside loss: if validation loss rises while accuracy stays flat, the model is becoming overconfident on its mistakes, which matters if you use its probabilities. Then act: keep the epoch-6 checkpoint with early stopping and a patience of several epochs so noise doesn't stop you too soon; add weight decay (with AdamW), dropout, or data augmentation that preserves labels; reduce model size or get more data. In the course's synthetic run, validation loss hit its minimum of 0.364 at epoch 6 and climbed to about 1.05 by epoch 200, while L2 regularization flattened that rise. Report the selected checkpoint's score on a held-out test set, not its validation score.

</details>

<details>
<summary><strong>You have budget for 20 training runs to tune learning rate, weight decay, and dropout. How do you spend it?</strong></summary>

Random search on log scales, with learning rate prioritized, not a grid. A $3 \times 3 \times 2$ grid of 18 runs tries only three learning rates; 20 random samples try 20 distinct values of each hyperparameter. Since one or two hyperparameters usually dominate (often learning rate), random search explores the important axis far better. Sample learning rate log-uniformly over, say, $10^{-4}$ to $10^{-2}$, weight decay log-uniformly over $10^{-5}$ to $10^{-1}$, and dropout uniformly over 0 to 0.5. Use early stopping or successive halving to kill bad runs early, which stretches the budget further. With leftover runs, narrow the range around the best region or switch to Bayesian optimization. Select on validation or cross-validation, re-run the winner with two or three seeds to check it wasn't luck, and touch the test set once at the end.

</details>
