---
title: Loss functions
summary: Pick the right loss for regression, classification, imbalanced data, and embeddings, compute each one and its gradient by hand, and explain why cross-entropy beats squared error for classifiers.
skill: deep-learning
minutes: 45
prerequisites: [neural-network-forward-pass, information-theory]
related: [numerical-stability, logistic-regression, backpropagation, intro-to-pytorch]
adapted_from: [phases/03-deep-learning-core/05-loss-functions]
---

# Loss functions

> **Mental model.** The loss is the one number the optimizer is allowed to look at. It is a price list for mistakes: how much each kind of wrong answer costs. The model will find the cheapest way to lower the bill, so the price list had better match what you actually care about.

**You will learn to**
- Compute MSE, MAE, Huber, binary cross-entropy, and softmax cross-entropy, with their gradients.
- Explain why cross-entropy trains classifiers better than squared error, using the gradient at the logit.
- Choose a loss for regression with outliers, multi-class, multi-label, imbalanced, and embedding tasks.
- Apply label smoothing and focal loss, and say what each changes.
- Separate the loss (what you optimize) from the metric (what you report).

**Why it matters.** Changing the architecture rarely flips a project from failure to success; a wrong loss often does. The loss also decides what "confident" means: squared error and cross-entropy agree on which prediction is best but disagree wildly on how hard to push a wrong one. Loss choice is a common interview topic because it shows you understand what training actually optimizes.

## 1. Intuition

Training is a loop: predict, measure the mistake, nudge the weights to make the mistake smaller. The **loss function** is the measuring step. It takes the model's prediction and the true answer and returns one number. The optimizer only ever sees that number and its gradient.

**Regression** predicts a number. The natural price is the distance from the truth. Squaring it (**MSE**) makes big errors very expensive: off by 10 costs 100 times as much as off by 1, so one outlier can dominate. Taking the absolute value (**MAE**) charges linearly and shrugs off outliers. **Huber** does both: squared for small errors, linear for big ones.

**Classification** predicts probabilities. The natural price is surprise, $-\ln(\text{probability given to the true class})$, which is **cross-entropy** from [information theory](../00-foundations/05-information-theory.md). Giving the right answer 0.9 costs 0.105. Giving it 0.01 costs 4.6. The price explodes as a confident prediction turns out wrong.

Why not just use squared error on the probability? Because of the sigmoid or softmax in front of it. When the model is confidently wrong, the sigmoid is flat, and squared error multiplies by that flat slope, so the gradient almost vanishes exactly when the model most needs correcting. Cross-entropy's log cancels the flat slope, and the gradient becomes simply "predicted minus true".

A caveat worth getting right: on a balanced dataset, a model that knows nothing does best by predicting 0.5 for everything, under *either* loss (MSE 0.25, cross-entropy 0.693). Squared error doesn't make the model lazier than cross-entropy at the starting line. The difference is how strongly each loss pushes a model that has started to commit, and especially one that has committed to the wrong answer.

Finally, **the loss is not the metric**. You might report accuracy, F1, or revenue, but those are flat or jumpy and have no useful gradient. The loss is a smooth stand-in you can differentiate. Keep both on your dashboard.

## 2. Visualization

![Left: for a true label of 1, cross-entropy minus ln p climbs steeply toward infinity as p goes to 0, while squared error (1 − p)² never exceeds 1. Middle: the size of the gradient on the logit for label 1; cross-entropy's gradient 1 − p approaches 1 for confidently wrong logits below −3, while MSE's gradient peaks near 0.3 and falls to almost 0 in the same shaded region. Right: training accuracy of the same small network from the same initialization on synthetic circle data; cross-entropy reaches 0.84 at epoch 10 and MSE 0.73, both end near 0.98.](../../figures/loss-functions.png)

*Synthetic data: 400 points uniform in a square, labelled 1 inside a circle. The network starts deliberately overconfident (large output weights), so many points begin confidently wrong. The middle panel is the reason for the right panel.*

**Try it** (predict, then run the script):
1. In the middle panel, read off the push at $z = -5$ for both losses. The script prints them; how many times bigger is cross-entropy's?
2. In the left panel, at what $p$ does cross-entropy first exceed 1? (Solve $-\ln p = 1$.)
3. Remove the line `W2[:] *= 2.0` in `train` so the network starts less confident. Predict whether the gap between the curves shrinks or grows.
4. In the regression example, change the outlier's prediction from 16 to 106. Predict how much MSE, MAE, and Huber each grow.

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $y$ | true target (number, 0/1 label, or class index) | scalar per example |
| $\hat{y}$ | regression prediction | scalar per example |
| $z$ | logit (pre-activation output) | scalar, or $[K]$ for $K$ classes |
| $p = \sigma(z)$ | predicted probability of class 1 | scalar |
| $q = \text{softmax}(z)$ | predicted class distribution | $[K]$ |
| $n$ | batch size | integer |
| $\delta$ | Huber threshold | scalar |
| $\varepsilon$ | label-smoothing amount | scalar in $[0, 1)$ |
| $\gamma$ | focal-loss focusing exponent | scalar $\ge 0$ |
| $\tau$ | temperature (contrastive loss) | scalar $> 0$ |

### The losses

Regression, averaged over $n$ examples with residual $r = \hat{y} - y$:

```math
\text{MSE} = \frac{1}{n}\sum r^2, \qquad
\text{MAE} = \frac{1}{n}\sum |r|, \qquad
\text{Huber}_\delta = \frac{1}{n}\sum \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta(|r| - \tfrac12\delta) & |r| > \delta \end{cases}
```

MSE is the right loss when noise is Gaussian (it is the negative log-likelihood); its minimizer is the conditional **mean**. MAE's minimizer is the **median**, which is why it ignores outliers.

Binary cross-entropy (BCE) and softmax cross-entropy:

```math
\text{BCE} = -\big[y\ln p + (1 - y)\ln(1 - p)\big], \qquad
\text{CE} = -\ln q_y = -z_y + \ln\sum_j e^{z_j}
```

### The gradient that explains everything

With $p = \sigma(z)$ and $\sigma'(z) = p(1 - p)$, the chain rule gives:

```math
\frac{\partial\,\text{BCE}}{\partial z} = \frac{\partial\,\text{BCE}}{\partial p}\cdot p(1-p) = \Big(\frac{p - y}{p(1-p)}\Big)\, p(1-p) = p - y
```

```math
\frac{\partial\,(p - y)^2}{\partial z} = 2(p - y)\cdot p(1 - p)
```

The $p(1-p)$ cancels for BCE and stays for MSE. When the model is confidently wrong, $p(1-p) \approx 0$, so MSE's push nearly vanishes. Softmax cross-entropy has the same clean form: $\partial\,\text{CE}/\partial z = q - \text{onehot}(y)$.

### Worked example 1: confidently wrong, label 1, logit $z = -5$

$p = \sigma(-5) = 1/(1 + e^{5}) = 1/149.41 = 0.0067$.

- BCE gradient: $p - y = 0.0067 - 1 = -0.9933$.
- MSE gradient: $2(p - y)\,p(1-p) = 2 \times (-0.9933) \times 0.0067 \times 0.9933 = -0.0132$.

Cross-entropy pushes $0.9933 / 0.0132 \approx 75$ times harder. At $z = 0$ ($p = 0.5$) the ratio is only 2 ($-0.5$ vs $-0.25$).

### Worked example 2: softmax cross-entropy and its gradient

Logits $[2.0, 1.0, 0.1]$, true class 0. Softmax (from the [information theory](../00-foundations/05-information-theory.md) lesson) is $[0.6590, 0.2424, 0.0986]$. Loss $= -\ln 0.6590 = 0.4170$.

Gradient $= q - \text{onehot} = [0.6590 - 1,\ 0.2424,\ 0.0986] = [-0.3410,\ 0.2424,\ 0.0986]$. The true class's logit goes up; the others go down, each in proportion to how much probability it stole. The components sum to 0.

With **label smoothing** $\varepsilon = 0.1$, the target becomes $[0.9 + 0.1/3,\ 0.1/3,\ 0.1/3] = [0.9333, 0.0333, 0.0333]$. Log-probabilities are $[-0.4170, -1.4170, -2.3170]$, so

```math
\text{loss} = 0.9333 \times 0.4170 + 0.0333 \times 1.4170 + 0.0333 \times 2.3170 = 0.3892 + 0.0472 + 0.0772 = 0.5137
```

The loss can now never reach 0, so the model has no reason to drive logits to infinity.

### Worked example 3: one outlier in regression

Targets $[3, 5, 4, 6]$, predictions $[2.5, 5.5, 4, 16]$, residuals $[-0.5, 0.5, 0, 10]$.

- MSE: $(0.25 + 0.25 + 0 + 100)/4 = 100.5/4 = 25.125$. The outlier is 99.5% of it.
- MAE: $(0.5 + 0.5 + 0 + 10)/4 = 11/4 = 2.75$.
- Huber, $\delta = 1$: small residuals give $\tfrac12(0.25) = 0.125$ each; the outlier gives $1 \times (10 - 0.5) = 9.5$. Total $(0.125 + 0.125 + 0 + 9.5)/4 = 9.75/4 = 2.4375$.

### Worked example 4: focal loss

$\text{FL} = -(1 - p_t)^\gamma \ln p_t$, with $p_t$ the probability of the true class. With $\gamma = 2$:

| $p_t$ | CE | weight $(1-p_t)^2$ | focal |
|---|---|---|---|
| 0.9 (easy) | 0.1054 | 0.01 | 0.0011 |
| 0.5 | 0.6931 | 0.25 | 0.1733 |
| 0.1 (hard) | 2.3026 | 0.81 | 1.8651 |

Easy examples are discounted 100 times; hard ones keep most of their loss. In detection, where most candidate boxes are easy background, this keeps the easy majority from drowning out the rare hard cases.

### Contrastive loss (InfoNCE)

For embedding models with no class labels: an anchor $a$, one positive $b^+$ (an augmented view, a matching caption), and negatives $b_1^-, \dots$:

```math
\mathcal{L} = -\ln\frac{e^{\cos(a, b^+)/\tau}}{e^{\cos(a, b^+)/\tau} + \sum_k e^{\cos(a, b_k^-)/\tau}}
```

It is softmax cross-entropy where the "correct class" is the positive. Lower $\tau$ sharpens the softmax. With 7 negatives, a model that can't tell them apart scores $\ln 8 = 2.079$; the script's well-separated synthetic example scores 0.125 at $\tau = 0.1$. **Triplet loss**, $\max(0, d(a, b^+) - d(a, b^-) + m)$, is the older margin-based cousin.

## 4. Implementation

### Build it from scratch

Regression losses are one line each. Huber uses `np.where` to switch from quadratic to linear at $\delta$.

```python
def mse(pred, y):
    return np.mean((pred - y) ** 2)

def mae(pred, y):
    return np.mean(np.abs(pred - y))

def huber(pred, y, delta=1.0):
    r = np.abs(pred - y)
    return np.mean(np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta)))   # quadratic near 0, linear far out
```

*Predict:* at $|r| = \delta$, do the two Huber pieces give the same value? (Both give $\tfrac12\delta^2$, so the loss is continuous, and so is its slope.)

BCE should be computed from the logit, not from $p$, so that `log` never sees 0 (see [numerical stability](../00-foundations/06-numerical-stability.md)). The gradient function is where the two classification losses differ by exactly one factor.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def bce_with_logits(z, y):
    # -[y log s(z) + (1-y) log(1-s(z))], rewritten so exp only sees non-positive numbers
    return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))

def grad_wrt_logit(z, y, loss):
    """dLoss/dz for one example with a sigmoid output p = s(z)."""
    p = sigmoid(z)
    if loss == "bce":
        return p - y                              # the sigmoid's slope cancels out
    return 2 * (p - y) * p * (1 - p)              # MSE on p: chain rule leaves p(1-p) in
```

*Predict:* for label 0 and $z = +5$ (confidently wrong the other way), what are the two gradients?

Softmax cross-entropy works in log space, supports label smoothing by building a soft target, and has the $q - \text{onehot}$ gradient. The script checks that gradient with finite differences.

```python
def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def cross_entropy(logits, target, smoothing=0.0):
    """Mean CE over a batch. logits [n, K], target [n] ints. smoothing spreads eps over all K classes."""
    n, k = logits.shape
    q = np.full((n, k), smoothing / k)
    q[np.arange(n), target] += 1 - smoothing
    return -(q * log_softmax(logits)).sum(axis=1).mean()

def ce_grad(logits, target):
    """dCE/dlogits for one example = softmax(z) - onehot(y)."""
    g = np.exp(log_softmax(logits))
    g[target] -= 1
    return g
```

*Predict:* why do the components of `ce_grad` always sum to 0? (Softmax sums to 1 and you subtract exactly 1.)

Focal loss reweights cross-entropy; InfoNCE is cross-entropy over similarity scores, with the positive at index 0.

```python
def focal_loss(p_true, gamma=2.0, alpha=1.0):
    return -alpha * (1 - p_true) ** gamma * np.log(p_true)       # (1-p)^gamma down-weights easy examples

def info_nce(anchor, positive, negatives, tau=0.1):
    """Cross-entropy where the 'correct class' is the positive among [positive, *negatives]."""
    cands = np.vstack([positive, negatives])
    sims = cands @ anchor / (np.linalg.norm(cands, axis=1) * np.linalg.norm(anchor))   # cosine similarity
    return -log_softmax(sims / tau)[0]
```

*Predict:* what does focal loss reduce to when $\gamma = 0$?

The training comparison uses one small ReLU network, trained full-batch. The only line that changes between runs is the gradient at the logit.

```python
def train(X, y, loss, epochs=300, lr=0.5, hidden=16, seed=1):
    rng = np.random.default_rng(seed)
    W1, b1 = rng.normal(0, 1, (2, hidden)), np.zeros(hidden)
    W2, b2 = rng.normal(0, 1, hidden), 0.0
    W2[:] *= 2.0                                                   # a deliberately overconfident start
    acc = []
    for _ in range(epochs):
        H = np.maximum(X @ W1 + b1, 0)                             # [n, hidden]
        z = H @ W2 + b2                                            # [n]
        dz = grad_wrt_logit(z, y, loss) / len(y)                   # the only line that differs
        dH = np.outer(dz, W2) * (H > 0)
        W2 -= lr * H.T @ dz
        b2 -= lr * dz.sum()
        W1 -= lr * X.T @ dH
        b1 -= lr * dH.sum(axis=0)
        acc.append(((z > 0) == (y > 0.5)).mean())
    return acc
```

*Predict:* if you raised MSE's learning rate to make up for its smaller gradients, would that fix the confidently-wrong examples? (No: it scales every gradient equally, including the ones that were already fine, and can destabilize them.)

### Use the library

```python
import torch.nn.functional as F
F.mse_loss(pred, y);  F.l1_loss(pred, y);  F.huber_loss(pred, y, delta=1.0)
F.binary_cross_entropy_with_logits(z, y, pos_weight=torch.tensor([9.0]))   # takes LOGITS; pos_weight for imbalance
F.cross_entropy(logits, target, label_smoothing=0.1, weight=class_w)       # logits [n, K], target [n] int64
torchvision.ops.sigmoid_focal_loss(z, y, alpha=0.25, gamma=2.0)            # focal loss for detection
```

`F.cross_entropy` fuses log-softmax and NLL, which the scratch version made explicit. Multi-label problems (several classes can be true at once) use `binary_cross_entropy_with_logits` with one logit per class, not softmax. For contrastive learning, InfoNCE is usually written as `F.cross_entropy(sim / tau, torch.arange(n))` over a batch similarity matrix, where the diagonal holds the positives.

### Run it

```bash
uv run code/10-neural-networks/loss_functions.py
```

Key output:

```
MSE = 25.1250   MAE = 2.7500   Huber(delta=1) = 2.4375
z=-5  p=0.0067  dBCE/dz=-0.9933  dMSE/dz=-0.0132
grad = softmax - onehot = [-0.341   0.2424  0.0986]
finite-difference check max error = 7.7e-11
label smoothing eps=0.1: CE = 0.5137
p_true=0.9: CE=0.1054  focal=0.0011  weight (1-p)^2=0.01
tau=0.1: loss=0.1255
mse: accuracy at epoch 10=0.728, 50=0.945, 300=0.975
bce: accuracy at epoch 10=0.840, 50=0.965, 300=0.983
```

## 5. Engineering

**Choosing a loss.**

| Task | Output layer | Loss |
|---|---|---|
| Regression, Gaussian-ish noise | linear | MSE |
| Regression with outliers | linear | Huber (or MAE) |
| Binary classification | 1 logit | BCE with logits |
| Multi-class, one label per example | $K$ logits | softmax cross-entropy |
| Multi-label (tags) | $K$ logits | BCE with logits per class |
| Heavy class imbalance | as above | class weights or `pos_weight`; focal loss for detection |
| Overconfident classifier | $K$ logits | cross-entropy + label smoothing (around 0.1) |
| Embeddings / retrieval | normalized vectors | InfoNCE with in-batch negatives |
| Next-token prediction (LLMs) | vocab logits | softmax cross-entropy per token |

**Loss vs metric.** Optimize a differentiable loss; select models and thresholds on the metric. A model's validation loss can rise (it got more confident on its mistakes) while accuracy still improves. Track both.

**Imbalance.** With 1% positives, an unweighted loss is dominated by negatives. Reweighting (`pos_weight` = negatives/positives) or resampling changes the optimal predicted probabilities, so recalibrate before reading them as probabilities. Often it's better to keep the plain loss and move the decision threshold.

**Contrastive training at scale.** InfoNCE quality depends on the number and hardness of negatives, so batch size and temperature matter. Collapse (all embeddings identical) shows up as falling embedding variance; watch it.

**Reductions and masking.** `reduction="mean"` over a padded sequence batch averages over padding unless you mask it (`ignore_index=-100` in `F.cross_entropy`). Mean versus sum also changes the effective learning rate.

> [!WARNING]
> **Failure modes.** Applying softmax before `F.cross_entropy` (double softmax: it trains, but gradients are squashed); using softmax for multi-label data; MSE on sigmoid outputs for classification; averaging the loss over padding tokens; reporting loss to stakeholders instead of a metric they understand.

### Common mistakes

- Treating "lower loss" as "better model" across different losses. A Huber loss of 2.4 and an MSE of 25 are not comparable.
- Thinking label smoothing is free. It improves calibration of the argmax class but blurs the remaining probabilities, which can hurt distillation from that model.
- Forgetting that MSE predicts the mean. If you want the median or a quantile (delivery-time promises), use MAE or a quantile loss.

## 6. Knowledge check

<!-- quiz:loss-functions -->
**[Take the loss functions quiz](../../quizzes/loss-functions.md)**
<!-- /quiz -->

**Practice exercise.** A binary classifier outputs logit $z = 3$ for an example whose label is 0. Compute the BCE loss and the gradient at the logit for BCE and for MSE on $p$.

<details>
<summary>Solution</summary>

$p = \sigma(3) = 1/(1 + e^{-3}) = 1/1.0498 = 0.9526$.

BCE $= -\ln(1 - p) = -\ln 0.0474 = 3.049$.

BCE gradient: $p - y = 0.9526$. MSE gradient: $2(p - y)\,p(1-p) = 2 \times 0.9526 \times 0.9526 \times 0.0474 = 0.0860$. Cross-entropy pushes about 11 times harder.

</details>

**Implementation challenge.** Add a quantile (pinball) loss, $\max(\tau r, (\tau - 1) r)$ with $r = y - \hat{y}$, to the script. Fit a single constant to the synthetic targets `[1, 2, 3, 4, 100]` by gradient descent for $\tau = 0.5$ and $\tau = 0.9$, and compare with MSE's constant.

<details>
<summary>Solution</summary>

```python
def pinball(pred, y, tau):
    r = y - pred
    return np.mean(np.maximum(tau * r, (tau - 1) * r))

y = np.array([1, 2, 3, 4, 100.0])
for tau in (0.5, 0.9):
    c = 0.0
    for _ in range(50000):
        g = -np.mean(np.where(y > c, tau, tau - 1))   # subgradient w.r.t. c
        c -= 0.05 * g
    print(tau, round(c, 2))
```

$\tau = 0.5$ settles at 3 (the median). $\tau = 0.9$ climbs all the way to 100: with only five points, the 90th percentile is the largest one, since any lower value leaves 20% of the points above it. MSE's constant is the mean, 22. Pinball loss is how you train a model to predict "90% of deliveries arrive by" instead of an average.

</details>

## Summary

- The loss is what the optimizer sees; the metric is what you report. They differ.
- MSE targets the mean and punishes outliers; MAE targets the median; Huber blends them.
- Cross-entropy with a sigmoid or softmax has gradient $p - y$; MSE on probabilities keeps an extra $p(1-p)$ that vanishes when the model is confidently wrong.
- Label smoothing caps confidence; focal loss down-weights easy examples; class weights handle imbalance.
- InfoNCE is cross-entropy over similarities, the workhorse of embedding training.

**Next:** [Intro to PyTorch](03-intro-to-pytorch.md)

**Related:** [Information theory](../00-foundations/05-information-theory.md) · [Numerical stability](../00-foundations/06-numerical-stability.md) · [Logistic regression](../04-classification/01-logistic-regression.md) · [Backpropagation](../11-gradient-descent-backprop/02-backpropagation.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Loss Functions](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/05-loss-functions).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Why use cross-entropy instead of MSE for classification?</strong></summary>

Look at the gradient with respect to the logit. With a sigmoid output p, cross-entropy's gradient is p − y: the sigmoid's slope p(1 − p) cancels against the derivative of the log. MSE on p gives 2(p − y)·p(1 − p), so that slope stays in. When the model is confidently wrong, say logit −5 for a positive example, p is 0.0067 and p(1 − p) is tiny: cross-entropy pushes with −0.99, MSE with −0.013, about 75 times weaker. So MSE learns slowly exactly on the examples that matter most. There are two more reasons. Cross-entropy is the negative log-likelihood of a Bernoulli or categorical model, so it's the principled choice and gives calibrated probabilities. And as a function of the logits it's convex for a linear model, while MSE through a sigmoid is not. Both losses agree on the best constant prediction for uninformative data; the difference is in how hard they correct.

</details>

<details>
<summary><strong>MSE, MAE, or Huber for a regression target with occasional huge outliers?</strong></summary>

It depends on whether the outliers are errors or the thing you care about. MSE estimates the conditional mean and charges the square of each error, so one point off by 10 contributes 100 while points off by 0.5 contribute 0.25: in a four-point example the outlier is 99.5% of the loss and drags the fit toward itself. MAE estimates the median and charges linearly, so it's robust, but its gradient has constant size near zero, which can make convergence jittery. Huber is quadratic below a threshold delta and linear above, so it behaves like MSE on normal points and like MAE on outliers; it's my default when outliers are noise, with delta set around the typical residual scale. If the outliers are real and costly, like rare huge orders, I keep MSE or model them explicitly. If the business needs a guarantee, I use a quantile loss instead.

</details>

<details>
<summary><strong>Your multi-class classifier trains, but accuracy plateaus well below a baseline and predicted probabilities are all close to uniform. You find `loss = F.cross_entropy(F.softmax(logits, -1), y)`. What's wrong?</strong></summary>

That's a double softmax. F.cross_entropy expects raw logits and applies log-softmax itself, so here the probabilities from softmax are being treated as logits. They're all between 0 and 1, so the second softmax sees inputs that differ by at most 1, and its output can never be far from uniform: with 10 classes the true class can get at most about e/(e + 9), roughly 0.23. The loss can't go below about 1.46, and the gradients are squashed by the extra softmax's Jacobian, so training stalls. The fix is to pass logits directly. The same family of bugs: sigmoid before binary_cross_entropy_with_logits, or applying softmax for a multi-label problem where you need independent sigmoids. A quick check is the step-0 loss: it should be about ln K. Also check that the model's last layer has no activation.

</details>

<details>
<summary><strong>You're training a fraud classifier with 0.5% positives. How do you set up the loss?</strong></summary>

Start with plain BCE with logits and a stratified validation set, and measure precision-recall AUC and recall at the precision the business needs, not accuracy, since predicting "never fraud" scores 99.5%. If the model under-learns positives, add pos_weight around negatives/positives, here about 199, or a milder value like the square root, about 14, and tune it on validation. Weighting shifts the predicted probabilities upward, so recalibrate (Platt scaling or isotonic regression) before anyone reads them as probabilities. Focal loss is an option if there are many easy negatives that dominate the gradient; it's standard in object detection with gamma 2. Often the cleanest approach is no reweighting at all: train the calibrated model and move the decision threshold to hit the precision target. Whatever I pick, I'd compare the variants on the same validation metric and check calibration curves.

</details>
