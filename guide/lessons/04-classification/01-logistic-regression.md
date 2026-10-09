---
title: Logistic regression
summary: Turn a linear score into a probability with the sigmoid, train it with cross-entropy, and extend it to many classes with softmax.
skill: classical-ml
minutes: 35
prerequisites: [linear-regression, probability-and-statistics]
related: [classification-metrics, neural-network-forward-pass, decoding]
---

# Logistic regression

> **Mental model.** Compute a linear score, exactly like linear regression, then squeeze it into a probability between 0 and 1 with the sigmoid function. Train by making the probability of the correct label as high as possible.

**You will learn to**
- Compute a logit, a sigmoid probability, and binary cross-entropy by hand.
- Interpret weights as changes in log-odds.
- Derive the gradient $(\hat{p} - y)x$ and take a training step.
- Extend to $K$ classes with softmax and categorical cross-entropy.
- Explain why the decision boundary is linear and when that is a problem.

**Why it matters.** Logistic regression is the default baseline for classification: fast, interpretable, and usually well calibrated. Its pieces (logits, sigmoid, softmax, cross-entropy) are exactly the output layer of every neural classifier and every language model.

## 1. Intuition

Linear regression outputs any real number, but a probability must lie in $[0, 1]$. Logistic regression keeps the linear score, called the **logit** $z = w \cdot x + b$, and passes it through the **sigmoid** $\sigma(z) = 1/(1 + e^{-z})$:

- very negative $z$ gives a probability near 0,
- $z = 0$ gives exactly 0.5,
- very positive $z$ gives a probability near 1.

The boundary where the model is undecided ($p = 0.5$) is where $z = 0$, a straight line (or flat plane) in feature space.

Training uses **cross-entropy**: the negative log of the probability the model gave the correct label. Predicting 0.9 for a true positive costs $-\ln 0.9 = 0.105$; predicting 0.01 costs $-\ln 0.01 = 4.6$. Confident mistakes are punished hard.

## 2. Visualization

![Left: probability contours of a logistic regression on synthetic two-class data with a straight black decision boundary at p = 0.5. Right: binary cross-entropy curves, −log p for a positive label and −log(1−p) for a negative label, rising steeply as the prediction becomes confidently wrong.](../../figures/logistic-regression.png)

*Synthetic data. Probability changes smoothly across the boundary; the model is most uncertain near the black line.*

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $x$ | features | $[p]$ |
| $w, b$ | weights and bias (parameters) | $[p]$, scalar |
| $z$ | logit (log-odds), $w \cdot x + b$ | scalar |
| $\hat{p}$ | predicted probability of class 1, $\sigma(z)$ | scalar |
| $y \in \{0, 1\}$ | true label | scalar |
| $z_k$, $p_k$ | logit and probability for class $k$ (multiclass) | $[K]$ |

### Model and loss

```math
\hat{p} = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w\cdot x + b)}}
\qquad
L = -\big[y \ln \hat{p} + (1 - y)\ln(1 - \hat{p})\big]
```

### Log-odds interpretation

Rearranging the sigmoid: $\ln\frac{\hat{p}}{1 - \hat{p}} = w \cdot x + b$. Increasing feature $j$ by one unit adds $w_j$ to the log-odds, which multiplies the odds by $e^{w_j}$.

### Gradient

Using $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ and the chain rule, the sigmoid's derivative cancels:

```math
\frac{\partial L}{\partial z} = \hat{p} - y, \qquad \frac{\partial L}{\partial w} = (\hat{p} - y)\,x, \qquad \frac{\partial L}{\partial b} = \hat{p} - y
```

Same form as linear regression: (prediction − target) × input.

### Multiclass: softmax and categorical cross-entropy

```math
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad L = -\ln p_{\text{true class}}
```

### Worked example 1: one prediction

One feature (hours studied) with $w = 1.5$, $b = -4$. A student studied $x = 3$ hours.

1. Logit: $z = 1.5 \times 3 - 4 = 0.5$.
2. Probability: $\hat{p} = 1/(1 + e^{-0.5}) = 1/(1 + 0.6065) = 0.6225$.
3. If they passed ($y = 1$): $L = -\ln 0.6225 = 0.474$. If they failed ($y = 0$): $L = -\ln(1 - 0.6225) = -\ln 0.3775 = 0.974$.
4. Gradient for $y = 1$: $\partial L/\partial w = (0.6225 - 1)(3) = -1.133$, so a gradient step increases $w$.
5. Odds: $0.6225/0.3775 = 1.65$. Each extra hour multiplies the odds by $e^{1.5} = 4.48$.

### Worked example 2: softmax

Logits $[2, 1, 0.1]$. Exponentials $[7.389, 2.718, 1.105]$, sum $11.212$. Probabilities $[0.659, 0.242, 0.099]$. If class 0 is correct, $L = -\ln 0.659 = 0.417$.

## 4. Implementation

**From scratch (NumPy):**

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

def fit_logistic(X, y, lr=0.1, steps=2000):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)                 # [n]
        w -= lr * X.T @ (p - y) / len(y)       # average gradient (p - y) x
        b -= lr * np.mean(p - y)
    return w, b
```

**With scikit-learn:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))  # C = 1/λ (L2)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]        # probabilities; choose a threshold separately
```

Runnable script (worked examples, figures, and metrics): [`code/04-classification/classification.py`](../../code/04-classification/classification.py).

## 5. Engineering

**Good use cases.** Baselines for any classification problem; sparse high-dimensional data such as TF-IDF text; credit and risk models that must be explained; any setting where calibrated probabilities matter.

**Regularization.** scikit-learn applies L2 by default with strength `C = 1/λ` (smaller C means more regularization). Without regularization, perfectly separable data drives weights to infinity.

**Preprocessing.** Scale features (for the optimizer and the penalty); one-hot categoricals; add interaction or polynomial features if the boundary should curve.

**Cost.** Training is fast (convex, solved with L-BFGS or similar); inference is one dot product.

**Thresholds are separate.** `predict` uses 0.5, which is rarely the right business threshold. Use `predict_proba` and choose the threshold from error costs (next lesson).

> [!WARNING]
> **Failure modes.** Nonlinear boundaries (XOR-like patterns) that a linear score can't separate; perfect separation without regularization; correlated features making weights unstable and hard to interpret; class imbalance making the 0.5 threshold useless.

### Common mistakes

- Using MSE instead of cross-entropy (the loss becomes non-convex in $w$ and gradients vanish).
- Computing `log(sigmoid(z))` directly for large negative $z$; use stable library losses.
- Reading coefficients without scaling features first.
- Treating predicted probabilities as calibrated without checking.

## 6. Knowledge check

<!-- quiz:logistic-regression -->
**[Take the logistic regression quiz](../../quizzes/logistic-regression.md)**
<!-- /quiz -->

**Practice exercise.** With $w = [2, -1]$, $b = 0.5$, and $x = [1, 3]$, compute $z$, $\hat{p}$, and the loss if $y = 0$.

<details>
<summary>Solution</summary>

$z = 2 - 3 + 0.5 = -0.5$. $\hat{p} = 1/(1 + e^{0.5}) = 1/2.6487 = 0.3775$. Loss for $y = 0$: $-\ln(1 - 0.3775) = -\ln 0.6225 = 0.474$.
</details>

**Implementation challenge.** Train `fit_logistic` on a synthetic 2D dataset, check your weights against scikit-learn with `penalty=None`, and plot the decision boundary $w_1x_1 + w_2x_2 + b = 0$.

## Summary

- Logit $z = w\cdot x + b$; probability $\hat{p} = \sigma(z)$; the boundary $z = 0$ is linear.
- Binary cross-entropy is $-\ln$ of the probability of the true label; confident mistakes cost a lot.
- The gradient is $(\hat{p} - y)x$, the same shape as linear regression.
- Each weight changes the log-odds; $e^{w_j}$ is the odds multiplier.
- Softmax plus categorical cross-entropy generalizes to many classes and is the output layer of neural classifiers and LLMs.

**Next:** [Classification metrics and thresholds](02-classification-metrics.md)

**Related:** [Linear regression](../03-regression/01-linear-regression.md) · [Forward pass](../10-neural-networks/01-neural-network-forward-pass.md) · [Model card: logistic regression](../../models/logistic-regression.md)

## Interview angle

<details>
<summary><strong>Why use cross-entropy rather than MSE to train logistic regression?</strong></summary>

Two reasons. With a sigmoid output, MSE is non-convex in the weights, so optimization can stall on flat regions. More importantly, its gradient vanishes exactly when the model is confidently wrong: the MSE gradient with respect to the logit is $2(\hat{p} - y)\,\hat{p}(1 - \hat{p})$, so if $y = 1$ but $\hat{p} = 0.001$, the factor $\hat{p}(1 - \hat{p}) \approx 0.001$ kills the signal. Cross-entropy's gradient with respect to the logit is simply $\hat{p} - y$ because the $\sigma'$ term cancels, so a confident mistake gets a gradient near $-1$, the strongest possible push. Cross-entropy is also the negative log-likelihood of a Bernoulli model, so minimizing it is maximum likelihood, it's convex for logistic regression, and it tends to produce calibrated probabilities. Squared error on probabilities (the Brier score) remains a fine evaluation metric.

</details>

<details>
<summary><strong>Interpret a logistic regression coefficient of 0.7 on a standardized feature.</strong></summary>

A one-standard-deviation increase in that feature, holding the others fixed, adds 0.7 to the log-odds, which multiplies the odds by $e^{0.7} \approx 2.01$: roughly doubles them. It does not mean the probability rises by 0.7 or by any fixed amount; the effect on probability depends on the starting point. From $\hat{p} = 0.5$ (odds 1), the odds become 2.01, so $\hat{p} \approx 0.67$. From $\hat{p} = 0.05$ (odds 0.053), the odds become 0.106, so $\hat{p} \approx 0.096$. Caveats an interviewer expects: "holding others fixed" is unrealistic when features are correlated, and correlation makes individual coefficients unstable; regularization shrinks coefficients toward zero, so they are biased estimates; and none of this is causal. Coefficients on unstandardized features aren't comparable in magnitude.

</details>

<details>
<summary><strong>While training an unregularized logistic regression, the weights keep growing and the training loss approaches zero. What's happening?</strong></summary>

The training data is linearly separable (or nearly so). If some $w$ classifies every example correctly, scaling it up by any factor still classifies them correctly and pushes every $\hat{p}$ closer to 0 or 1, which always lowers cross-entropy. So the unregularized optimum is at infinity: weights grow without bound, predicted probabilities become 0 and 1, and coefficients and their standard errors are meaningless. It's common with many features relative to examples (sparse text), or with a leaky feature that perfectly predicts the label, which is worth checking first. Fix: add regularization (scikit-learn's default is L2 with `C = 1`, where smaller `C` means stronger), which gives a finite, unique solution, and tune `C` by cross-validation. Then check calibration, since overconfident probabilities are the symptom downstream users will notice.

</details>

<details>
<summary><strong>Logits are [2, 1, 0.1] and the true class is 1. Compute the cross-entropy loss and its gradient with respect to the logits.</strong></summary>

Softmax: $e^2 = 7.389$, $e^1 = 2.718$, $e^{0.1} = 1.105$, sum $11.212$, so $p = [0.659, 0.242, 0.099]$. Loss $= -\ln p_1 = -\ln 0.242 = 1.417$. The gradient of softmax cross-entropy with respect to the logits is $p - \text{onehot}(y) = [0.659, -0.758, 0.099]$: push the confidently wrong class 0 down hardest, push the true class up, and nudge class 2 down slightly. It's the same form as the binary case, $\hat{p} - y$. Implementation follow-up: compute the loss directly from logits with log-sum-exp, $L = \log\sum_j e^{z_j} - z_y$, subtracting $\max_j z_j$ first so large logits don't overflow. PyTorch's `cross_entropy` does exactly this when given raw logits, which is why you should not apply softmax yourself before calling it.

</details>
