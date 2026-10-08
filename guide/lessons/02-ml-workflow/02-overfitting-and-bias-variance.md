---
title: Overfitting and the bias-variance trade-off
summary: Diagnose underfitting and overfitting from training and validation error, understand bias and variance, and pick model complexity with validation curves.
skill: classical-ml
minutes: 30
prerequisites: [ml-workflow, linear-regression]
related: [regularization-and-regression-metrics, training-and-regularization, k-nearest-neighbors]
---

# Overfitting and the bias-variance trade-off

> **Mental model.** A model that is too simple misses the pattern (underfitting). A model that is too flexible memorizes the noise (overfitting). The sweet spot is where error on new data, not training data, is lowest.

**You will learn to**
- Recognize underfitting and overfitting from training and validation error.
- Explain bias and variance, and how model complexity trades one for the other.
- Read validation curves and learning curves.
- Choose the remedies that match each diagnosis.

**Why it matters.** "Why can a model with lower training loss have worse test performance?" is a standard interview question for a reason. Every complexity knob in this course (polynomial degree, K in KNN, tree depth, number of boosting rounds, network size, training epochs) moves you along this trade-off.

## 1. Intuition

Fit points that follow a wavy curve plus noise:

- A **straight line** can't bend. It is wrong in the same systematic way on training and new data: high training error, high validation error. That is **underfitting**, or high **bias**.
- A **degree-15 polynomial** wiggles through nearly every training point, including the noise. It looks perfect on training data and is wild between and beyond the points: low training error, high validation error. That is **overfitting**, or high **variance**: retrain it on a different sample and you'd get a very different curve.
- A **degree-4 polynomial** follows the true shape and ignores the noise: both errors low and close together.

## 2. Visualization

![Four panels. Degree 1 underfits the sine-shaped data (train MSE 0.275, validation 0.27). Degree 4 follows it (0.075 and 0.09). Degree 15 wiggles wildly (0.056 and 0.52). A validation curve shows training error falling steadily with degree while validation error is lowest around degree 3 to 5 and rises after.](../../figures/overfitting-and-bias-variance.png)

*Synthetic data: 30 training points from $y = \sin(2\pi x)$ plus noise, 200 validation points. Training error keeps falling as complexity grows; validation error falls, bottoms out (degree 5 here), then rises.*

*Interactive: the KNN lab shows the same trade-off with K (try K = 1 versus K = 25). [Open the KNN lab](https://sreshtalluri.github.io/ml-ai-learn/labs/knn/).*

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $f(x)$ | the true function generating the data |
| $\varepsilon$ | noise with mean 0 and variance $\sigma^2$, so $y = f(x) + \varepsilon$ |
| $\hat{f}_{\mathcal{D}}(x)$ | the model fitted on a particular training set $\mathcal{D}$ |
| $\mathbb{E}_{\mathcal{D}}$ | average over many possible training sets |

### Bias-variance decomposition (squared error)

For a fixed input $x$, the expected squared error of the fitted model, averaged over training sets and noise, splits into three parts:

```math
\mathbb{E}\big[(y - \hat{f}_{\mathcal{D}}(x))^2\big] =
\underbrace{\big(f(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)]\big)^2}_{\text{bias}^2}
+ \underbrace{\mathbb{E}_{\mathcal{D}}\big[(\hat{f}_{\mathcal{D}}(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)])^2\big]}_{\text{variance}}
+ \underbrace{\sigma^2}_{\text{irreducible noise}}
```

- **Bias:** how far the *average* fitted model is from the truth. Simple models have high bias.
- **Variance:** how much the fitted model changes from one training set to another. Flexible models have high variance.
- **Noise:** no model can remove it.

Increasing complexity usually lowers bias and raises variance, so total error is U-shaped.

### Worked example: diagnosis table

| Training error | Validation error | Gap | Diagnosis |
|---|---|---|---|
| 0.275 | 0.267 | small, both high | underfitting (high bias) |
| 0.075 | 0.089 | small, both low | good fit |
| 0.056 | 0.524 | large | overfitting (high variance) |

(Degrees 1, 4, and 15 from the figure.) The degree-15 model has the *lowest* training error and validation error almost 10 times higher than degree 4: lower training loss, worse generalization.

### Worked example: a tiny bias-variance calculation

Suppose the truth at some $x$ is $f(x) = 5$, and fitting the same model on four different training sets gives predictions $4, 6, 7, 3$. The average prediction is $5$, so bias $= 5 - 5 = 0$. Variance $= \frac{(4-5)^2 + (6-5)^2 + (7-5)^2 + (3-5)^2}{4} = \frac{1 + 1 + 4 + 4}{4} = 2.5$. This model is unbiased but unstable. A model that always predicts 4 has variance 0 and bias² $= 1$: lower total error here (1 versus 2.5).

## 4. Implementation

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

model = make_pipeline(PolynomialFeatures(), LinearRegression())
degrees = np.arange(1, 16)
train_scores, val_scores = validation_curve(
    model, X, y, param_name="polynomialfeatures__degree", param_range=degrees,
    cv=5, scoring="neg_mean_squared_error")
best_degree = degrees[val_scores.mean(axis=1).argmax()]

# Learning curve: does more data help?
sizes, tr, va = learning_curve(model.set_params(polynomialfeatures__degree=best_degree), X, y, cv=5)
```

Runnable script with the figure: [`code/02-ml-workflow/workflow.py`](../../code/02-ml-workflow/workflow.py).

## 5. Engineering

**Remedies by diagnosis.**

| If you see | Try |
|---|---|
| Underfitting | more expressive model, better features, less regularization, train longer |
| Overfitting | more data, regularization (L1/L2, dropout), simpler model, early stopping, data augmentation, ensembling (bagging) |
| Both errors fine offline but bad in production | not a bias-variance problem: check leakage, drift, and training-serving skew |

**Learning curves** plot error against training-set size. If validation error is still dropping as you add data, more data will help (variance). If training and validation errors have converged at a high value, more data won't help; you need a better model or features (bias).

**Modern deep learning** often uses very large models that could memorize the training set, then controls variance with data scale, regularization, and early stopping. The U-shape still describes what you observe as training progresses (epochs).

> [!WARNING]
> **Failure modes.** Choosing complexity by training error; tuning so many hyperparameters on one validation set that you overfit *it*; interpreting a small train/validation gap as success when both errors are high.

### Common mistakes

- Thinking overfitting means "high training accuracy." It means a large gap between training and validation performance.
- Adding data to fix underfitting.
- Forgetting that noise sets a floor no model can beat.

## 6. Knowledge check

<!-- quiz:overfitting-and-bias-variance -->
**[Take the bias-variance quiz](../../quizzes/overfitting-and-bias-variance.md)**
<!-- /quiz -->

**Practice exercise.** A model's predictions at one point across five training sets are 9, 11, 10, 12, 8, and the truth is 12. Compute bias² and variance.

<details>
<summary>Solution</summary>

Mean prediction 10. Bias² $= (12 - 10)^2 = 4$. Variance $= \frac{1 + 1 + 0 + 4 + 4}{5} = 2$.
</details>

**Implementation challenge.** Simulate the decomposition: draw 200 training sets from the sine data, fit polynomials of degree 1, 4, and 15 on each, and estimate bias² and variance at 100 test points. Plot both against degree.

## Summary

- Underfitting: high error everywhere (bias). Overfitting: low training error, high validation error (variance).
- Expected error = bias² + variance + irreducible noise; complexity trades bias for variance.
- Choose complexity with validation curves; use learning curves to decide whether more data will help.
- Lower training loss with worse validation performance is overfitting.

**Next:** [Linear regression](../03-regression/01-linear-regression.md)

**Related:** [Regularization](../03-regression/02-regularization-and-regression-metrics.md) · [Training and regularization for neural networks](../12-training-regularization/01-training-and-regularization.md)
