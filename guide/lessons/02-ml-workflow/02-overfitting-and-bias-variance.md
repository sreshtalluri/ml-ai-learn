---
title: Overfitting and the bias-variance trade-off
summary: Diagnose underfitting and overfitting from training and validation error, understand bias and variance, and pick model complexity with validation curves.
skill: classical-ml
minutes: 30
prerequisites: [ml-workflow, linear-regression]
related: [regularization-and-regression-metrics, training-and-regularization, k-nearest-neighbors]
adapted_from: [phases/02-ml-fundamentals/10-bias-variance]
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

<!-- lab:fit-explorer -->
![Four panels. Degree 1 underfits the sine-shaped data (train MSE 0.275, validation 0.27). Degree 4 follows it (0.075 and 0.09). Degree 15 wiggles wildly (0.056 and 0.52). A validation curve shows training error falling steadily with degree while validation error is lowest around degree 3 to 5 and rises after.](../../figures/overfitting-and-bias-variance.png)

*Synthetic data: 30 training points from $y = \sin(2\pi x)$ plus noise, 200 validation points. Training error keeps falling as complexity grows; validation error falls, bottoms out (degree 5 here), then rises.*

*Interactive version: [open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/fit-explorer/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Before you move anything: at degree 15 with 20 points, will validation error be higher or lower than at degree 3? Check with the slider.
2. Keep degree 12 and raise training points to 60. Predict what happens to the gap between the two curves, then look.
3. Set noise to 0. Which degree is best now, and why does the overfitting almost disappear?

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

You can't measure bias and variance from one dataset: both terms average over *many possible training sets*. With synthetic data you know the true function, so you can draw 200 training sets, fit 200 models, and compute the decomposition from section 3 directly.

### Build it from scratch

The data generator is the "truth" plus Gaussian noise with σ = 0.5, so the irreducible error σ² is 0.25.

```python
NOISE = 0.5

def true_f(x):
    return np.sin(1.5 * x) + 0.5 * x

def sample(n, rng):
    x = rng.uniform(-3, 3, n)
    return x, true_f(x) + rng.normal(0, NOISE, n)
```

A polynomial fit is linear regression on the columns 1, x, x², ... (`np.vander` builds them). The optional L2 penalty `lam` is ridge regression; it skips the intercept so the penalty does not pull predictions toward zero. The `1e-10` keeps the solve stable when high powers make the columns nearly collinear.

```python
def fit_poly(x, y, degree, lam=0.0):
    """Least squares on [1, x, x², ...]. lam > 0 adds an L2 penalty (not on the intercept)."""
    X = np.vander(x, degree + 1, increasing=True)          # [n, degree+1]
    P = lam * np.eye(degree + 1)
    P[0, 0] = 0
    return np.linalg.solve(X.T @ X + P + 1e-10 * np.eye(degree + 1), X.T @ y)

def predict_poly(x, w):
    return np.vander(x, len(w), increasing=True) @ w
```

**Predict:** a degree-10 polynomial has how many parameters? With 50 training points, is the fit forced to pass through every point?

The decomposition itself: stack every model's predictions on a fixed test grid into a `[n_sets, 100]` array. The column mean is the "average model". Its squared distance from the truth is bias²; the spread of each column is variance.

```python
def decompose(degree, lam=0.0, n_sets=200, n_train=50, seed=42):
    rng = np.random.default_rng(seed)
    x_test = np.linspace(-2.5, 2.5, 100)
    preds = np.array([predict_poly(x_test, fit_poly(*sample(n_train, rng), degree, lam))
                      for _ in range(n_sets)])              # [n_sets, 100]
    avg = preds.mean(axis=0)
    bias2 = np.mean((avg - true_f(x_test)) ** 2)            # how far the AVERAGE model is from the truth
    var = np.mean(preds.var(axis=0))                        # how much models disagree with each other
    return bias2, var
```

**Predict:** sketch the bias² and variance columns for degrees 1, 3, 5, 8, 10 before you run it. Where is expected MSE (bias² + variance + 0.25) lowest? Then: adding a strong L2 penalty (λ = 10) to the degree-10 model, which term goes up and which goes down?

<details>
<summary>Answers</summary>

11 parameters (powers 0 to 10). With 50 points it can't hit them all, but it has enough freedom to chase the noise. Bias² falls from 0.43 (degree 1, a straight line through a sine wave) to about 0.001 by degree 5 and stays there; variance climbs from about 0.03 to 0.06. Expected MSE is lowest at degree 5 (0.278). Beyond that, extra flexibility buys almost no bias and costs variance. With λ = 10, bias² rises to 0.14 and variance falls only slightly (0.060 to 0.051): too strong a penalty for this data. The right λ is a validation question.

</details>

### Use the library

In real work you have one dataset and no `true_f`, so you can't compute bias² and variance separately. You estimate their sum with cross-validation and read the shape of the curves instead:

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

`validation_curve` sweeps one hyperparameter (complexity) and `learning_curve` sweeps training-set size. A large, persistent train/validation gap is the variance signature; both curves flat and high is the bias signature. scikit-learn scores are "higher is better", hence `neg_mean_squared_error` and `argmax`.

### Run it

From `guide/`:

```bash
uv run code/02-ml-workflow/bias_variance_scratch.py
```

```text
irreducible noise σ² = 0.250
degree    bias²  variance  expected MSE
     1   0.4304    0.0286        0.7090
     3   0.0583    0.0223        0.3306
     5   0.0012    0.0271        0.2783
     8   0.0002    0.0441        0.2943
    10   0.0003    0.0600        0.3104
degree 10 with an L2 penalty:
  λ=0.0   bias² 0.0003  variance 0.0600
  λ=0.1   bias² 0.0012  variance 0.0537
  λ=10.0  bias² 0.1418  variance 0.0513
sklearn validation_curve: alpha=0.0001  CV MSE 0.267
sklearn validation_curve: alpha=0.01    CV MSE 0.379
sklearn validation_curve: alpha=1       CV MSE 0.720
sklearn validation_curve: alpha=100     CV MSE 1.196
checks pass: low degree = high bias, high degree = high variance
```

The data is synthetic, so the numbers are illustrative, not a benchmark. Full script: [`code/02-ml-workflow/bias_variance_scratch.py`](../../code/02-ml-workflow/bias_variance_scratch.py). The figure and diagnosis-table numbers come from [`code/02-ml-workflow/workflow.py`](../../code/02-ml-workflow/workflow.py).

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

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Bias-Variance Tradeoff](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/10-bias-variance).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Explain the bias-variance trade-off. Where do K in KNN, tree depth, and ensembles sit on it?</strong></summary>

Expected squared error at a point decomposes into bias² (how far the average fitted model is from the truth), variance (how much the fit changes across training sets), and irreducible noise $\sigma^2$. Making a model more flexible usually lowers bias and raises variance, so test error is U-shaped in complexity. KNN with $K = 1$ is low bias and high variance: the prediction follows individual noisy points. Large $K$ averages many neighbors, lowering variance but smoothing over real structure, so bias rises. A deep decision tree is low bias and high variance; a stump is the opposite. Ensembles each target one term: bagging and random forests average decorrelated deep trees to cut variance, and boosting adds shallow trees sequentially to cut bias. Regularization strength $\lambda$ moves a model along the same axis.

</details>

<details>
<summary><strong>Model 1 has training error 0.05 and validation error 0.50. Model 2 has 0.27 and 0.27. Diagnose each and say what you'd try.</strong></summary>

Model 1 is overfitting (high variance): it fits the training data but the large gap means it learned noise. Try more data, regularization (L2, dropout, shallower trees, larger $K$), a simpler model, early stopping, data augmentation, or bagging. Model 2 is underfitting (high bias): the gap is small, but both errors are high, and a small gap is not success. Try a more expressive model, better features (interactions, nonlinear transforms), less regularization, or longer training; more data won't help much. To confirm, plot learning curves against training-set size. A validation curve still falling means more data will help; two curves converged at a high plateau mean you need capacity or features. Also compare against the noise floor, such as human-level or best-known error, so you know how much improvement is achievable.

</details>

<details>
<summary><strong>The true value at some x is 10. Trained on four different datasets, model A predicts 8, 9, 7, 8 and model B predicts 6, 14, 10, 10. Compute bias and variance for each. Which would you deploy?</strong></summary>

Model A: mean prediction $(8 + 9 + 7 + 8)/4 = 8$, so bias $= 8 - 10 = -2$ and bias² $= 4$. Variance $= \frac{0^2 + 1^2 + (-1)^2 + 0^2}{4} = 0.5$. Expected error, excluding noise, $= 4.5$. Model B: mean $(6 + 14 + 10 + 10)/4 = 10$, so bias $= 0$. Variance $= \frac{16 + 16 + 0 + 0}{4} = 8$, expected error $8$. The biased but stable model A wins, even though B is right on average. That's the point being probed: unbiasedness isn't the goal, total error is. Ridge and lasso deliberately add bias to cut variance for the same reason. And if you could average many independently trained copies of B, its variance would shrink toward zero, which is exactly the motivation for bagging low-bias, high-variance models like deep trees.

</details>

<details>
<summary><strong>Why does early stopping act as a regularizer?</strong></summary>

Gradient descent from small initial weights moves the model gradually from simple to complex: early epochs fit the dominant, generalizable structure, and later epochs increasingly fit idiosyncrasies of the training set. Validation error therefore traces the same U-shape over epochs that it traces over model complexity. Early stopping monitors validation loss, keeps the best checkpoint, and stops after it fails to improve for a patience window, so the number of steps becomes the complexity knob. For linear models with squared loss there's a formal link: gradient descent stopped after $t$ steps with learning rate $\eta$ behaves approximately like ridge regression with $\lambda \propto 1/(\eta t)$. Practical notes: it's nearly free, but you are now selecting on the validation set, so report results on a separate test set, and use enough patience that a noisy validation curve doesn't stop training too early.

</details>
