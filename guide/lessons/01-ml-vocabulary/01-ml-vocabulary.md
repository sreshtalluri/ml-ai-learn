---
title: The language of ML
summary: Name every object in a machine learning problem precisely, and keep loss vs metric, parameter vs hyperparameter, and training vs inference apart.
skill: classical-ml
minutes: 20
prerequisites: [vectors-and-matrices]
related: [learning-paradigms, linear-regression, ml-workflow]
---

# The language of ML

> **Mental model.** A model is a parameterized function. Training searches for parameter values that make its predictions useful on examples it has not seen. Everything else is vocabulary for the pieces of that sentence.

**You will learn to**
- Define sample, feature, target, label, parameter, hyperparameter, prediction, loss, metric, training, inference, and generalization.
- Compute a prediction, residual, and squared error for one example.
- Say why a loss and a metric are often different functions.
- Explain what "generalization" means and why it is the actual goal.

**Why it matters.** Imprecise words cause real bugs: tuning a hyperparameter on the test set, optimizing a loss nobody cares about, reporting a training metric as if it were production performance. Precise vocabulary makes those mistakes visible.

## 1. Intuition

You want to predict house prices.

- Each house in your data is a **sample** (also called an observation, example, or row).
- What you know about it (size, bedrooms, location) are **features**, written $x$.
- What you want to predict (the sale price) is the **target**; in a training set it is the **label**, written $y$.
- Your **model** is a function $f_\theta(x)$ that outputs a **prediction** $\hat{y}$.
- The numbers inside the model that training adjusts (slopes, weights) are **parameters** $\theta$.
- Settings you choose *before* training (how complex the model may be, how fast it learns) are **hyperparameters**.
- A **loss** scores how wrong a prediction is in a way the optimizer can minimize.
- A **metric** is how people judge the model: RMSE in dollars, precision, latency.
- **Training** fits the parameters on known examples; **inference** uses the fitted model on new ones.
- **Generalization** is how well the model does on data it never saw. That is the only performance that matters.

## 2. Visualization

![A scatter of synthetic houses (size versus price) with a fitted line labeled as the model with its parameters, one residual drawn as a vertical arrow, and a star marking a prediction for a new house.](../../figures/ml-vocabulary.png)

*Synthetic data. Every labeled object in the figure is one of the terms above.*

## 3. The math

### Symbols

| Term | Symbol | Example |
|---|---|---|
| Sample | $(x_i, y_i)$ | one house |
| Feature vector | $x_i \in \mathbb{R}^p$ | `[2 bedrooms, 1.5 thousand sq ft]` |
| Target / label | $y_i$ | sale price |
| Parameters | $\theta$ (e.g. $w$, $b$) | regression slope and intercept |
| Hyperparameter | e.g. $\eta$, $\lambda$, $K$ | learning rate, penalty strength, number of neighbors |
| Prediction | $\hat{y}_i = f_\theta(x_i)$ | predicted price |
| Loss | $\ell(\hat{y}, y)$; training objective $L(\theta) = \frac{1}{n}\sum_i \ell(\hat{y}_i, y_i)$ | squared error |
| Metric | e.g. RMSE, F1 | error in dollars |

### Training versus inference

```math
\text{training: } \theta^* = \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n} \ell\big(f_\theta(x_i), y_i\big) \qquad \text{inference: } \hat{y}_{\text{new}} = f_{\theta^*}(x_{\text{new}})
```

### Worked example

A model has parameters $w = [30, 120]$ (thousand dollars per bedroom and per thousand sq ft) and $b = 50$. A house has $x = [2, 1.5]$ and sold for $y = 310$ thousand.

1. Prediction: $\hat{y} = 2 \times 30 + 1.5 \times 120 + 50 = 60 + 180 + 50 = 290$.
2. Residual: $y - \hat{y} = 310 - 290 = 20$.
3. Squared-error loss: $20^2 = 400$ (thousand dollars squared, a unit nobody reports).
4. Metric for humans: across a test set you would report RMSE, the square root of the mean squared error, back in thousands of dollars.

Here, the loss and the metric are related. Often they aren't: a fraud model trains on cross-entropy but the business cares about recall at a fixed review budget.

## 4. Implementation

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()                 # hyperparameters are constructor arguments
model.fit(X_train, y_train)                # training: learns parameters coef_ and intercept_
print(model.coef_, model.intercept_)       # parameters
y_pred = model.predict(X_test)             # inference
rmse = mean_squared_error(y_test, y_pred) ** 0.5   # metric, on data the model never saw
```

Runnable script for this figure: [`code/02-ml-workflow/workflow.py`](../../code/02-ml-workflow/workflow.py).

## 5. Engineering

**Who sets what.** Parameters come from data via the optimizer. Hyperparameters come from you, chosen on a validation set (never the test set). Configuration such as feature lists and thresholds is part of the model version too.

**Training-serving skew.** Inference must compute features exactly as training did. A feature computed one way in a notebook and another way in the service silently degrades predictions.

**Generalization is measured, not assumed.** A training score says how well the model memorized. Only held-out data, split the way production data will differ, estimates generalization.

> [!IMPORTANT]
> **Loss versus metric.** The loss is what the optimizer minimizes (differentiable, per example). The metric is what stakeholders judge (can be non-differentiable, can depend on thresholds and costs). Choose the metric first, then a loss that pushes in the same direction.

### Common mistakes

- Calling the learning rate a "parameter," then trying to learn it with gradient descent.
- Reporting training accuracy as model quality.
- Tuning hyperparameters by looking at the test set.
- Optimizing accuracy when error costs are asymmetric.

## 6. Knowledge check

<!-- quiz:ml-vocabulary -->
**[Take the vocabulary quiz](../../quizzes/ml-vocabulary.md)**
<!-- /quiz -->

**Practice exercise.** In a spam filter built with logistic regression, classify each as feature, label, parameter, hyperparameter, loss, or metric: (a) the word count of "free"; (b) the regularization strength C; (c) the weight on "free"; (d) whether the user marked the email as spam; (e) binary cross-entropy; (f) precision on last week's emails.

<details>
<summary>Solution</summary>

(a) feature; (b) hyperparameter; (c) parameter; (d) label; (e) loss; (f) metric.
</details>

**Implementation challenge.** Fit `LinearRegression` on a synthetic dataset and print, side by side, training RMSE and test RMSE. Then add 50 random noise features and print both again. Explain the change in terms of generalization.

## Summary

- Samples have features $x$ and, in training data, labels $y$; the model maps $x$ to a prediction $\hat{y}$.
- Parameters are learned; hyperparameters are chosen; both define a model version.
- The loss drives optimization; the metric reflects the real goal; they can differ.
- Training fits on known data; inference predicts on new data; generalization is what counts.

**Next:** [Learning paradigms](02-learning-paradigms.md)

**Related:** [Linear regression](../03-regression/01-linear-regression.md) · [The ML workflow](../02-ml-workflow/01-ml-workflow.md) · [Glossary](../../glossary.md)
