---
title: Support vector machines
summary: Find the separating boundary with the widest margin, trade margin for errors with C, and bend the boundary with kernels.
skill: classical-ml
minutes: 30
prerequisites: [logistic-regression, vectors-and-matrices]
related: [logistic-regression, k-nearest-neighbors, text-to-vectors]
---

# Support vector machines

> **Mental model.** Of all the lines that separate two classes, choose the one with the widest empty street around it. Only the points on the edge of the street (the support vectors) decide where it goes. Kernels let the street curve.

**You will learn to**
- Define the margin and support vectors, and compute the margin width from the weights.
- Explain the soft-margin trade-off controlled by C.
- Write the hinge loss and compare it with logistic loss.
- Explain the kernel trick and the RBF kernel's $\gamma$ parameter.
- Decide when an SVM is a good choice in practice.

**Why it matters.** Linear SVMs are excellent on high-dimensional sparse data like TF-IDF text. Kernel SVMs dominated many benchmarks before deep learning and remain strong on small and medium datasets. The margin idea also explains why some models generalize better than others that fit the training data equally well.

## 1. Intuition

Many lines can separate two well-separated groups. A line that barely squeezes past some points will misclassify new points that land slightly off. A line with a wide buffer on both sides is more robust. The SVM maximizes that buffer, the **margin**.

Real data overlaps, so a **soft margin** allows some points inside the street or on the wrong side, at a cost. The hyperparameter **C** sets the price: large C insists on classifying training points correctly (narrow street, risk of overfitting); small C accepts some errors for a wider street (smoother, more regularized).

When no straight line works, the **kernel trick** computes similarities as if the data had been mapped into a richer feature space where a linear separator exists, without ever building that space explicitly.

## 2. Visualization

![Left: a linear SVM with C = 100 has a narrow margin (dashed lines) and 19 circled support vectors. Middle: C = 0.05 has a margin more than twice as wide with 35 support vectors. Right: on two concentric rings, an RBF-kernel SVM draws a circular boundary with 100% accuracy where a linear kernel gets 64%.](../../figures/support-vector-machines.png)

*Synthetic data. Both linear models reach the same training accuracy (0.925); the wider margin is the more regularized choice.*

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $y_i \in \{-1, +1\}$ | labels (SVMs use ±1) |
| $w, b$ | boundary $w \cdot x + b = 0$ |
| $\xi_i \ge 0$ | slack: how far point $i$ violates the margin |
| $C$ | penalty per unit of slack |
| $K(x, x')$ | kernel: similarity between two points |
| $\gamma$ | RBF kernel width parameter |

### Hard margin

The margin edges are $w\cdot x + b = \pm 1$. The distance between them is $2/\lVert w \rVert$, so maximizing the margin means minimizing $\lVert w \rVert$:

```math
\min_{w,b} \tfrac12\lVert w\rVert^2 \quad \text{subject to} \quad y_i(w\cdot x_i + b) \ge 1 \;\text{for all } i
```

### Soft margin and hinge loss

```math
\min_{w,b} \tfrac12\lVert w\rVert^2 + C\sum_i \xi_i
\;\;\Longleftrightarrow\;\;
\min_{w,b} \tfrac12\lVert w\rVert^2 + C\sum_i \max\big(0,\; 1 - y_i(w\cdot x_i + b)\big)
```

The **hinge loss** $\max(0, 1 - y\,f(x))$ is zero for points correctly classified beyond the margin, so they have no influence. That is why only support vectors matter.

### Kernels

The solution depends on the data only through dot products $x_i \cdot x_j$. Replacing them with a kernel gives a nonlinear boundary:

```math
f(x) = \sum_{i \in \text{SV}} \alpha_i y_i K(x_i, x) + b, \qquad K_{\text{RBF}}(x, x') = \exp\!\big(-\gamma\lVert x - x'\rVert^2\big)
```

Large $\gamma$: each support vector influences only its immediate neighborhood (wiggly boundary, overfitting risk). Small $\gamma$: smooth boundary.

### Worked example

A trained linear SVM has $w = (3, 4)$ and $b = -2$.

1. Margin width: $2/\lVert w \rVert = 2/\sqrt{9 + 16} = 2/5 = 0.4$.
2. Point $x = (1, 0.5)$ with $y = +1$: $f(x) = 3 + 2 - 2 = 3$. $y f(x) = 3 \ge 1$, so it is outside the margin: hinge loss 0, not a support vector.
3. Point $x = (0.2, 0.3)$ with $y = +1$: $f(x) = 0.6 + 1.2 - 2 = -0.2$. $y f(x) = -0.2$: misclassified, hinge loss $= 1 - (-0.2) = 1.2$.
4. RBF similarity between $(0, 0)$ and $(1, 1)$ with $\gamma = 0.5$: $\exp(-0.5 \times 2) = e^{-1} = 0.368$.

## 4. Implementation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

# Linear SVM: fast on large sparse data (e.g. TF-IDF)
text_clf = make_pipeline(TfidfVectorizer(), LinearSVC(C=1.0))

# RBF SVM: tune C and gamma together, always with scaling
rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
search = GridSearchCV(rbf, {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1]}, cv=5)
search.fit(X_train, y_train)
```

`TfidfVectorizer` comes from `sklearn.feature_extraction.text`. Runnable script: [`code/05-instance-and-probabilistic/nb_svm.py`](../../code/05-instance-and-probabilistic/nb_svm.py).

## 5. Engineering

**Good fits.** Text classification with TF-IDF (linear SVM); small-to-medium datasets with complex boundaries (RBF); high-dimensional data where $p > n$.

**Poor fits.** Very large $n$ for kernel SVMs: training scales between $O(n^2)$ and $O(n^3)$ and memory holds a kernel matrix. Use linear SVMs, approximate kernels, or gradient boosting instead.

**Preprocessing.** Scale features; SVMs use distances and dot products.

**Probabilities.** SVM outputs are margins, not probabilities. `SVC(probability=True)` fits a calibration step with extra cross-validation; or use `CalibratedClassifierCV`.

> [!WARNING]
> **Failure modes.** Unscaled features; untuned C and γ (defaults are rarely right); kernel SVMs on hundreds of thousands of rows; treating decision-function values as probabilities.

### Common mistakes

- Tuning C without tuning γ (they interact).
- Using an RBF kernel on sparse text where linear works better and is much faster.
- Expecting interpretable weights from a kernel SVM.

## 6. Knowledge check

<!-- quiz:support-vector-machines -->
**[Take the SVM quiz](../../quizzes/support-vector-machines.md)**
<!-- /quiz -->

**Practice exercise.** With $w = (1, -1)$ and $b = 0$, compute the margin width and the hinge loss for $x = (2, 1.5)$, $y = +1$.

<details>
<summary>Solution</summary>

$\lVert w \rVert = \sqrt{2} = 1.414$, so the margin is $2/1.414 = 1.414$. $f(x) = 2 - 1.5 = 0.5$; $y f(x) = 0.5 < 1$, so hinge loss $= 1 - 0.5 = 0.5$: correctly classified but inside the margin, so it is a support vector.
</details>

**Implementation challenge.** Implement a linear SVM with subgradient descent on the hinge loss plus $\frac{\lambda}{2}\lVert w\rVert^2$ and compare its accuracy and weights with `LinearSVC`.

## Summary

- SVMs maximize the margin $2/\lVert w\rVert$ between classes; only support vectors define the boundary.
- C trades margin width against training errors; hinge loss ignores well-classified points.
- Kernels (RBF with width γ) give nonlinear boundaries using only pairwise similarities.
- Linear SVMs shine on sparse text; kernel SVMs don't scale to very large datasets; scale features and calibrate if you need probabilities.

**Next:** [Decision trees](../06-trees-and-ensembles/01-decision-trees.md)

**Related:** [Logistic regression](../04-classification/01-logistic-regression.md) · [K-nearest neighbors](01-k-nearest-neighbors.md) · [Model card: SVM](../../models/support-vector-machine.md)
