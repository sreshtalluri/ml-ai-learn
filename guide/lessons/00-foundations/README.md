---
title: Mathematical and programming foundations
summary: Vectors, matrices, derivatives, probability, and the Python stack, with the shape habit that prevents most bugs.
skill: math
---

# Module 0: Mathematical and programming foundations

> **Core mental model.** A model is a parameterized function. Training searches for parameter values that make its predictions useful on examples it has not seen.

You need only a small, precise toolkit: linear algebra to represent data and models, calculus to know which way to adjust parameters, probability to reason about uncertainty, and NumPy to do it all fast. Each lesson below is built around the exact operations used later in the course.

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Vectors and matrices](01-vectors-and-matrices.md) | compute dot products and matrix products, and track tensor shapes |
| 2 | [Calculus for ML](02-calculus-for-ml.md) | take derivatives, partial derivatives, and gradients, and apply the chain rule |
| 3 | [Probability and statistics](03-probability-and-statistics.md) | use conditional probability, Bayes' rule, expectation, and variance; separate correlation from causation |
| 4 | [The Python toolkit](04-python-toolkit.md) | write vectorized NumPy, load data with pandas, and plot with matplotlib |

**The shape habit.** Write tensor shapes next to every operation. If $X$ is $[\text{batch}, \text{features}]$ and $W$ is $[\text{features}, \text{outputs}]$, then $XW$ is $[\text{batch}, \text{outputs}]$.

**Code:** [`code/00-foundations/`](../../code/00-foundations/)
