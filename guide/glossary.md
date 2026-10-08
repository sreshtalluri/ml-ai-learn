# Glossary

Plain-English definitions first, then formal ones, with a small example and the lesson that teaches each term. On the [website](https://sreshtalluri.github.io/ml-ai-learn/glossary/) this page is searchable and terms show hover definitions inside lessons.

### Parameter

**Plain English:** A number the model learns from data during training.

**Formal:** An entry of $\theta$, the vector of values the optimizer adjusts to minimize the loss $L(\theta)$.

**Example:** The slope $w$ and bias $b$ in $\hat{y} = wx + b$.

**Related:** [Hyperparameter](#hyperparameter), [Gradient](#gradient)

**Lesson:** [Linear regression](lessons/03-regression/01-linear-regression.md)

### Hyperparameter

**Plain English:** A setting you choose before training that controls how learning happens.

**Formal:** A value fixed outside the optimization of $\theta$, usually selected on a validation set.

**Example:** Learning rate, tree depth, $K$ in K-nearest neighbors.

**Related:** [Parameter](#parameter)

**Lesson:** [Linear regression](lessons/03-regression/01-linear-regression.md)
