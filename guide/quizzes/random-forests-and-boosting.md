<!-- GENERATED from random-forests-and-boosting.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Random forests and gradient boosting

Covers the lesson [Random forests and gradient boosting](../lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/random-forests-and-boosting/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

Boosting with $F_0 = 50$ and learning rate 0.2. A tree predicts residual $h = 15$ for an example. What is $F_1$ for that example?

<details>
<summary>Answer</summary>

**53** (within ±0.0001)

$50 + 0.2 \times 15 = 53$.

</details>

## 2. Fill in (medium)

Targets $y = [3, 5, 10]$ and $F_0 = \bar{y}$. What are the first residuals? Answer like `[a, b, c]`.

<details>
<summary>Answer</summary>

**[-3, -1, 4]** or **[-3,-1,4]** or **-3,-1,4** or **-3, -1, 4**

$\bar{y} = 6$, so residuals are $3-6, 5-6, 10-6 = [-3, -1, 4]$.

</details>

## 3. Match (medium)

Match each property to bagging or boosting.

| Concept | Options |
|---|---|
| Trees trained independently, in parallel | Boosting |
| Each tree fits the errors of the ensemble so far | Bagging (random forest) |
| Mainly reduces variance | Boosting |
| Mainly reduces bias; needs early stopping | Bagging (random forest) |

<details>
<summary>Answer</summary>

- Trees trained independently, in parallel → Bagging (random forest)
- Each tree fits the errors of the ensemble so far → Boosting
- Mainly reduces variance → Bagging (random forest)
- Mainly reduces bias; needs early stopping → Boosting

Bagging averages; boosting corrects sequentially.

</details>

## 4. Multiple choice (medium)

You halve the learning rate of a gradient-boosting model. What should you usually do with the number of trees?

- **A.** Halve it.
- **B.** Roughly double it (and let early stopping pick the exact number).
- **C.** Keep it the same.
- **D.** Set it to 1.

<details>
<summary>Answer</summary>

**B.** Roughly double it (and let early stopping pick the exact number).

Each tree contributes half as much, so more trees are needed to reach the same fit.

</details>

## 5. Multiple choice (medium)

Why does a random forest consider only a random subset of features at each split?

- **A.** To train faster, with no effect on accuracy.
- **B.** To decorrelate the trees, so averaging reduces variance more.
- **C.** To avoid scaling features.
- **D.** To make each tree shallower.

<details>
<summary>Answer</summary>

**B.** To decorrelate the trees, so averaging reduces variance more.

If every tree picks the same strong feature first, trees are highly correlated and averaging helps less.

</details>

## 6. Reflection (hard)

A gradient-boosting model forecasts sales well in testing, but in a month when prices hit an all-time high its predictions flatten out. Why?

<details>
<summary>Answer</summary>

**Model answer.** Tree ensembles predict piecewise-constant values learned from the training range. For inputs beyond that range, every tree lands in its outermost leaf, so predictions stop changing: tree models can't extrapolate trends. Options: add features that stay in range (ratios, differences), use a linear or hybrid model for the trend, and monitor for inputs outside the training distribution.

Leaves hold constants, so trees can't extrapolate.

</details>
