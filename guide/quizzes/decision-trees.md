<!-- GENERATED from decision-trees.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Decision trees

Covers the lesson [Decision trees](../lessons/06-trees-and-ensembles/01-decision-trees.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/decision-trees/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

A node has class counts [3, 1]. What is its Gini impurity?

<details>
<summary>Answer</summary>

**0.375** (within ±0.0001)

$1 - (0.75^2 + 0.25^2) = 1 - (0.5625 + 0.0625) = 0.375$.

</details>

## 2. Calculation (medium)

Parent [6, 6]. A split gives left [5, 1] and right [1, 5]. What is the Gini decrease? (4 decimals)

<details>
<summary>Answer</summary>

**0.2222** (within ±0.001)

Parent Gini 0.5. Each child: $1 - (5/6)^2 - (1/6)^2 = 1 - 0.6944 - 0.0278 = 0.2778$. Weighted 0.2778. Decrease $0.5 - 0.2778 = 0.2222$.

</details>

## 3. Calculation (medium)

What is the entropy (in bits) of a node with counts [4, 4]?

<details>
<summary>Answer</summary>

**1** (within ±0.0001)

$-2 \times 0.5\log_2 0.5 = 1$ bit, the maximum for two classes.

</details>

## 4. Multiple choice (easy)

Why don't decision trees need feature scaling?

- **A.** They standardize features internally.
- **B.** Each split compares one feature to a threshold, so rescaling a feature just rescales its threshold.
- **C.** They only work with binary features.
- **D.** They do need scaling.

<details>
<summary>Answer</summary>

**B.** Each split compares one feature to a threshold, so rescaling a feature just rescales its threshold.

Splits depend on the ordering of values, not their scale.

</details>

## 5. Select all that apply (medium)

Which settings reduce overfitting in a single decision tree? Select all that apply.

- **A.** Lower max_depth
- **B.** Higher min_samples_leaf
- **C.** Cost-complexity pruning (ccp_alpha > 0)
- **D.** Removing all depth limits

<details>
<summary>Answer</summary>

**A, B, C**

All three restrict how finely the tree can carve the training data.

</details>

## 6. Reflection (hard)

A tree's top split is on `customer_id`, with a huge impurity decrease. What is going on and what do you do?

<details>
<summary>Answer</summary>

**Model answer.** An ID has a unique value per row, so a tree can split it to isolate individual examples and look very "pure" on training data, but it carries no generalizable information (new customers have new IDs). Drop ID-like columns, and check feature importance with permutation importance on validation data rather than impurity importance.

High-cardinality features inflate impurity-based importance.

</details>
