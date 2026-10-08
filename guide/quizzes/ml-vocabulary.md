<!-- GENERATED from ml-vocabulary.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: The language of ML

Covers the lesson [The language of ML](../lessons/01-ml-vocabulary/01-ml-vocabulary.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/ml-vocabulary/) grades these interactively and tracks a review queue.

## 1. Match (easy)

Match each item from a decision-tree project to its term.

| Concept | Options |
|---|---|
| Maximum tree depth | Label |
| The split threshold chosen at the root | Feature |
| Customer age | Parameter |
| Whether the customer churned | Hyperparameter |

<details>
<summary>Answer</summary>

- Maximum tree depth → Hyperparameter
- The split threshold chosen at the root → Parameter
- Customer age → Feature
- Whether the customer churned → Label

Depth is set before training; split thresholds are learned from data.

</details>

## 2. Calculation (easy)

A model predicts 82 for a sample whose label is 75. What is the squared-error loss?

<details>
<summary>Answer</summary>

**49**

Residual $75 - 82 = -7$; squared $49$.

</details>

## 3. Calculation (easy)

With $w = [2, -1]$, $b = 3$, and $x = [4, 5]$, what is $\hat{y} = w \cdot x + b$?

<details>
<summary>Answer</summary>

**6**

$2 \times 4 - 1 \times 5 + 3 = 8 - 5 + 3 = 6$.

</details>

## 4. Multiple choice (medium)

A fraud model is trained with binary cross-entropy but judged by "fraud caught when reviewing the top 100 alerts per day." What is this an example of?

- **A.** A bug; loss and metric must be identical.
- **B.** Loss and metric serving different purposes: one trains, one judges business value.
- **C.** Overfitting.
- **D.** Data leakage.

<details>
<summary>Answer</summary>

**B.** Loss and metric serving different purposes: one trains, one judges business value.

Cross-entropy is differentiable and trainable; "precision at 100" reflects the real constraint. Picking a threshold or ranking that serves the metric is a separate step.

</details>

## 5. Multiple choice (medium)

Which number best estimates how the model will perform for new customers next month?

- **A.** Training accuracy
- **B.** Accuracy on a held-out set drawn from the most recent period
- **C.** Accuracy on the data used to choose hyperparameters
- **D.** The final training loss

<details>
<summary>Answer</summary>

**B.** Accuracy on a held-out set drawn from the most recent period

Only data the model and your tuning never saw, split the way the future differs, estimates generalization.

</details>

## 6. Reflection (hard)

Your model scored well offline, but in production its predictions are noticeably worse from day one. Name one likely vocabulary-level cause and how to check it.

<details>
<summary>Answer</summary>

**Model answer.** Training-serving skew: features at inference are computed differently from training (different code path, units, missing-value handling, or time window). Log production feature values and compare their distributions against the training features for the same entities; better, share one feature pipeline for both.

Inference must reproduce training-time feature computation exactly.

</details>
