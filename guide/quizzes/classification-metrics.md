<!-- GENERATED from classification-metrics.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Classification metrics and thresholds

Covers the lesson [Classification metrics and thresholds](../lessons/04-classification/02-classification-metrics.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/classification-metrics/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

TP = 40, FP = 10, FN = 20, TN = 930. What is precision?

<details>
<summary>Answer</summary>

**0.8** (within ±0.0001)

$40 / (40 + 10) = 0.8$.

</details>

## 2. Calculation (easy)

Same counts (TP = 40, FP = 10, FN = 20, TN = 930). What is recall? (4 decimals)

<details>
<summary>Answer</summary>

**0.6667** (within ±0.001)

$40 / (40 + 20) = 0.667$.

</details>

## 3. Calculation (medium)

Precision 0.8 and recall 0.6667. What is F1? (4 decimals)

<details>
<summary>Answer</summary>

**0.7273** (within ±0.001)

$2(0.8)(0.6667)/(0.8 + 0.6667) = 1.0667/1.4667 = 0.727$.

</details>

## 4. Multiple choice (medium)

You lower the classification threshold from 0.5 to 0.2. What usually happens?

- **A.** Precision rises and recall falls.
- **B.** Recall rises and false positives increase.
- **C.** Both precision and recall rise.
- **D.** Nothing; thresholds don't affect metrics.

<details>
<summary>Answer</summary>

**B.** Recall rises and false positives increase.

More examples are flagged, so more true positives are caught but more negatives are flagged too.

</details>

## 5. Calculation (hard)

At threshold A there are 30 FP and 5 FN; at threshold B, 8 FP and 12 FN. A false negative costs 20 and a false positive costs 2. What is the cost of the cheaper threshold?

<details>
<summary>Answer</summary>

**160**

A $= 30(2) + 5(20) = 160$. B $= 8(2) + 12(20) = 256$. A is cheaper at 160.

</details>

## 6. Select all that apply (medium)

On a dataset with 0.5% positives, which metrics are most informative? Select all that apply.

- **A.** Accuracy
- **B.** Precision and recall at the operating threshold
- **C.** Precision-recall AUC (average precision)
- **D.** Expected cost at the chosen threshold

<details>
<summary>Answer</summary>

**B, C, D**

Always predicting negative gives 99.5% accuracy, so accuracy says almost nothing here.

</details>

## 7. Reflection (hard)

Two models have the same ROC AUC, but one is badly overconfident. When does that matter, and what would you do?

<details>
<summary>Answer</summary>

**Model answer.** AUC only measures ranking, so for "flag the top N" decisions they behave the same. It matters whenever probabilities are used as probabilities: expected-cost thresholds, risk scores shown to users, combining models, or comparing across segments. Check a reliability diagram and recalibrate on held-out data with Platt scaling or isotonic regression.

Ranking quality and calibration are different properties.

</details>
