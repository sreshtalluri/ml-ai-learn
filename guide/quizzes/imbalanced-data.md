<!-- GENERATED from imbalanced-data.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Imbalanced data

Covers the lesson [Imbalanced data](../lessons/24-applied-ml/02-imbalanced-data.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/imbalanced-data/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

A fraud dataset has 99.9% legitimate transactions and 0.1% fraud. A model predicts "legitimate" for every transaction. What is its accuracy?

- **A.** 99.9%
- **B.** 0.1%
- **C.** 50%
- **D.** 100%

<details>
<summary>Answer</summary>

**A.** 99.9%

Accuracy is 999 correct out of 1,000, so 99.9%, while the model catches zero fraud. This is why accuracy is misleading on rare positives.

- **A:** Correct. Every legitimate transaction is a true negative.
- **B:** 0.1% is the fraud rate, which would be the accuracy of always predicting fraud.
- **C:** 50% would only come from random guessing on balanced classes.
- **D:** The model misses every fraud case, so it can't be 100%.

</details>

## 2. Multiple choice (easy)

Which metric exposes the always-predict-negative model as useless?

- **A.** Accuracy
- **B.** True negative rate (specificity)
- **C.** Recall or F1 on the positive class
- **D.** The fraction of rows predicted negative

<details>
<summary>Answer</summary>

**C.** Recall or F1 on the positive class

Recall = TP / (TP + FN) = 0, so F1 = 0 as well. Metrics centered on the positive class show it catches nothing.

- **A:** Accuracy is 99.9% here, which hides the failure.
- **B:** Specificity is 100% because every negative is correctly rejected.
- **C:** Correct. Both are 0.
- **D:** That fraction is 100% and says nothing about catching positives.

</details>

## 3. Multiple choice (easy)

How does SMOTE create synthetic minority samples?

- **A.** By flipping the labels of majority samples near the boundary
- **B.** By interpolating between a minority sample and one of its k nearest minority neighbors
- **C.** By sampling points uniformly anywhere in feature space
- **D.** By duplicating existing minority samples exactly

<details>
<summary>Answer</summary>

**B.** By interpolating between a minority sample and one of its k nearest minority neighbors

SMOTE picks a minority point x_i, one of its k nearest minority neighbors x_j, and returns x_i + λ(x_j − x_i) with λ uniform on [0, 1).

- **A:** Flipping labels corrupts data; SMOTE never changes existing labels.
- **B:** Correct.
- **C:** Uniform sampling would scatter positives into regions with no positive evidence.
- **D:** That is random oversampling, which SMOTE was designed to improve on.

</details>

## 4. Multiple choice (medium)

You lower the classification threshold from 0.5 to 0.3. What usually happens?

- **A.** Precision and recall both increase
- **B.** Precision increases and recall decreases
- **C.** Recall increases and precision usually decreases
- **D.** Neither changes; the threshold only affects speed

<details>
<summary>Answer</summary>

**C.** Recall increases and precision usually decreases

More rows are predicted positive. You catch more true positives (recall cannot go down) but usually admit more false positives, so precision tends to fall.

- **A:** Lowering the threshold trades precision for recall; it doesn't improve both at once.
- **B:** That is what raising the threshold does.
- **C:** Correct.
- **D:** The threshold directly decides which rows are flagged.

</details>

## 5. Multiple choice (medium)

Why is AUPRC often more informative than ROC AUC on highly imbalanced data?

- **A.** ROC AUC cannot be computed on imbalanced data
- **B.** AUPRC uses a single fixed threshold
- **C.** AUPRC is always higher than ROC AUC
- **D.** A random ranking scores the positive rate on AUPRC but 0.5 on ROC AUC, so gains on the rare class are easier to see

<details>
<summary>Answer</summary>

**D.** A random ranking scores the positive rate on AUPRC but 0.5 on ROC AUC, so gains on the rare class are easier to see

The false positive rate in ROC divides by the huge negative count, so many false positives barely move it. Precision counts every false positive directly, and AUPRC's random baseline is the prevalence.

- **A:** ROC AUC can be computed; it is just less sensitive to false positives when negatives dominate.
- **B:** Both are threshold-free summaries over all thresholds.
- **C:** On rare positives AUPRC is usually much lower than ROC AUC.
- **D:** Correct.

</details>

## 6. Calculation (medium)

A training set has 2,400 rows, 121 of them positive. Using balanced class weights $w_c = n / (C \cdot n_c)$, what is the weight of one positive row? (Two decimals.)

<details>
<summary>Answer</summary>

**9.92** (within ±0.01)

$w_1 = 2400 / (2 \cdot 121) = 2400 / 242 = 9.917$. The negative weight is $2400 / (2 \cdot 2279) = 0.5265$, so one positive counts as $2279 / 121 = 18.83$ negatives.

</details>

## 7. Calculation (hard)

On a test set, TP = 27, FP = 128, FN = 7, TN = 638. Compute the Matthews correlation coefficient (three decimals).

<details>
<summary>Answer</summary>

**0.32** (within ±0.002)

Numerator: $27 \cdot 638 - 128 \cdot 7 = 17226 - 896 = 16330$.
Denominator: $\sqrt{155 \cdot 34 \cdot 766 \cdot 645} = \sqrt{2603748900} = 51026.9$.
MCC $= 16330 / 51026.9 = 0.320$.

</details>

## 8. Calculation (hard)

A logistic model trained with balanced class weights (negatives outnumber positives 19 to 1) outputs $p_w = 0.6$. Using prior correction (divide the odds by 19), what is the real-world probability? (Three decimals.)

<details>
<summary>Answer</summary>

**0.073** (within ±0.002)

Odds $= 0.6 / 0.4 = 1.5$. Divide by 19: $0.0789$. Probability $= 0.0789 / 1.0789 = 0.073$.

</details>

## 9. Multiple choice (hard)

A teammate applies SMOTE to the whole dataset, then runs 5-fold cross-validation, and F1 jumps from 0.45 to 0.80. What is the most likely explanation?

- **A.** SMOTE genuinely improved the model's ranking ability
- **B.** Synthetic points built from validation-fold positives leaked into training folds, inflating the score
- **C.** Cross-validation is not valid for classification
- **D.** F1 always rises when the training set grows

<details>
<summary>Answer</summary>

**B.** Synthetic points built from validation-fold positives leaked into training folds, inflating the score

Resampling before splitting puts near-copies of evaluation positives into training. Resample inside each training fold only, for example with imbalanced-learn's Pipeline, and keep evaluation folds at the real class ratio.

- **A:** Possible in principle, but a jump this large from resampling alone is a leakage red flag; check AUPRC with a leak-free setup first.
- **B:** Correct.
- **C:** Stratified cross-validation is the standard way to evaluate classifiers.
- **D:** More rows don't guarantee better F1, and these extra rows are synthetic.

</details>

## 10. Multiple choice (medium)

You have 500 million ad impressions with a 0.2% click rate and need to train daily within a fixed compute budget. Which strategy fits best?

- **A.** SMOTE the clicks up to 50% of the data
- **B.** Randomly downsample non-clicks, then correct predicted probabilities for the sampling rate
- **C.** Train on all data with no changes and report accuracy
- **D.** Oversample clicks by duplication until the classes match

<details>
<summary>Answer</summary>

**B.** Randomly downsample non-clicks, then correct predicted probabilities for the sampling rate

With huge negative counts, downsampling negatives cuts training cost with little information loss. Because click probabilities feed bidding, you must undo the sampling, by dividing the odds by the inverse sampling rate or by weighting kept negatives.

- **A:** SMOTE on 1 million positives would add hundreds of millions of synthetic rows, increasing cost, and interpolation in sparse ad features is not meaningful.
- **B:** Correct.
- **C:** Training on everything may not fit the budget, and accuracy is meaningless at a 0.2% click rate.
- **D:** Duplicating clicks also grows the data enormously and adds no new information.

</details>

## 11. Reflection (hard)

Class weights raised recall from 0.44 to 0.79 but AUPRC barely moved (0.548 to 0.551). Explain what the weights did, and how you could get a similar result without them.

<details>
<summary>Answer</summary>

**Model answer.** The weights made the model favor positives by inflating predicted odds (about 18.8 times here), which moves the 0.5 decision boundary toward the negatives. That changes the operating point (more recall, less precision) but not the order in which rows are ranked, which is what AUPRC measures. Lowering the threshold of the unweighted model on a validation set reaches the same operating point: the weighted model's tuned 0.86 maps back to about 0.25 on the unweighted scale, close to the 0.24 found directly.

Re-weighting mostly shifts the intercept of a logistic model, which is equivalent to moving the threshold.

</details>
