<!-- GENERATED from ml-workflow.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Workflow, splits, and leakage

Covers the lesson [Workflow, splits, and leakage](../lessons/02-ml-workflow/01-ml-workflow.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/ml-workflow/) grades these interactively and tracks a review queue.

## 1. Match (medium)

Match each problem to the right split.

| Concept | Options |
|---|---|
| Forecast next week's sales | Random split |
| Diagnose disease from multiple scans per patient | Stratified random split |
| Classify independent product reviews, 2% negative | Group split by patient |
| Predict house prices from a single snapshot of listings | Time split |

<details>
<summary>Answer</summary>

- Forecast next week's sales → Time split
- Diagnose disease from multiple scans per patient → Group split by patient
- Classify independent product reviews, 2% negative → Stratified random split
- Predict house prices from a single snapshot of listings → Random split

Match the split to how production data will differ from training data.

</details>

## 2. Select all that apply (medium)

Which are data leakage? Select all that apply.

- **A.** Fitting a StandardScaler on the full dataset before the train-test split
- **B.** Including `days_until_cancellation` when predicting cancellation
- **C.** Using a customer's account age at prediction time
- **D.** Exact duplicate rows appearing in both train and test

<details>
<summary>Answer</summary>

**A, B, D**

Scaler statistics include test data; days-until-cancellation encodes the label; duplicates let the model memorize test answers. Account age at prediction time is legitimate.

</details>

## 3. Calculation (easy)

In a dataset with 950 negatives and 50 positives, what accuracy does "always predict negative" achieve?

<details>
<summary>Answer</summary>

**0.95** (within ±0.0001)

950 / 1000 = 0.95. Any model must be judged against this, and accuracy is the wrong metric here.

</details>

## 4. Multiple choice (hard)

Your churn model has 0.99 AUC, and 80% of its importance comes from `last_login_date`. What is most likely?

- **A.** The model found a genuinely powerful signal.
- **B.** The feature leaks the outcome (churned customers stop logging in, so the date was taken after churn began).
- **C.** The model is underfitting.
- **D.** AUC is miscalculated.

<details>
<summary>Answer</summary>

**B.** The feature leaks the outcome (churned customers stop logging in, so the date was taken after churn began).

An overwhelming single feature and a near-perfect score on a hard problem are classic leakage smells. Check when the feature is computed relative to the prediction time.

</details>

## 5. Multiple choice (medium)

You compared 40 model variants on the test set and picked the best. What is wrong with reporting its test score?

- **A.** Nothing.
- **B.** The test set has been used for selection, so the score is optimistically biased; use a validation set or CV for selection and keep a fresh test set.
- **C.** 40 is too few variants.
- **D.** The test set should be larger than the training set.

<details>
<summary>Answer</summary>

**B.** The test set has been used for selection, so the score is optimistically biased; use a validation set or CV for selection and keep a fresh test set.

Selecting on the test set turns it into a validation set. The winner's score includes luck.

</details>

## 6. Arrange in order (easy)

Order the workflow steps.

- Deploy and monitor
- Evaluate and analyze errors
- Train and tune models
- Fit a baseline
- Split into train, validation, and test
- Explore for missing values, duplicates, and leakage
- Collect and version data
- Frame the decision and metric

<details>
<summary>Answer</summary>

1. Frame the decision and metric
2. Collect and version data
3. Explore for missing values, duplicates, and leakage
4. Split into train, validation, and test
5. Fit a baseline
6. Train and tune models
7. Evaluate and analyze errors
8. Deploy and monitor

Framing and splitting come before any modeling.

</details>
