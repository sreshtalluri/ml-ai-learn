<!-- GENERATED from feature-engineering.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Feature engineering

Covers the lesson [Feature engineering](../lessons/24-applied-ml/01-feature-engineering.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/feature-engineering/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

Why is feature engineering often more impactful than choosing a fancier algorithm on tabular data?

- **A.** It makes the code run faster
- **B.** It removes the need for a test set
- **C.** Fancy algorithms cannot process raw data
- **D.** Good features expose patterns that raw columns hide, so even simple models can use them

<details>
<summary>Answer</summary>

**D.** Good features expose patterns that raw columns hide, so even simple models can use them

A feature like BMI (weight / height²) or debt-to-income hands the model the relationship directly; a linear model cannot build a ratio from two separate columns.

- **A:** Speed is a side effect at best; added features can make training slower.
- **B:** Every model still needs held-out evaluation, however good its features are.
- **C:** Many algorithms train fine on raw numeric columns; the question is how well they generalize.
- **D:** Correct.

</details>

## 2. Multiple choice (easy)

What does one-hot encoding do?

- **A.** Replaces each category with its frequency in the dataset
- **B.** Creates one 0/1 column per category, with exactly one column set to 1 in each row
- **C.** Encodes the label as a probability
- **D.** Rescales all features to between 0 and 1

<details>
<summary>Answer</summary>

**B.** Creates one 0/1 column per category, with exactly one column set to 1 in each row

color = red/green/blue becomes three columns: is_red, is_green, is_blue. No fake ordering is introduced.

- **A:** That is frequency (count) encoding.
- **B:** Correct.
- **C:** Encoding the label is a different topic; one-hot encodes input categories.
- **D:** That is min-max scaling of numeric features.

</details>

## 3. Multiple choice (medium)

You target-encode a high-cardinality ID column using a table built from all training rows, then train on those same rows. What is the risk?

- **A.** Training becomes too slow
- **B.** Each row's own label is inside its encoding, so the model over-trusts the column and generalizes worse
- **C.** It only works with binary labels
- **D.** It creates too many columns

<details>
<summary>Answer</summary>

**B.** Each row's own label is inside its encoding, so the model over-trusts the column and generalizes worse

For a rare category the encoding is mostly the row's own label. Encode out of fold (cross-fitting) so each training row's encoding never saw its label.

- **A:** Target encoding is cheap; one pass per category.
- **B:** Correct.
- **C:** Target encoding works for regression and multiclass targets too.
- **D:** Target encoding produces one column (per class for multiclass), which is why it is used instead of one-hot.

</details>

## 4. Calculation (medium)

A category appears in 4 training rows with 3 positives. The global positive rate is 0.2. What is its smoothed target encoding with $m = 6$? (Round to 2 decimals.)

<details>
<summary>Answer</summary>

**0.42** (within ±0.005)

$\text{enc} = (n_c \bar y_c + m \bar y)/(n_c + m) = (3 + 6 \times 0.2)/(4 + 6) = (3 + 1.2)/10 = 0.42$.

</details>

## 5. Calculation (easy)

Training values are $[2, 4, 4, 4, 5, 5, 7, 9]$ (mean 5, population std 2). A new row has value 10. What is its standardized value?

<details>
<summary>Answer</summary>

**2.5** (within ±0.001)

Use the training statistics, not the new row's. $(10 - 5)/2 = 2.5$.

</details>

## 6. Multiple choice (medium)

Two features have correlation 0.98 (square feet and square meters, say). Why might you drop one?

- **A.** Highly correlated features always make the model crash
- **B.** They carry nearly the same information, so keeping both adds dimensions and instability without new signal
- **C.** Correlated features make the data non-stationary
- **D.** A correlation above 0.5 means the features measure different things

<details>
<summary>Answer</summary>

**B.** They carry nearly the same information, so keeping both adds dimensions and instability without new signal

Near-duplicates inflate coefficient variance in linear models (multicollinearity) and split importance between two columns in trees, while adding no information.

- **A:** Models train fine; the problem is unstable coefficients and wasted capacity, not crashes.
- **B:** Correct.
- **C:** Stationarity is a time-series property, unrelated to correlation between two columns.
- **D:** High correlation means they move together, so they measure nearly the same thing.

</details>

## 7. Multiple choice (hard)

A teammate's notebook does: `X = StandardScaler().fit_transform(X_all)`, then `train_test_split(X, y)`, then trains and reports test accuracy. What is wrong?

- **A.** Nothing; scaling is unsupervised, so it cannot leak
- **B.** The test rows influenced the mean and standard deviation, so the test score is slightly optimistic; fit the scaler on the training split inside a Pipeline
- **C.** StandardScaler should be replaced with MinMaxScaler
- **D.** The split should come after model training

<details>
<summary>Answer</summary>

**B.** The test rows influenced the mean and standard deviation, so the test score is slightly optimistic; fit the scaler on the training split inside a Pipeline

Any statistic fit on data that includes the test set leaks information about it. For a scaler the effect is usually small, but the same habit with target encoding or feature selection gives large, fake gains. Put preprocessing inside a Pipeline.

- **A:** Unsupervised statistics still describe the test rows; it is leakage, if mild.
- **B:** Correct.
- **C:** The choice of scaler does not fix the order of operations.
- **D:** Splitting after training would mean evaluating on training data.

</details>

## 8. Multiple choice (medium)

The label depends on a feature as $y = x^2$, with $x$ symmetric around 0. Which filter would keep this feature?

- **A.** Pearson correlation with the label
- **B.** Mutual information with the label
- **C.** Variance threshold alone tells you it is predictive
- **D.** Neither; filters cannot detect it

<details>
<summary>Answer</summary>

**B.** Mutual information with the label

Correlation measures linear association, which is about 0 for a symmetric parabola. Mutual information is positive whenever the feature reduces uncertainty about the label, linear or not.

- **A:** Correlation is near 0 here because positive and negative x cancel.
- **B:** Correct.
- **C:** Variance says the column varies, not that it relates to the label.
- **D:** Mutual information detects it.

</details>

## 9. Multiple choice (medium)

You have a merchant ID column with 200,000 values for a fraud model trained with logistic regression. Which encoding is the best first choice?

- **A.** One-hot encoding
- **B.** Label encoding (merchant_id → integer)
- **C.** Cross-fitted target encoding (or frequency/hashing), with a fallback for unseen merchants
- **D.** Drop the column

<details>
<summary>Answer</summary>

**C.** Cross-fitted target encoding (or frequency/hashing), with a fallback for unseen merchants

One-hot would add 200,000 sparse columns, most seen a handful of times. Label encoding imposes a meaningless order on a linear model. A cross-fitted target encoding compresses it to one informative column; unseen merchants get the global mean.

- **A:** Workable with sparse matrices and strong regularization, but huge and very noisy for rare merchants.
- **B:** A linear model would treat merchant 150,000 as "more" than merchant 3.
- **C:** Correct.
- **D:** Merchant identity is often among the strongest fraud signals.

</details>

## 10. Reflection (medium)

Why add a "was_missing" column when you impute a numeric feature with its median?

<details>
<summary>Answer</summary>

**Model answer.** Missingness is often not random (a blank income field may mean self-employed or refused). After imputation the model can no longer tell a real median value from a filled one, so the indicator preserves that signal. It also makes the model robust to changes in missing rates at serving time, which you can monitor.

Imputation hides the fact that the value was missing; the indicator keeps it.

</details>
