<!-- GENERATED from anomaly-detection.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Anomaly detection

Covers the lesson [Anomaly detection](../lessons/24-applied-ml/05-anomaly-detection.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/anomaly-detection/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

Why is anomaly detection usually framed as an unsupervised problem rather than classification?

- **A.** Unsupervised models are always more accurate
- **B.** Labeled anomalies are rare, and new anomaly types differ from the ones seen before
- **C.** Classification can't handle numeric features
- **D.** Anomalies never have labels by definition

<details>
<summary>Answer</summary>

**B.** Labeled anomalies are rare, and new anomaly types differ from the ones seen before

Modeling normal data catches deviations of any kind, including novel ones a classifier was never shown.

- **A:** With plenty of labels for a known pattern, supervised models usually win on that pattern.
- **B:** Correct.
- **C:** Classifiers handle numeric features fine.
- **D:** Some anomalies get labeled (chargebacks, incidents); they're just scarce.

</details>

## 2. Multiple choice (easy)

A temperature of 90°F is normal in summer but anomalous in winter. What type of anomaly is this?

- **A.** Point anomaly
- **B.** Contextual anomaly
- **C.** Collective anomaly
- **D.** Label noise

<details>
<summary>Answer</summary>

**B.** Contextual anomaly

The value is only strange given its context (season). Adding context as a feature, or comparing within context, turns it into a detectable point anomaly.

- **A:** 90°F on its own is not unusual.
- **B:** Correct.
- **C:** Collective anomalies are groups that are strange together.
- **D:** Nothing here is mislabeled.

</details>

## 3. Multiple choice (medium)

When does the rule "flag |z| > 3" fail?

- **A.** Only when the data has more than one feature
- **B.** When data is skewed or multimodal, or when outliers inflate the mean and standard deviation
- **C.** When the data is perfectly Gaussian
- **D.** It never fails if there are more than 30 points

<details>
<summary>Answer</summary>

**B.** When data is skewed or multimodal, or when outliers inflate the mean and standard deviation

The rule assumes one roughly Gaussian bump, and outliers in the data distort the very statistics used to judge them.

- **A:** It can fail even in 1-D, as the masking example shows.
- **B:** Correct.
- **C:** Gaussian data is the case it was designed for.
- **D:** Sample size doesn't fix masking or skew.

</details>

## 4. Calculation (medium)

For $x = [10, 11, 12, 12, 13, 14, 50]$ the median is 12 and the MAD is 1. What is the robust z-score $0.6745 (x - \tilde x)/\text{MAD}$ of 50? (2 decimals)

<details>
<summary>Answer</summary>

**25.63** (within ±0.01)

$0.6745 \times (50 - 12)/1 = 0.6745 \times 38 = 25.63$. The plain z-score of 50 is only 2.44.

</details>

## 5. Calculation (hard)

An isolation forest uses ψ = 256, so c(256) = 10.245. A point's average path length is 4. What is its score $s = 2^{-E[h]/c(\psi)}$? (3 decimals)

<details>
<summary>Answer</summary>

**0.763** (within ±0.002)

$4/10.245 = 0.3904$; $2^{-0.3904} = 0.763$. Scores near 1 are anomalous; 0.5 is average.

</details>

## 6. Multiple choice (medium)

How does an isolation forest differ from distance-based detectors?

- **A.** It computes distances to cluster centers
- **B.** It isolates points with random splits; anomalies need fewer splits because they are few and different
- **C.** It requires labeled anomalies to train
- **D.** It fits a Gaussian to each feature

<details>
<summary>Answer</summary>

**B.** It isolates points with random splits; anomalies need fewer splits because they are few and different

No distances are computed; the anomaly score comes from average path length across random trees.

- **A:** That describes centroid-distance scoring.
- **B:** Correct.
- **C:** It is unsupervised.
- **D:** That describes per-feature z-scores or Gaussian density models.

</details>

## 7. Multiple choice (hard)

You fit `LocalOutlierFactor(n_neighbors=20)` on last month's data and call `score_samples` on today's events. It raises an error. What's wrong?

- **A.** LOF can only score training data unless created with novelty=True
- **B.** n_neighbors must equal the number of features
- **C.** LOF requires labels
- **D.** Today's data must be scaled with a different scaler

<details>
<summary>Answer</summary>

**A.** LOF can only score training data unless created with novelty=True

By default LOF is for outlier detection within the fitted data (use negative_outlier_factor_). For scoring new points, set novelty=True.

- **A:** Correct.
- **B:** n_neighbors is unrelated to the number of features.
- **C:** LOF is unsupervised.
- **D:** You should reuse the same fitted scaler, and it isn't the cause of the error.

</details>

## 8. Calculation (medium)

A detector flags its top 30 points; 20 of them are true anomalies, and there are 25 anomalies in total. What is the recall at this budget?

<details>
<summary>Answer</summary>

**0.8** (within ±0.001)

Recall = 20/25 = 0.8. Precision = 20/30 = 0.67.

</details>

## 9. Multiple choice (medium)

You have both an unsupervised anomaly detector and a supervised fraud classifier trained on past chargebacks. When should you lean on the unsupervised detector?

- **A.** When the fraud pattern is well represented in historical labels
- **B.** To catch novel fraud patterns that differ from the labeled history
- **C.** When you need the highest precision on known fraud types
- **D.** Never; supervised models dominate

<details>
<summary>Answer</summary>

**B.** To catch novel fraud patterns that differ from the labeled history

The classifier learns known patterns; the detector flags anything unusual, including new attacks. Production systems often combine them.

- **A:** That's where the supervised model is strongest.
- **B:** Correct.
- **C:** Supervised models usually win on known patterns.
- **D:** They miss what they were never shown.

</details>

## 10. Reflection (hard)

You have 50 labeled past incidents and analysts can review 200 alerts a day. How do you evaluate a new detector?

<details>
<summary>Answer</summary>

**Model answer.** Fix the operating point at the top 200 scores per day and measure precision and recall of the labeled incidents within that budget on held-out days, plus AUPRC (whose random baseline is the incident rate). Don't tune and report on the same incidents. Run the detector in shadow mode and have analysts label a sample of its top alerts, since past labels only cover what the old system caught. Monitor alert volume and analyst precision after launch.

Evaluation follows the alert budget, not overall accuracy.

</details>
