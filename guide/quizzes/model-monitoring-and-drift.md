<!-- GENERATED from model-monitoring-and-drift.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Model monitoring and drift

Covers the lesson [Model monitoring and drift](../lessons/22-ml-system-design/03-model-monitoring-and-drift.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/model-monitoring-and-drift/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

Which monitoring layer is the only one that directly measures whether the model is still correct?

- **A.** System health (latency, errors)
- **B.** Input distributions
- **C.** Prediction distributions
- **D.** Outcomes (accuracy, calibration on labelled events)

<details>
<summary>Answer</summary>

**D.** Outcomes (accuracy, calibration on labelled events)

Only outcome metrics compare predictions with truth, and they arrive after the label delay.

- **A:** A service can be fast and error-free while being wrong.
- **B:** Inputs can shift without hurting accuracy, and accuracy can fall with no input shift.
- **C:** Predictions show behaviour change, not correctness.
- **D:** Correct.

</details>

## 2. Multiple choice (easy)

Fraudsters change tactics, so the same basket size now carries a different fraud risk. Which kind of drift is this?

- **A.** Covariate drift
- **B.** Prior drift
- **C.** Concept drift
- **D.** Training-serving skew

<details>
<summary>Answer</summary>

**C.** Concept drift

$P(y \mid x)$ changed, so the same input now has a different correct answer.

- **A:** Covariate drift changes $P(x)$ while $P(y \mid x)$ stays.
- **B:** Prior drift changes $P(y)$ while $P(x \mid y)$ stays.
- **C:** Correct.
- **D:** Skew is a mismatch between how features are computed offline and online.

</details>

## 3. Calculation (medium)

One of ten decile bins holds 10% of the reference and 30% of the current window. What does this bin add to PSI? Give three decimals.

<details>
<summary>Answer</summary>

**0.22** (within ±0.002)

$(0.30 - 0.10)\ln(0.30/0.10) = 0.20 \times 1.0986 = 0.2197$.

</details>

## 4. Calculation (easy)

A monitor checks 40 features every hour at α = 0.05 each, and nothing has changed. About how many false alarms per day?

<details>
<summary>Answer</summary>

**48**

Per hour $40 \times 0.05 = 2$. Per day $2 \times 24 = 48$.

</details>

## 5. Calculation (medium)

A chi-square test on a two-row table with 400 total events gives $\chi^2 = 16$. What is Cramér's V?

<details>
<summary>Answer</summary>

**0.2** (within ±0.001)

$V = \sqrt{\chi^2 / N} = \sqrt{16 / 400} = \sqrt{0.04} = 0.2$.

</details>

## 6. Multiple choice (medium)

With one million events per window, a KS test reports p = 2e-7 on a feature, with D = 0.004. What should the policy do?

- **A.** Page the on-call engineer immediately
- **B.** Log at most a warning, because the effect size is tiny
- **C.** Retrain the model
- **D.** Disable the feature

<details>
<summary>Answer</summary>

**B.** Log at most a warning, because the effect size is tiny

Huge samples make trivial shifts significant. Require an effect-size threshold (for example D ≥ 0.1) as well as a small p-value.

- **A:** Paging on tiny, significant-but-irrelevant shifts causes alert fatigue.
- **B:** Correct.
- **C:** A 0.004 gap in CDFs is unlikely to affect the model.
- **D:** Nothing suggests the feature is broken.

</details>

## 7. Multiple choice (medium)

Input monitors are quiet, but accuracy on newly labelled events drops from 0.95 to 0.75. What is the right next step?

- **A.** Ignore it until input drift appears
- **B.** Treat it as likely concept drift, collect recent labels, investigate, and retrain on recent data
- **C.** Recalibrate thresholds only
- **D.** Widen the PSI bands

<details>
<summary>Answer</summary>

**B.** Treat it as likely concept drift, collect recent labels, investigate, and retrain on recent data

Unchanged inputs with falling accuracy point to $P(y \mid x)$ changing. Old labels teach the old concept, so retrain on fresh ones.

- **A:** Concept drift never shows up in input monitors.
- **B:** Correct.
- **C:** Recalibration fits prior drift, where the base rate moved but the input-label relationship did not.
- **D:** Input bands are irrelevant to a change only labels can show.

</details>

## 8. Multiple choice (medium)

Overnight, PSI on `transaction_amount` jumps to 4.8, and every value is now 100 times larger. What should you do first?

- **A.** Retrain on the new data immediately
- **B.** Check the upstream pipeline for a unit change (for example dollars to cents) and fix it
- **C.** Lower the PSI alert threshold
- **D.** Switch the model to a bigger one

<details>
<summary>Answer</summary>

**B.** Check the upstream pipeline for a unit change (for example dollars to cents) and fix it

A sudden, exact scale change is almost always a pipeline bug, not real-world drift. Retraining on it would bake the bug into the model.

- **A:** Retraining on broken data makes the bug permanent.
- **B:** Correct.
- **C:** The alert already fired; thresholds aren't the problem.
- **D:** Model capacity is unrelated to a unit error.

</details>

## 9. Calculation (hard)

Concept drift begins at event 6,500. Labels arrive 500 events after each prediction, windows are 500 events and move by 250, and an alert needs 2 consecutive drifting windows. In the demo, the first window whose labelled slice is mostly post-drift ends at 7,250. At which window end does the alert fire?

<details>
<summary>Answer</summary>

**7500**

The window ending at 7,250 is the first drifting one; persistence needs a second, which is the window ending $7{,}250 + 250 = 7{,}500$. That is 1,000 events after the change, 500 of them label delay.

</details>

## 10. Match (medium)

Match each finding to the most appropriate action.

| Concept | Options |
|---|---|
| Unit change in an upstream column | Collect recent labels and retrain |
| Covariate drift with stable outcomes | Recalibrate scores or thresholds |
| Base rate doubled; inputs per class unchanged | Keep watching; usually no action |
| Accuracy falls with normal-looking inputs | Fix the pipeline; do not retrain |

<details>
<summary>Answer</summary>

- Unit change in an upstream column → Fix the pipeline; do not retrain
- Covariate drift with stable outcomes → Keep watching; usually no action
- Base rate doubled; inputs per class unchanged → Recalibrate scores or thresholds
- Accuracy falls with normal-looking inputs → Collect recent labels and retrain

Different causes need different fixes; retraining is not the default answer to every alert.

</details>
