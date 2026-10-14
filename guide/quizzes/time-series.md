<!-- GENERATED from time-series.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Time series

Covers the lesson [Time series](../lessons/24-applied-ml/04-time-series.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/time-series/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

Why is a random train/test split invalid for time-series forecasting?

- **A.** It makes the test set too small
- **B.** It puts future rows in training, so the model can use information it won't have in production
- **C.** Random splits only work for classification
- **D.** It changes the series' mean

<details>
<summary>Answer</summary>

**B.** It puts future rows in training, so the model can use information it won't have in production

Forecasts are made from the past only. Random splits let the model interpolate between neighbors in time, which inflates scores.

- **A:** Split size is a separate choice.
- **B:** Correct.
- **C:** Random splits are wrong for time-ordered regression too.
- **D:** The mean of the whole series doesn't change; the issue is leakage across time.

</details>

## 2. Multiple choice (easy)

What does it mean for a series to be (weakly) stationary?

- **A.** Its values never change
- **B.** Its mean, variance, and autocorrelation do not change over time
- **C.** It has no seasonality but may have a trend
- **D.** It has been scaled to mean 0 and standard deviation 1

<details>
<summary>Answer</summary>

**B.** Its mean, variance, and autocorrelation do not change over time

Stationarity is about stable statistical properties, not constant values.

- **A:** Stationary series fluctuate; only their statistics stay stable.
- **B:** Correct.
- **C:** A trend changes the mean, so a trending series is not stationary.
- **D:** Standardizing the whole series doesn't remove a trend.

</details>

## 3. Multiple choice (easy)

What is the main purpose of differencing a series?

- **A.** To remove trend (and, at lag m, seasonality) so the series is closer to stationary
- **B.** To smooth out noise
- **C.** To fill in missing values
- **D.** To convert it into a classification problem

<details>
<summary>Answer</summary>

**A.** To remove trend (and, at lag m, seasonality) so the series is closer to stationary

Modeling the change y_t − y_{t−1} removes a linear trend; y_t − y_{t−m} removes a period-m seasonal pattern.

- **A:** Correct.
- **B:** Differencing amplifies noise (variance doubles for white noise); smoothing is a rolling mean.
- **C:** Imputation is a separate step.
- **D:** The target stays numeric.

</details>

## 4. Multiple choice (easy)

When predicting $y_t$, what is the lag-3 feature?

- **A.** $y_{t+3}$
- **B.** $y_{t-3}$
- **C.** The mean of $y_{t-3}, \dots, y_t$
- **D.** $y_t - y_{t-3}$

<details>
<summary>Answer</summary>

**B.** $y_{t-3}$

A lag-k feature is the value k steps in the past.

- **A:** That is the future; using it is leakage.
- **B:** Correct.
- **C:** That window includes y_t itself, which leaks the target.
- **D:** That is a lag-3 difference, not a lag feature.

</details>

## 5. Calculation (medium)

Actuals for 4 days are $[25, 24, 26, 28]$ and the naive forecasts (previous day's value) are $[23, 25, 24, 26]$. What is the MAE?

<details>
<summary>Answer</summary>

**1.75** (within ±0.001)

Absolute errors 2, 1, 2, 2 sum to 7; 7/4 = 1.75.

</details>

## 6. Calculation (medium)

A model's MAE is 3.6. The in-sample MAE of the seasonal-naive forecast is 4.5. What is the MASE?

<details>
<summary>Answer</summary>

**0.8** (within ±0.001)

MASE = 3.6 / 4.5 = 0.8, so the model has 20% less error than seasonal naive.

</details>

## 7. Multiple choice (hard)

A teammate builds `df['roll7'] = df['sales'].rolling(7).mean()` and uses it to predict `df['sales']` on the same row. Walk-forward MAE is suspiciously low. What is wrong?

- **A.** Nothing; rolling(7) only looks backward
- **B.** The window includes the current row, so the target is one seventh of its own feature; add .shift(1)
- **C.** Rolling means should use 30 days
- **D.** Walk-forward validation can't detect this, so switch to K-fold

<details>
<summary>Answer</summary>

**B.** The window includes the current row, so the target is one seventh of its own feature; add .shift(1)

pandas rolling windows end at and include the current row. shift(1) moves the window so row t only sees t−7..t−1.

- **A:** It looks backward but includes today, which is the value being predicted.
- **B:** Correct.
- **C:** Window length doesn't fix the inclusion of y_t.
- **D:** K-fold would hide the problem even more; the bug is in the feature.

</details>

## 8. Multiple choice (hard)

After seasonal differencing at lag 7, the ACF is near zero everywhere except about −0.5 at lag 7. What does that indicate?

- **A.** The differencing failed and the seasonality is still there
- **B.** Expected; differencing white noise e_t − e_{t−7} gives autocorrelation −0.5 at lag 7, so the seasonal pattern is gone
- **C.** The series has a trend
- **D.** The model must use lag 7 as a feature

<details>
<summary>Answer</summary>

**B.** Expected; differencing white noise e_t − e_{t−7} gives autocorrelation −0.5 at lag 7, so the seasonal pattern is gone

Var(e_t − e_{t−7}) = 2σ², Cov at lag 7 = −σ², ratio −0.5. It's a sign of (slight) over-differencing, not leftover seasonality, which would show as a positive spike.

- **A:** Leftover seasonality appears as a positive spike.
- **B:** Correct.
- **C:** A trend shows as slowly decaying positive autocorrelation at all lags.
- **D:** It describes the differenced noise, not a required feature.

</details>

## 9. Multiple choice (medium)

You must forecast daily demand 14 days ahead for 20,000 store-product series with promotions and holidays. What is a strong first approach?

- **A.** One ARIMA model per series with default settings
- **B.** One global gradient-boosting model on lags of at least 14 days, rolling stats, calendar and promotion features, validated walk-forward against seasonal naive
- **C.** An LSTM per series
- **D.** A random 80/20 split and the model with best accuracy

<details>
<summary>Answer</summary>

**B.** One global gradient-boosting model on lags of at least 14 days, rolling stats, calendar and promotion features, validated walk-forward against seasonal naive

A global model shares patterns across series and handles exogenous features; lags of 14 or more are available at a 14-day horizon (or use a direct/recursive strategy).

- **A:** Workable for a few series, but 20,000 local models can't share information and handle promotions poorly.
- **B:** Correct.
- **C:** Thousands of tiny neural models overfit and are costly to maintain.
- **D:** A random split leaks future data, and accuracy isn't a forecasting metric.

</details>

## 10. Reflection (medium)

Explain the difference between an expanding and a sliding training window in walk-forward validation, and when you'd pick each.

<details>
<summary>Answer</summary>

**Model answer.** An expanding window trains each fold on all data before the test block, so training grows over time; use it when the process is stable and more history helps. A sliding window keeps a fixed-length recent window; use it when the process drifts (pricing changes, new behavior), so old data would mislead. Compare both by walk-forward error.

The trade-off is more data versus more relevant data.

</details>
