---
title: Model monitoring and drift
summary: Monitor a deployed model across system, input, prediction, and outcome layers, tell covariate, prior, and concept drift apart, compute PSI, the KS statistic, and chi-square from scratch, and design an alert policy that catches real drift without paging people for noise.
skill: engineering
minutes: 45
prerequisites: [ml-system-design, probability-and-statistics, classification-metrics]
related: [experimentation-and-ab-testing, reliability-cost-and-observability, fairness, ml-workflow]
adapted_from: [phases/02-ml-fundamentals/20-model-monitoring-and-drift]
---

# Model monitoring and drift

> **Mental model.** A deployed model is a map drawn from last year's survey. The roads change, the map doesn't, and the model keeps answering with full confidence. Software monitoring asks "is the service up?" Model monitoring asks "is the map still right?", and because the true answers (labels) arrive late, you mostly watch for signs that the territory has changed before you can prove the map is wrong.

**You will learn to**
- Choose signals across four layers: system health, inputs, predictions, and outcomes, and say when each becomes available.
- Distinguish covariate drift, prior drift, and concept drift by which part of the joint distribution changes, and say which monitor can see each.
- Compute the Population Stability Index, the two-sample Kolmogorov-Smirnov statistic, and a chi-square test with Cramér's V, by hand and in code.
- Design reference and current windows, and reason about window size, label delay, and seasonality.
- Build an alert policy with multiple-testing correction, effect-size thresholds, and persistence.
- Decide whether a given drift needs no action, recalibration, new labels, or retraining.

**Why it matters.** Models fail silently. A fraud model returns HTTP 200 with normal latency while fraudsters change tactics; a churn model keeps scoring after a pricing change. The first sign is often a business report weeks later. Monitoring is how you find out in days instead, and "how would you monitor this?" is a standard part of every [ML system design](01-ml-system-design.md) interview.

## 1. Intuition

**Four layers, three of them early.**

| Layer | Signals | Available | Question |
|---|---|---|---|
| System | latency, errors, throughput, memory | immediately | does the service answer? |
| Inputs | each feature's distribution, missing rate, schema, new categories | immediately | does live data look like training data? |
| Predictions | score distribution, positive rate, share of default answers | immediately | does the model behave as before? |
| Outcomes | accuracy, precision, calibration, business metric | after the label delay | is the model still right? |

Only the last layer measures correctness, and it arrives late: seconds for an ad click, weeks for a chargeback, months for a loan default. The first three are early warnings.

**Three kinds of [drift](../../glossary.md#drift).** Write the joint distribution of inputs $x$ and labels $y$ two ways: $P(x, y) = P(x)\,P(y \mid x) = P(y)\,P(x \mid y)$. Each kind of drift changes one factor:

| Drift | What changes | Payment example | Seen by |
|---|---|---|---|
| Covariate | $P(x)$; $P(y \mid x)$ stays | a campaign brings app users with bigger baskets | input monitors |
| Prior (label shift) | $P(y)$; $P(x \mid y)$ stays | the fraud rate doubles during an attack, but fraud looks the same | prediction and outcome monitors |
| Concept | $P(y \mid x)$ | the same basket is now riskier because fraudsters changed tactics | outcome monitors only |

Covariate drift does not always hurt: if the model learned $P(y \mid x)$ well, it stays right for each input. It hurts when traffic moves into regions with little training data. Concept drift is the dangerous one, because the inputs can look completely normal.

**Compare two windows.** Every drift test compares a **reference window** (usually the training data, or a validated stretch of production) with a **current window** (the last $N$ events or hours). Small windows react fast but mistake noise for drift; large windows are stable but slow. Seasonality breaks naive windows: Monday traffic is not Saturday traffic, so compare like with like or make the reference cover a full cycle.

**Three statistics.**
- **PSI** (Population Stability Index): cut the reference into bins (usually deciles), compare the share of current data in each bin, and sum a log-ratio term. Common reading: under 0.1 stable, 0.1 to 0.25 moderate shift, over 0.25 significant. These bands are a convention from credit scoring, not a test.
- **KS statistic** $D$: the largest vertical gap between the two empirical cumulative distribution curves. No bins. 0 means identical samples, 1 means no overlap.
- **Chi-square** for categorical features: compare category counts against what you'd expect if both windows had the same mix. **Cramér's V** turns it into an effect size between 0 and 1.

**Significance is not importance.** With a million events per window, a KS test calls a 0.02-standard-deviation shift "significant". With many features checked every hour, some tests fire by chance. A useful alert needs a real effect size, survives correction for the number of tests, and persists for more than one window.

> [!IMPORTANT]
> **Input monitors cannot see concept drift.** If the inputs look normal but the world's answer to them changed, only outcome data shows it. Budget for labels: fast proxies, a labelled sample, or human review, so you are not blind for the whole label delay.

## 2. Visualization

![Two stacked panels over a synthetic stream of 9,000 payment events, with shaded regions for covariate drift (events 3,000 to 5,000) and concept drift (6,500 to 9,000). Top: input monitors. PSI on amount jumps from about 0.02 to 0.75 to 0.93 during covariate drift, KS D to about 0.4, and Cramér's V on channel to about 0.2; all return to baseline by 5,500 and stay flat during concept drift. Bottom: accuracy on labelled events with a 500-event delay stays between 0.91 and 0.96 through covariate drift, then falls from 0.96 to about 0.73 starting around 7,250, crossing the alert line at 0.907.](../../figures/model-monitoring-and-drift.png)

*Synthetic stream. Covariate drift lights up the input monitors while accuracy barely moves. Concept drift leaves the inputs untouched and shows up only in accuracy, a window late because labels arrive 500 events after each prediction.*

**Try it** (run [`drift_monitor.py`](../../code/22-ml-system-design/drift_monitor.py), predict first, then check):
1. Change the window from 500 to 200 events. Predict whether you get false alarms in the clean stretch before event 3,000.
2. Set `patience=1` and `min_v=0`. Predict how many alerts fire in the clean stretch from 5,500 to 6,500.
3. Change `delay` from 500 to 1,500. Predict the event at which the accuracy alert fires.
4. Add a prior-drift segment: keep the amount distribution the same but double the share of fraud. Predict which monitors react.

## 3. The math

### Symbols

| Symbol | Meaning | Shape / range |
|---|---|---|
| $e_b$, $a_b$ | share of reference and current values in bin $b$ | $[0, 1]$, sum to 1 |
| $F_R(x)$, $F_C(x)$ | empirical CDFs of reference and current windows | $[0, 1]$ |
| $D$ | KS statistic | $[0, 1]$ |
| $O_{rc}$, $E_{rc}$ | observed and expected count in row $r$ (window), column $c$ (category) | counts |
| $N$ | total count in the contingency table | count |
| $k$ | number of categories | count |
| $V$ | Cramér's V | $[0, 1]$ |
| $\alpha$, $m$ | family false-alarm rate; number of tests | probability; count |

### PSI

```math
\text{PSI} = \sum_{b=1}^{B} (a_b - e_b)\,\ln\frac{a_b}{e_b}
```

Each term is non-negative, because $a_b - e_b$ and $\ln(a_b/e_b)$ always have the same sign. PSI equals the KL divergence in both directions added together. Clip zero shares to a small value (0.0001) so the log stays finite.

**Worked example.** Ten decile bins, so every reference share is 0.10. In the current window, bins 1 and 2 drop to 0.05 and bin 3 rises to 0.20; the rest stay at 0.10.

1. Bin 1: $(0.05 - 0.10)\ln(0.05/0.10) = (-0.05)(-0.6931) = 0.0347$.
2. Bin 2: the same, $0.0347$.
3. Bin 3: $(0.20 - 0.10)\ln(0.20/0.10) = (0.10)(0.6931) = 0.0693$.
4. Bins 4 to 10: $(0.10 - 0.10) \times \ln 1 = 0$.
5. PSI $= 0.0347 + 0.0347 + 0.0693 = 0.1386$: a moderate shift by the usual bands.

### KS statistic

```math
D = \max_x \left|F_R(x) - F_C(x)\right|
```

**Worked example.** Reference $\{1, 2, 3, 4\}$, current $\{3, 4, 5, 6\}$.

1. At $x = 2$: $F_R = 2/4 = 0.5$, $F_C = 0/4 = 0$. Gap $0.5$.
2. At $x = 4$: $F_R = 1.0$, $F_C = 2/4 = 0.5$. Gap $0.5$.
3. At $x = 3$: $F_R = 0.75$, $F_C = 0.25$. Gap $0.5$. No point does better, so $D = 0.5$.

The p-value depends on the effective sample size $nm/(n+m)$; with large windows, tiny $D$ values become "significant", so alert on $D$ itself as well.

### Chi-square and Cramér's V

```math
E_{rc} = \frac{(\text{row}_r \text{ total})(\text{column}_c \text{ total})}{N}, \qquad
\chi^2 = \sum_{r,c}\frac{(O_{rc} - E_{rc})^2}{E_{rc}}, \qquad
V = \sqrt{\frac{\chi^2}{N}} \;\text{(two rows)}
```

with $k - 1$ degrees of freedom.

**Worked example.** Channel mix (app, store, web): reference 100 events $= (35, 15, 50)$, current 100 events $= (60, 10, 30)$.

1. Column totals: $95, 25, 80$; $N = 200$; each row total 100.
2. Expected per row: $100 \times 95/200 = 47.5$, $100 \times 25/200 = 12.5$, $100 \times 80/200 = 40$.
3. Reference row: $\frac{(35-47.5)^2}{47.5} + \frac{(15-12.5)^2}{12.5} + \frac{(50-40)^2}{40} = \frac{156.25}{47.5} + \frac{6.25}{12.5} + \frac{100}{40} = 3.289 + 0.5 + 2.5 = 6.289$.
4. Current row: deviations have the same size and opposite sign, so also $6.289$.
5. $\chi^2 = 12.579$ with 2 degrees of freedom, $p = 0.0019$.
6. $V = \sqrt{12.579/200} = \sqrt{0.0629} = 0.251$: a real shift in mix, not just a significant one.

### False alarms

With $m$ independent tests at level $\alpha$ each and no real change, the expected number of false alarms is $m\alpha$ per round. 50 features checked hourly at $\alpha = 0.05$: $50 \times 0.05 = 2.5$ per hour, $2.5 \times 24 = 60$ per day. The Bonferroni correction tests each at $\alpha/m$ to keep the family rate near $\alpha$; the demo uses $0.01/2 = 0.005$ for its two tests.

## 4. Implementation

### Build it from scratch

The full script is [`code/22-ml-system-design/drift_monitor.py`](../../code/22-ml-system-design/drift_monitor.py). NumPy for the statistics, SciPy only for the chi-square and KS p-value distributions and for checking our numbers. The stream is synthetic: 9,000 payment events with covariate drift injected at event 3,000 (bigger amounts, more app traffic), a return to normal at 5,000, and concept drift at 6,500 (the fraud boundary on amount drops from 120 to 105).

**PSI on reference deciles.** Bin edges come from the reference only, so each reference bin holds 10%; `searchsorted` assigns current values to the same bins.

```python
def psi(reference, current, bins=10, eps=1e-4):
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1])      # reference deciles
    e = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=bins) / len(reference)
    a = np.bincount(np.searchsorted(edges, current, side="right"), minlength=bins) / len(current)
    e, a = np.maximum(e, eps), np.maximum(a, eps)                            # avoid log(0)
    return float(np.sum((a - e) * np.log(a / e)))
```

*Predict:* the current window has values far above the reference maximum. Which bin do they land in, and does PSI see how far out they are?

**KS and chi-square.** KS evaluates both empirical CDFs at every observed value and takes the largest gap. Chi-square builds the expected table from row and column totals.

```python
def ks_statistic(x, y):
    """Largest vertical gap between the two empirical CDFs, evaluated at every observed value."""
    grid = np.sort(np.concatenate([x, y]))
    fx = np.searchsorted(np.sort(x), grid, side="right") / len(x)
    fy = np.searchsorted(np.sort(y), grid, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))

def chi_square(ref_counts, cur_counts):
    table = np.array([ref_counts, cur_counts], dtype=float)                 # [2, k categories]
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    chi2 = float(((table - expected) ** 2 / expected).sum())
    dof = table.shape[1] - 1
    return chi2, float(stats.chi2.sf(chi2, dof)), float(np.sqrt(chi2 / table.sum()))  # Cramer's V for 2 rows
```

*Predict:* a brand-new category appears only in the current window. What happens to its expected count in the reference row, and to $\chi^2$?

**Persistence.** The policy counts consecutive drifting windows per signal and alerts only when the streak reaches `patience`; it reports "resolved" on the first clean window after an alert.

```python
class Policy:
    def __init__(self, n_tests, family_alpha=0.01, psi_drift=0.25, ks_drift=0.10, min_v=0.10, patience=2):
        self.alpha, self.psi_drift, self.ks_drift, self.min_v, self.patience = family_alpha / n_tests, psi_drift, ks_drift, min_v, patience
        self.streak = {}

    def update(self, name, drifting):
        prev = self.streak.get(name, 0)
        self.streak[name] = prev + 1 if drifting else 0
        if self.streak[name] == self.patience:
            return "ALERT"
        if prev >= self.patience and not drifting:
            return "resolved"
        return None
```

*Predict:* with a sliding window that moves 250 events and is 500 long, how many events after a sudden change can the first alert fire at the earliest?

**The monitoring loop.** For each window: the three statistics, accuracy on the slice whose labels have arrived, and a drift flag per signal that needs both significance and effect size.

```python
        known = slice(end - delay - window, end - delay)                     # labels only exist `delay` events back
        acc = float((pred[known] == label[known]).mean())
        flags = {"amount": p >= policy.psi_drift or (d >= policy.ks_drift and ks_p < policy.alpha),
                 "channel": chi_p < policy.alpha and v >= policy.min_v,
                 "accuracy": acc < ref_acc - tol}
```

*Predict:* at the window ending at event 3,250, the channel test has $p = 0.0011$, below the per-test $\alpha$ of 0.005, but $V = 0.074$. Does it count as drift?

### Use the library

`scipy.stats` provides the tests directly, and the script checks its own numbers against them:

```python
from scipy import stats
stats.ks_2samp(reference, current)              # D and an exact or asymptotic p-value
stats.chi2_contingency([ref_counts, cur_counts])  # chi2, p, dof, expected table
```

Monitoring tools package the statistics with storage, scheduling, and dashboards. Evidently picks a drift test per column by type and size; NannyML estimates performance before labels arrive from the model's own calibrated probabilities (which handles covariate shift but, by design, cannot see concept drift); TensorFlow Data Validation checks schema and distribution skew between training and serving. Cloud ML platforms offer the same as managed services. The from-scratch code is what each of them computes per feature; what they add is the plumbing.

### Run it

```bash
uv run code/22-ml-system-design/drift_monitor.py
```

Key lines of output:

```
PSI = 0.1386
model threshold 120.8, reference accuracy 0.957
  end    PSI   KS D   chi2 p     V    acc  events
 3250  0.180  0.187  1.1e-03 0.074  0.954
 3500  0.752  0.376  1.3e-21 0.196  0.956  ALERT amount
 3750  0.926  0.402  1.4e-16 0.171  0.942  ALERT channel
 5500  0.024  0.061  8.6e-01 0.011  0.912  resolved amount, resolved channel
 7500  0.023  0.039  9.7e-02 0.043  0.728  ALERT accuracy
scratch KS D = 0.3850;  scipy ks_2samp D = 0.3850
scratch chi2 = 56.08;  scipy chi2_contingency = 56.08
```

Read it in three parts. Covariate drift starts at 3,000; the amount alert fires at 3,500 and accuracy stays above 0.9 because the fraud rule did not change. Inputs return to normal at 5,000 and both alerts resolve at 5,500. Concept drift starts at 6,500; the input monitors stay quiet, and the accuracy alert fires at 7,500, 1,000 events later, of which 500 are the label delay.

## 5. Engineering

**Monitor what matters most.** You cannot alert on every feature. Rank features by importance to the model and by how likely they are to break (upstream joins, third-party sources), watch the top ones closely, and summarize the rest. Always watch the prediction distribution: it is one signal that integrates all inputs.

**Catch pipeline bugs first.** Most "drift" in practice is a broken pipeline: a unit change (dollars to cents), a column filled with defaults, a renamed category, a feature computed differently online than offline ([training-serving skew](../../glossary.md#training-serving-skew)). Schema checks, missing-value rates, and range checks catch these cheaply and should page faster than statistical drift.

**Get labels sooner.** Use fast proxies (early chargebacks, first-week engagement), label a random sample by hand, or join delayed labels as soon as they land and compute outcome metrics on that slice. Track calibration as well as accuracy.

**Decide the action by drift type.**

| Finding | Action |
|---|---|
| Pipeline bug | fix the pipeline; do not retrain on broken data |
| Covariate drift, outcomes stable | usually none; keep watching outcomes |
| Covariate drift into sparse regions | retrain with recent data covering those regions |
| Prior drift | recalibrate scores or thresholds to the new base rate |
| Concept drift | collect recent labels and retrain; old labels teach the old concept |

**Retraining triggers.** Schedules are simple but retrain when nothing changed and wait when something did. Drift- or performance-triggered retraining reacts to evidence. Either way, validate the new model on the most recent labelled data, roll it out through shadow and canary stages, and reset the reference window afterwards.

**Feedback loops.** When the model's decisions control which labels you see (rejected loans never default, blocked transactions never charge back), outcome metrics are biased. Keep a small randomized holdout, as in [experimentation](02-experimentation-and-ab-testing.md), to measure the truth.

**Per-group monitoring.** Aggregate metrics can be stable while one segment degrades. Track key metrics per region, platform, and, where relevant, per protected group (see [Fairness](../21-safety-security/02-fairness.md)).

> [!WARNING]
> **Failure modes.** Alerting on p-values alone with large windows; dozens of uncorrected tests and an ignored alert channel; no outcome monitoring, so concept drift goes unseen; a reference window that is itself unrepresentative; seasonality read as drift; retraining on data from a broken pipeline; and dashboards nobody owns.

### Common mistakes

- Monitoring only latency and errors, and calling the model "healthy".
- Treating PSI bands as a statistical test.
- Using a reference window from a different season, then chasing weekly cycles.
- Retraining automatically on any drift, including drift caused by an upstream bug.
- Measuring accuracy on all predictions, including those whose labels haven't arrived.
- Forgetting to reset the reference after retraining, so every later window looks drifted.

## 6. Knowledge check

<!-- quiz:model-monitoring-and-drift -->
**[Take the model monitoring and drift quiz](../../quizzes/model-monitoring-and-drift.md)**
<!-- /quiz -->

**Practice exercise.** A feature is binned into five reference quintiles (0.20 each). The current window's shares are $(0.10, 0.15, 0.20, 0.25, 0.30)$. Compute PSI and classify it with the usual bands.

<details>
<summary>Solution</summary>

1. $(0.10-0.20)\ln(0.10/0.20) = (-0.10)(-0.6931) = 0.0693$.
2. $(0.15-0.20)\ln(0.75) = (-0.05)(-0.2877) = 0.0144$.
3. $(0.20-0.20)\ln 1 = 0$.
4. $(0.25-0.20)\ln(1.25) = (0.05)(0.2231) = 0.0112$.
5. $(0.30-0.20)\ln(1.5) = (0.10)(0.4055) = 0.0405$.

PSI $= 0.0693 + 0.0144 + 0 + 0.0112 + 0.0405 = 0.1354$: a moderate shift (between 0.1 and 0.25). The mass has moved toward higher values.

</details>

**Implementation challenge.** Add a prediction monitor to [`drift_monitor.py`](../../code/22-ml-system-design/drift_monitor.py): track the model's positive-prediction rate per window and alert when it moves more than 5 percentage points from the reference rate for two windows in a row. Which drift period does it catch, and why does it miss the other?

<details>
<summary>Solution sketch</summary>

Compute `ref_pos = pred[:ref_size].mean()` and, per window, `pos = pred[cur].mean()`; flag `abs(pos - ref_pos) > 0.05` and pass it through `policy.update("positive_rate", ...)`. It fires during covariate drift, because bigger amounts push more predictions over the fixed threshold. It stays quiet during concept drift, because the model's inputs (and therefore its predictions) are unchanged; only the true labels moved. Prediction monitors integrate input drift; they cannot see concept drift any more than input monitors can.

</details>

## Summary

- Monitor four layers: system, inputs, predictions (all immediate), and outcomes (delayed by labels).
- Covariate drift changes $P(x)$, prior drift $P(y)$, concept drift $P(y \mid x)$; only outcome monitoring sees concept drift.
- PSI compares binned shares, KS takes the largest CDF gap, chi-square with Cramér's V handles categories; use effect sizes, not just p-values.
- Alert policies need multiple-testing correction, effect-size thresholds, and persistence, or people stop reading them.
- Match the action to the cause: fix pipelines, recalibrate for prior drift, retrain with fresh labels for concept drift, and reset the reference after retraining.

**Next:** [The 12-week plan](../23-projects/01-twelve-week-plan.md)

**Related:** [ML system design](01-ml-system-design.md) · [Experimentation and A/B testing](02-experimentation-and-ab-testing.md) · [Reliability, cost, and observability](../20-production-ai/02-reliability-cost-and-observability.md) · [Fairness](../21-safety-security/02-fairness.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Monitoring Models in Production — Data Drift and Concept Drift](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/20-model-monitoring-and-drift).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>How would you monitor a fraud model after deployment?</strong></summary>

In four layers. System health: latency, error rate, throughput, so the service answers. Inputs: for the most important features, PSI or KS against the training window, plus missing rates, schema checks, and new categories, because most incidents are pipeline bugs. Predictions: the score distribution and the share of transactions flagged, which reacts to almost any input change. Outcomes: precision, recall, and calibration on transactions whose chargebacks have arrived, which can take weeks, so I'd add faster proxies like early disputes and a manually reviewed sample. Alerts use Bonferroni-corrected thresholds, require an effect size, and need two consecutive windows. Because the model blocks transactions, blocked ones never charge back, so I'd keep a tiny randomized holdout to measure true fraud rates. And I'd break metrics down by region and channel.

</details>

<details>
<summary><strong>What's the difference between covariate drift and concept drift, and why does it matter for what you do next?</strong></summary>

Covariate drift is a change in the input distribution, P of x, while the relationship between inputs and the label stays the same. Concept drift is a change in that relationship, P of y given x: the same input now has a different correct answer. It matters because the monitors and the fixes differ. Input monitors like PSI catch covariate drift immediately, and often no action is needed if the model was right across those inputs; you retrain only if traffic moved into regions with little training data. Concept drift leaves the inputs looking normal, so only outcome metrics reveal it, after the label delay. Retraining on old labels doesn't help, because they encode the old concept; you need fresh labels. A third case, prior drift, where the base rate changes, is often fixed by recalibrating thresholds.

</details>

<details>
<summary><strong>Your drift dashboard pages the team several times a day and people have started ignoring it. What do you change?</strong></summary>

First I'd count where the pages come from. Usually it's three things. Too many uncorrected tests: 50 features checked hourly at 0.05 gives about 60 false alarms a day with no real change, so I'd apply a Bonferroni or false-discovery-rate correction. P-values without effect sizes: with big windows, tiny shifts are significant, so I'd require PSI above 0.25, KS D above 0.1, or Cramér's V above 0.1 as well. No persistence: one noisy window shouldn't page anyone, so I'd require two or three in a row. Then I'd check seasonality, comparing Monday with last Monday. Finally, split warnings from alerts: warnings go to a dashboard, and only drift on important features, schema breaks, or outcome drops page a person, each with an owner and a runbook.

</details>

<details>
<summary><strong>Ten reference deciles; in the current window, two bins drop to 5% and one rises to 20%. What's the PSI, and is it worth an alert?</strong></summary>

PSI sums current share minus reference share, times the log of their ratio, over bins. The two bins at 5% each contribute minus 0.05 times ln 0.5, which is 0.05 times 0.693, about 0.035. The bin at 20% contributes 0.10 times ln 2, about 0.069. The unchanged bins add zero. Total: 0.035 plus 0.035 plus 0.069, about 0.139. By the usual credit-scoring bands that's a moderate shift: above 0.1, below 0.25. I wouldn't page on it alone. I'd log a warning, check whether it persists over the next window or two, look at whether the feature matters to the model and whether the prediction distribution moved, and check outcome metrics as labels arrive. If it persists and the feature is important, it's worth investigating.

</details>
