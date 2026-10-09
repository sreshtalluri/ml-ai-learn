---
title: Experimentation and A/B testing
summary: Design an A/B test that can actually detect the effect you care about, read its result correctly, and avoid the traps (peeking, sample ratio mismatch, novelty, interference) that make online experiments lie.
skill: engineering
minutes: 45
prerequisites: [probability-and-statistics, ml-system-design]
related: [llm-evaluation, classification-metrics, reliability-cost-and-observability]
---

# Experimentation and A/B testing

> **Mental model.** An A/B test is a fair coin flip that decides who gets the new version, followed by a check of whether the difference you see is bigger than what the coin flip alone would produce. The coin flip removes every other explanation; the statistics tell you how big a difference luck can fake.

**You will learn to**
- Explain a [p-value](../../glossary.md#p-value), type I and type II errors, and [statistical power](../../glossary.md#statistical-power) without the common misreadings.
- Run a two-proportion z-test and compute the sample size and test length for a given [minimum detectable effect](../../glossary.md#minimum-detectable-effect).
- Show why peeking at a running test inflates false positives, and name the fixes.
- Diagnose [sample ratio mismatch](../../glossary.md#sample-ratio-mismatch), novelty and primacy effects, and interference between users.
- Use [CUPED](../../glossary.md#cuped) to cut variance, and decide how offline LLM evals and online A/B tests fit together.

**Why it matters.** Offline metrics decide which models are worth testing; online experiments decide what ships. Every ML system design answer ends with "and then we A/B test it", and interviewers follow up with "how many users, for how long, and what would make you distrust the result?" Getting this wrong ships regressions or kills real improvements.

## 1. Intuition

**Randomize, then compare.** Split eligible users at random into control (A, the current system) and treatment (B, the new model). Because the split is random, the only systematic difference between the groups is the change itself, so a difference in outcomes is evidence the change caused it, which is something no offline analysis of logs can establish on its own.

**Noise can fake a difference.** Even if B does nothing (an "A/A test"), two random halves of users will never have exactly the same conversion rate. The question is whether the observed gap is larger than what random assignment typically produces.

**Two ways to be wrong.**

| | B really has no effect | B really helps |
|---|---|---|
| Test says "significant" | **Type I error** (false positive), probability $\alpha$ | correct, probability = power $1 - \beta$ |
| Test says "not significant" | correct | **Type II error** (false negative), probability $\beta$ |

Convention: $\alpha = 0.05$ and power $= 0.8$. A test with low power is worse than it sounds. It usually misses real effects, and when it does reach significance, it overstates the effect size.

**The p-value.** If B truly had no effect, the p-value is the probability of seeing a difference at least as extreme as the one observed. It is *not* the probability that B has no effect, and $1 - p$ is not the probability that B works.

**Minimum detectable effect (MDE).** The smallest true effect your test is designed to detect with the chosen power. Smaller MDE means more users: halving the MDE roughly quadruples the sample size, because noise shrinks only with $\sqrt{n}$.

**Traps.**

- **Peeking (optional stopping).** Checking the p-value every day and stopping the first time it dips below 0.05 gives noise many chances to cross the line. With 14 daily looks the false-positive rate is about four times higher than the nominal 5% (section 2).
- **Novelty and primacy effects.** Users click a new design because it is new (novelty: the effect fades), or resist it because they are used to the old one (primacy: the effect grows). Look at the effect over time and for new versus returning users; run long enough to see it stabilize.
- **Interference.** The standard analysis assumes one user's treatment does not affect another's outcome. Marketplaces (riders and drivers share a pool), social networks (a feature changes what your friends see), and shared budgets (ads) break this. Randomize by cluster (city, friend group) or by time (switchback tests).
- **Sample ratio mismatch (SRM).** If you designed a 50/50 split and got 50.6/49.4 on 100,000 users, something in assignment or logging is broken, and the result cannot be trusted until you find it.

## 2. Visualization

<!-- lab:ab-test -->
![Left: power rises in an S-curve from near 0 at zero lift to near 1 at 20% relative lift for 14,749 users per arm, crossing 0.8 exactly at the 10% MDE; a dashed curve for half that sample size reaches only about 0.5 at 10%. Right: in synthetic A/A tests, the false-positive rate when stopping at the first daily p below 0.05 climbs from about 0.05 with one look to about 0.22 with 14 daily looks, far above the dashed nominal 0.05 line.](../../figures/experimentation-and-ab-testing.png)

*Left: the power curve for the worked example (baseline 10%, α = 0.05). Right: 4,000 synthetic A/A tests with 1,500 users per arm per day; one look at day 14 gives a false-positive rate of 0.051, stopping at the first significant daily look gives 0.216.*

*Interactive version: [open the lab on the website](https://sreshtalluri.github.io/ml-ai-learn/labs/ab-test/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Halve the MDE from 10% to 5%. Predict the new sample size before you look (hint: noise shrinks with $\sqrt{n}$).
2. Click **Rare event (1%)**. How does a lower baseline rate change the users needed for the same relative lift?
3. Drag the simulation length from 14 to 30 days. Does the peeking false-positive rate level off or keep climbing?
4. Click **New seed** a few times. How much does the one-look false-positive rate wander around 5% with 2,000 simulated tests?

## 3. The math

### Symbols

| Symbol | Meaning | Shape / unit |
|---|---|---|
| $p_A, p_B$ | true conversion rates of control and treatment | scalars in $[0, 1]$ |
| $\hat{p}_A, \hat{p}_B$ | observed rates | scalars |
| $n$ | users per arm | count |
| $\delta$ | absolute difference $p_B - p_A$ | scalar |
| $\alpha$ | false-positive rate (significance level), two-sided | scalar |
| $1 - \beta$ | power | scalar |
| $z_q$ | standard normal quantile: $\Phi(z_q) = q$ | scalar |
| $\rho$ | correlation between a pre-experiment covariate and the outcome | scalar |

### Two-proportion z-test

```math
\hat{p} = \frac{x_A + x_B}{n_A + n_B}, \qquad
\text{SE} = \sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}, \qquad
z = \frac{\hat{p}_B - \hat{p}_A}{\text{SE}}, \qquad
p\text{-value} = 2\,\big(1 - \Phi(|z|)\big)
```

where $x$ are conversion counts and $\hat{p}$ is the pooled rate (the best estimate if there were no difference).

### Sample size per arm

```math
n = \frac{\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2 \left[p_A(1-p_A) + p_B(1-p_B)\right]}{\delta^2}
```

The numerator's first factor is how many standard errors the true effect must span (enough to clear the significance threshold and also land above it with probability $1 - \beta$). The second is the variance of the difference for one user per arm. A handy shortcut for $\alpha = 0.05$ and power 0.8 is $n \approx 16\,p(1-p)/\delta^2$, since $2 \times 2.8016^2 = 15.7 \approx 16$.

### Power at a given n

```math
1 - \beta \approx \Phi\!\left( \frac{|\delta|}{\sqrt{\left[p_A(1-p_A) + p_B(1-p_B)\right]/n}} - z_{1-\alpha/2} \right)
```

### Worked example 1: how many users?

Baseline $p_A = 0.10$, relative MDE 10%, so $p_B = 0.10 \times 1.10 = 0.11$ and $\delta = 0.01$. Use $\alpha = 0.05$ (two-sided) and power 0.8.

1. Quantiles: $z_{0.975} = 1.9600$, $z_{0.80} = 0.8416$. Sum: $1.9600 + 0.8416 = 2.8016$. Squared: $2.8016^2 = 7.8489$.
2. Variance term: $0.10 \times 0.90 + 0.11 \times 0.89 = 0.0900 + 0.0979 = 0.1879$.
3. Effect squared: $\delta^2 = 0.01^2 = 0.0001$.
4. $n = 7.8489 \times 0.1879 / 0.0001 = 1.47480 / 0.0001 = 14{,}748.04$, rounded up to **14,749 per arm** (29,498 users in total).
5. Test length: with 3,000 eligible users per day split evenly, each arm gets 1,500 per day: $14{,}749 / 1{,}500 = 9.83$, so 10 days. Round up to **14 days** so both arms cover two full weekly cycles.

Shortcut check: $16 \times 0.09 / 0.0001 = 14{,}400$, within 3%.

How fast this grows: a 5% relative MDE on the same baseline needs 57,760 per arm; on a 5% baseline it needs 122,121 per arm.

### Worked example 2: reading the result

After the test, control converts 1,500 of 15,000 users ($\hat{p}_A = 0.100$) and treatment 1,650 of 15,000 ($\hat{p}_B = 0.110$). These are synthetic.

1. Pooled rate: $\hat{p} = (1500 + 1650) / 30000 = 3150 / 30000 = 0.105$.
2. $\hat{p}(1 - \hat{p}) = 0.105 \times 0.895 = 0.093975$; $1/n_A + 1/n_B = 2 / 15000 = 0.00013333$.
3. $\text{SE} = \sqrt{0.093975 \times 0.00013333} = \sqrt{0.00001253} = 0.003540$.
4. $z = (0.110 - 0.100) / 0.003540 = 0.010 / 0.003540 = 2.825$.
5. p-value $= 2(1 - \Phi(2.825)) = 2 \times 0.00236 = 0.0047$.
6. 95% confidence interval for $\delta$ (unpooled SE $\sqrt{0.0900/15000 + 0.0979/15000} = 0.00354$): $0.010 \pm 1.96 \times 0.00354 = 0.010 \pm 0.0069$, i.e. $[0.0031, 0.0169]$, a relative lift between about 3% and 17%.

Report the interval, not just "significant". The interval tells the product team the effect could be small.

### Worked example 3: sample ratio mismatch

A 50/50 test assigns 100,000 users: 50,600 to A and 49,400 to B. Expected 50,000 each. Chi-square with one degree of freedom:

```math
\chi^2 = \frac{(50600 - 50000)^2}{50000} + \frac{(49400 - 50000)^2}{50000} = \frac{360000}{50000} + \frac{360000}{50000} = 7.2 + 7.2 = 14.4
```

$\sqrt{14.4} = 3.79$ standard deviations, p-value $\approx 0.00015$. A 1.2-point imbalance looks harmless, but at this sample size it is almost impossible by chance. Typical causes: the treatment crashes or loads slower so fewer of its users get logged, bot filtering that treats the arms differently, or a redirect that drops users in one arm.

### Worked example 4: CUPED

CUPED (controlled-experiment using pre-experiment data) subtracts the part of each user's outcome that was predictable from before the experiment:

```math
Y^{\text{cuped}} = Y - \theta\,(X - \bar{X}), \qquad \theta = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)}, \qquad \operatorname{Var}(Y^{\text{cuped}}) = (1 - \rho^2)\operatorname{Var}(Y)
```

where $X$ is the same metric measured in the weeks before the test. Because $X$ was fixed before randomization, it is independent of treatment and the estimate stays unbiased. With $\rho = 0.5$: variance falls by $\rho^2 = 0.25$ to 75%, so the required sample drops from 14,749 to $0.75 \times 14{,}749 = 11{,}061.75$, about 11,062 per arm. Equivalently, the same test finishes 25% sooner.

### Worked example 5: what peeking costs

Seeded simulation of 4,000 A/A tests (synthetic, 1,500 users per arm per day, 14 days, true rate 10% in both arms):

| Decision rule | False-positive rate |
|---|---|
| One look at day 14 | 0.051 |
| Stop at the first of 14 daily looks with p < 0.05 | 0.216 |

The 14 looks are highly correlated (day 9's data contains day 8's), so the inflation is less than the $1 - 0.95^{14} = 0.51$ you would get from 14 independent tests, but still more than four times the nominal rate. Fixes: fix the sample size in advance and look once; use a group-sequential design with a spending function (O'Brien-Fleming style, which spends little $\alpha$ early); or use always-valid sequential methods (mixture sequential probability ratio tests, confidence sequences) that are built to be monitored continuously.

## 4. Implementation

```python
from statistics import NormalDist
import math

N = NormalDist()

def sample_size(p1, rel_mde, alpha=0.05, power=0.8):
    p2 = p1 * (1 + rel_mde)
    z = N.inv_cdf(1 - alpha / 2) + N.inv_cdf(power)
    return math.ceil(z**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2)

def z_test(x_a, n_a, x_b, n_b):
    p = (x_a + x_b) / (n_a + n_b)
    se = math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    z = (x_b / n_b - x_a / n_a) / se
    return z, 2 * (1 - N.cdf(abs(z)))

sample_size(0.10, 0.10)          # 14749
z_test(1500, 15000, 1650, 15000) # (2.825, 0.0047)
```

In production you would use `statsmodels` (`proportions_ztest`, `NormalIndPower`) or your experimentation platform, but write the formula once by hand so you can sanity-check what the platform tells you.

Runnable script (sample size, z-test, SRM, CUPED, the seeded A/A peeking simulation, and the figure): [`code/22-ml-system-design/ab_test.py`](../../code/22-ml-system-design/ab_test.py).

## 5. Engineering

**Choose the metric before the test.** Write down one primary metric, its MDE, the guardrails (latency, errors, revenue, complaint rate), the sample size, and the duration before launch. Changing the primary metric after seeing results is p-hacking. With many metrics, expect about 1 in 20 to be "significant" by chance; correct for multiple comparisons or treat secondary metrics as exploratory.

**Randomization unit.** Randomize by user (stable hashing of user ID plus an experiment salt), not by request. Otherwise one user sees both versions and the outcomes are not independent. When the metric is per click but the unit is the user (CTR = clicks / impressions), the naive binomial variance is too small; use the delta method or bootstrap by user.

**A/A tests and SRM checks.** Run A/A tests on the platform regularly. Their false-positive rate should be close to $\alpha$. Check SRM automatically on every experiment and block the readout if it fails.

**Duration.** At least one full week (weekday versus weekend behavior), longer if novelty or primacy effects are plausible. Plot the daily treatment effect. A shrinking effect suggests novelty, and a growing one suggests primacy or learning.

**Interference.** For marketplaces, ads budgets, and social features, use cluster randomization (by city or by friend cluster), switchback designs (alternate the whole market between A and B over time windows), or budget-split designs. They cost power but give unbiased answers.

**Interleaving for ranking.** For search and recommendation rankers, interleaving (merging both rankers' results into one list and crediting clicks to whichever ranker contributed the item) detects preference with far fewer users than an A/B test. Use it to screen candidates, then confirm the winner with an A/B test on business metrics.

**LLM features: offline evals versus online A/B.** Offline evaluation (golden test sets, rubric-based [LLM evaluation](../17-llm-evaluation/01-llm-evaluation.md) with an LLM judge, safety suites) is fast and catches regressions before any user sees them, but it measures a proxy on a fixed dataset. An online A/B test measures what users actually do: task completion, retries, thumbs down, escalation to a human, retention, plus cost and latency per session as guardrails. Use offline evals as the gate and A/B tests as the verdict. LLM-specific wrinkles: outputs are stochastic, so per-user variance is higher; a prompt or model change can shift cost per request, which must be a guardrail; and shared caches (prefix or response caches) can leak treatment effects across arms, so make sure cache keys include the experiment arm.

> [!WARNING]
> **Failure modes.** Peeking and stopping early; underpowered tests whose "wins" are inflated by chance (the winner's curse); SRM ignored; a novelty spike shipped as a permanent gain; interference between arms; a bug that changes something besides the intended treatment, such as latency.

### Common mistakes

- Reading the p-value as the probability that the treatment works.
- Declaring "no effect" from a non-significant, underpowered test (absence of evidence is not evidence of absence).
- Stopping as soon as the dashboard turns green.
- Randomizing by request or session when the metric is per user.
- Reporting relative lift without the confidence interval.
- Testing 20 metrics and celebrating the one that hit p < 0.05.

## 6. Knowledge check

<!-- quiz:experimentation-and-ab-testing -->
**[Take the experimentation and A/B testing quiz](../../quizzes/experimentation-and-ab-testing.md)**
<!-- /quiz -->

**Practice exercise.** Your sign-up page converts at 20%. You want to detect a 10% relative lift ($\alpha = 0.05$, power 0.8). How many users per arm do you need, and how many days at 2,000 eligible users per day?

<details>
<summary>Solution</summary>

$p_A = 0.20$, $p_B = 0.22$, $\delta = 0.02$, $\delta^2 = 0.0004$. Variance term: $0.20 \times 0.80 + 0.22 \times 0.78 = 0.16 + 0.1716 = 0.3316$. $n = 7.8489 \times 0.3316 / 0.0004 = 2.60270 / 0.0004 = 6{,}506.8$, so 6,507 per arm. At 1,000 per arm per day: $6507 / 1000 = 6.5$, so 7 days, which is also exactly one full week. A higher baseline needs fewer users for the same relative lift because the absolute difference is larger.
</details>

**Implementation challenge.** Extend `ab_test.py` with a group-sequential rule: look on days 7 and 14 only, using a two-sided threshold of $|z| > 2.797$ at day 7 and $|z| > 1.977$ at day 14 (O'Brien-Fleming boundaries for two looks at overall $\alpha = 0.05$). Measure the false-positive rate on the same A/A simulation.

<details>
<summary>Solution sketch</summary>

Reuse the cumulative z-statistics from `aa_false_positive_rate`; mark a test significant if $|z_7| > 2.797$ or $|z_{14}| > 1.977$. The false-positive rate should come out close to 0.05, because the strict early boundary spends almost none of the error budget at day 7. Compare with naive two looks at 1.96 each, which lands noticeably above 0.05.
</details>

## Summary

- Randomization makes the treatment the only systematic difference between arms; the statistics measure how much difference luck alone can produce.
- The p-value is $P(\text{data this extreme} \mid \text{no effect})$, not the probability the change works. Power is the chance of detecting a real effect of size MDE.
- Sample size per arm is $(z_{1-\alpha/2} + z_{1-\beta})^2 [p_A(1-p_A) + p_B(1-p_B)] / \delta^2$: 14,749 per arm for 10% to 11%.
- Peeking inflates false positives (about 22% with 14 daily looks in the simulation); fix the horizon or use sequential methods.
- Check SRM, watch for novelty, primacy, and interference, and use CUPED to cut variance.
- Offline evals gate; online A/B tests decide.

**Next:** [Twelve-week plan](../23-projects/01-twelve-week-plan.md)

**Related:** [ML system design](01-ml-system-design.md) · [Probability and statistics](../00-foundations/03-probability-and-statistics.md) · [Evaluating LLM systems](../17-llm-evaluation/01-llm-evaluation.md) · [Cheat sheet: ML system design](../../cheatsheets/ml-system-design.md)

## Interview angle

<details>
<summary><strong>What does a p-value of 0.03 actually mean, and what doesn't it mean?</strong></summary>

It means that if the change truly had no effect, you would see a difference at least this large about 3% of the time from random assignment alone. It is a statement about the data given the null hypothesis. It is not the probability that the null is true, and $1 - 0.03$ is not a 97% chance the feature works. That probability depends on how often your ideas work in the first place. If only 10% of experiments have a real effect, a fair share of "significant" results are false positives, especially in underpowered tests. It also says nothing about size. A tiny, commercially useless lift can have $p = 0.001$ with enough users. So report the effect with a confidence interval, and only trust a p-value if the sample size and stopping rule were fixed in advance.

</details>

<details>
<summary><strong>When would you randomize by cluster or by time window instead of by user?</strong></summary>

When users interfere with each other, so one user's treatment changes another user's outcome. In a ride-hailing marketplace, a pricing change for treated riders shifts driver supply for control riders. In a social network, a new sharing feature changes what control users' friends post. Ad experiments that share a budget have the same problem. User-level randomization then biases the estimate, often overstating the effect because treatment borrows from control. Cluster randomization assigns whole cities or friend clusters, and switchback designs flip the whole market between A and B in alternating time windows. The trade-off is power. You now have dozens of clusters or windows instead of millions of users, so variance is much higher and tests take longer. Use them only when interference is plausible, and check carryover between switchback windows.

</details>

<details>
<summary><strong>Your 50/50 experiment shows 50.6% of users in control and 49.4% in treatment out of 100,000. The treatment looks great. What do you do?</strong></summary>

Stop and do not trust the result: that is a sample ratio mismatch. With 100,000 users the chi-square statistic is $2 \times 600^2 / 50{,}000 = 14.4$, a p-value around 0.00015, so it is not chance. SRM means the users who were analyzed are not the users who were randomized, so the arms are no longer comparable. Look for where users go missing. The treatment may crash or load slower, so some sessions never log. Bot filtering may hit one arm more. A redirect may drop users, or assignment may happen after a condition the treatment affects. Slice the SRM by platform, browser, country, and day to localize it. Fix the cause, then rerun. Reweighting the data does not fix it, because you do not know which users went missing.

</details>

<details>
<summary><strong>Baseline conversion is 5%. How many users per arm do you need to detect a 5% relative lift at alpha 0.05 and 80% power?</strong></summary>

About 122,000 per arm. The target rate is $0.05 \times 1.05 = 0.0525$, so $\delta = 0.0025$ and $\delta^2 = 6.25 \times 10^{-6}$. The z-term is $(1.96 + 0.8416)^2 = 7.849$. The variance term is $0.05 \times 0.95 + 0.0525 \times 0.9475 = 0.0475 + 0.0497 = 0.0972$. Then $n = 7.849 \times 0.0972 / 6.25 \times 10^{-6} \approx 122{,}000$ per arm, about 244,000 in total. The quick rule $16\,p(1-p)/\delta^2 = 16 \times 0.0475 / 6.25 \times 10^{-6} = 121{,}600$ agrees. If you only get 20,000 eligible users a day, that is two weeks. If that is too long, you can test a bigger change, pick a more sensitive metric, use CUPED to cut variance, or accept a larger MDE.

</details>
