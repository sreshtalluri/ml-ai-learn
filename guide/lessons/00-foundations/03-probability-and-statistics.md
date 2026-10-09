---
title: Probability and statistics
summary: Conditional probability, Bayes' rule, expectation, variance, common distributions, sampling, and why correlation is not causation.
skill: math
minutes: 35
prerequisites: []
related: [naive-bayes, logistic-regression, classification-metrics]
---

# Probability and statistics

> **Mental model.** Probability is bookkeeping for uncertainty. A model's output is usually a probability, its loss is usually a log-probability, and its evaluation is a statistical estimate from a finite sample.

**You will learn to**
- Compute joint, marginal, and conditional probabilities.
- Apply Bayes' rule, including the base-rate effect that surprises almost everyone.
- Compute expectation and variance, and recognize normal, Bernoulli, binomial, and categorical distributions.
- Explain sampling error and why a metric from a test set is an estimate.
- Separate correlation from causation with a concrete example.

**Why it matters.** Logistic regression outputs $P(y=1 \mid x)$. Cross-entropy is a negative log-probability. Naive Bayes is Bayes' rule. Language models output a categorical distribution over tokens. And every evaluation number you report is a sample statistic with uncertainty.

## 1. Intuition

**Conditional probability** is probability after you learn something. The chance a random email is spam might be 20%; the chance it is spam *given* that it contains "free money" is much higher.

**Bayes' rule** flips a conditional: from "how often does a positive test happen when someone is sick" to "how likely is someone sick given a positive test." The second depends heavily on how common the disease is in the first place (the **base rate**). A 95%-accurate test for a rare disease produces mostly false alarms.

**Expectation** is the long-run average; **variance** measures spread around it.

**Correlation** means two quantities move together. It does not tell you that one causes the other: ice-cream sales and swimming incidents both rise in summer because of temperature, not because of each other.

## 2. Visualization

![Left: two normal density curves with different means and spreads. Middle: binomial probabilities for the number of heads in 10 flips with p = 0.3. Right: a scatter of ice-cream sales against swimming incidents with correlation 0.66, both generated from temperature.](../../figures/probability-and-statistics.png)

*Right panel is synthetic: both variables are generated from temperature and do not affect each other, yet their correlation is 0.66.*

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $P(A)$ | probability of event $A$, between 0 and 1 |
| $P(A, B)$ | probability that both $A$ and $B$ happen |
| $P(A \mid B)$ | probability of $A$ given that $B$ happened |
| $\mathbb{E}[X]$ | expectation (mean) of random variable $X$ |
| $\text{Var}(X)$, $\sigma$ | variance and standard deviation |
| $\mu$ | mean of a distribution |

### Core rules

```math
P(A \mid B) = \frac{P(A, B)}{P(B)}, \qquad
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}, \qquad
P(B) = \sum_{a} P(B \mid A=a)\,P(A=a)
```

The middle one is **Bayes' rule**: posterior = likelihood × prior / evidence.

```math
\mathbb{E}[X] = \sum_x x\,P(X=x), \qquad \text{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big]
```

### Distributions you will meet

| Distribution | Describes | Mean | Variance |
|---|---|---|---|
| Bernoulli($p$) | one yes/no outcome | $p$ | $p(1-p)$ |
| Binomial($n$, $p$) | successes in $n$ trials | $np$ | $np(1-p)$ |
| Categorical($p_1..p_K$) | one of $K$ classes (or tokens) | | |
| Normal($\mu$, $\sigma^2$) | sums of many small effects, measurement noise | $\mu$ | $\sigma^2$ |

### Worked example 1: Bayes' rule and the base rate

A disease affects 1% of people. A test detects it 95% of the time (sensitivity) and gives a false positive 5% of the time. You test positive. What is the chance you have the disease?

```math
P(D \mid +) = \frac{P(+ \mid D)P(D)}{P(+ \mid D)P(D) + P(+ \mid \neg D)P(\neg D)} = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.05 \times 0.99} = \frac{0.0095}{0.0095 + 0.0495} = \frac{0.0095}{0.059} = 0.161
```

Only 16.1%. Out of 10,000 people, 95 sick people test positive and 495 healthy people test positive. The false positives swamp the true ones because healthy people are 99 times more common. This is exactly why accuracy misleads on imbalanced classification.

### Worked example 2: expectation and variance

A die roll $X$: $\mathbb{E}[X] = (1+2+3+4+5+6)/6 = 3.5$. $\mathbb{E}[X^2] = (1+4+9+16+25+36)/6 = 91/6 = 15.17$, so $\text{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2 = 15.17 - 12.25 = 2.92$ and $\sigma = 1.71$.

### Worked example 3: sampling error

A classifier gets 870 of 1,000 test examples right: accuracy $\hat{p} = 0.87$. The standard error of a proportion is $\sqrt{\hat{p}(1-\hat{p})/n} = \sqrt{0.87 \times 0.13 / 1000} = \sqrt{0.0001131} = 0.0106$. A rough 95% interval is $0.87 \pm 1.96 \times 0.0106 = [0.849, 0.891]$. A competing model scoring 0.88 on the same set is not clearly better.

## 4. Implementation

```python
import numpy as np
rng = np.random.default_rng(0)

# Bayes by simulation: 1,000,000 people
sick = rng.random(1_000_000) < 0.01
positive = np.where(sick, rng.random(sick.size) < 0.95, rng.random(sick.size) < 0.05)
print(sick[positive].mean())          # ≈ 0.161

# Bootstrap confidence interval for accuracy
correct = rng.random(1000) < 0.87
boot = [rng.choice(correct, correct.size).mean() for _ in range(2000)]
print(np.percentile(boot, [2.5, 97.5]))
```

Runnable script: [`code/00-foundations/foundations.py`](../../code/00-foundations/foundations.py).

## 5. Engineering

**Every metric is an estimate.** Report confidence intervals (bootstrap is simple and general), especially for small test sets or rare classes. Compare models on the same examples (paired comparisons are far more sensitive).

**Base rates drive precision.** When positives are rare, even a good classifier's alarms are mostly false. Choose thresholds and metrics with the base rate in mind.

**Correlation versus causation.** A predictive model exploits correlations; it doesn't learn what would happen if you *intervened*. Don't read coefficients or feature importances as causal effects. Estimating causal effects needs experiments (A/B tests) or causal-inference methods.

**Sampling and representativeness.** A model learns the distribution it was trained on. If production users differ (new regions, seasons, products), performance changes. That is distribution shift.

> [!WARNING]
> **Failure modes.** Ignoring base rates; treating a 0.5-point metric difference on 500 examples as real; drawing causal conclusions from observational data; training on a sample that doesn't represent production.

### Common mistakes

- Confusing $P(A \mid B)$ with $P(B \mid A)$ (the prosecutor's fallacy).
- Assuming independence when events are linked.
- Reporting a single number without a sense of its uncertainty.

## 6. Knowledge check

<!-- quiz:probability-and-statistics -->
**[Take the probability quiz](../../quizzes/probability-and-statistics.md)**
<!-- /quiz -->

**Practice exercise.** 30% of emails are spam. "Winner" appears in 40% of spam and 2% of non-spam. What is $P(\text{spam} \mid \text{"winner"})$?

<details>
<summary>Solution</summary>

$\frac{0.4 \times 0.3}{0.4 \times 0.3 + 0.02 \times 0.7} = \frac{0.12}{0.12 + 0.014} = \frac{0.12}{0.134} = 0.896$.
</details>

**Implementation challenge.** Simulate the disease example for base rates of 0.1%, 1%, 10%, and 50%, and plot $P(D \mid +)$. Explain the curve.

## Summary

- Conditional probability updates beliefs with evidence; Bayes' rule flips the conditional and depends on the prior.
- Rare positives make even accurate tests produce mostly false alarms.
- Expectation is the long-run mean; variance is spread. Know Bernoulli, binomial, categorical, and normal.
- Every evaluation metric is an estimate with uncertainty.
- Correlation is not causation; predictive models capture correlations.

**Next:** [The Python toolkit](04-python-toolkit.md)

**Related:** [Naive Bayes](../05-instance-and-probabilistic/02-naive-bayes.md) · [Classification metrics](../04-classification/02-classification-metrics.md)

## Interview angle

<details>
<summary><strong>A fraud model has 99% recall and a 1% false-positive rate. 0.1% of transactions are fraud. What fraction of flagged transactions are actually fraud?</strong></summary>

Use Bayes' rule, or just count. Per million transactions: 1,000 are fraud, and 990 of them are flagged; 999,000 are legitimate, and 1% of those, 9,990, are flagged. Precision $= \frac{990}{990 + 9{,}990} = \frac{990}{10{,}980} \approx 0.090$. Only about 9% of flags are real fraud, even though both component rates sound excellent. The false-positive rate applies to a population 999 times larger, so it dominates. What to say next: precision depends on prevalence, so the same model has different precision in markets with different fraud rates, and a validation set rebalanced to 50/50 would report a wildly optimistic precision. To improve it, lower the false-positive rate (higher threshold, better features) or route flags through a cheap review step, and always report precision at the production base rate.

</details>

<details>
<summary><strong>Why do we minimize the negative log-likelihood instead of maximizing the likelihood directly?</strong></summary>

They have the same optimum, because the logarithm is monotonic, but the log version is far easier to work with. The likelihood of i.i.d. data is a product, $\prod_i P(y_i \mid x_i)$: with a million examples each around 0.5, that is $0.5^{10^6}$, which underflows to exactly 0 in floating point. The log turns the product into a sum, $\sum_i \ln P(y_i \mid x_i)$, which is numerically stable and decomposes per example, so minibatch SGD works. Negating it gives a loss to minimize. It also has an information-theoretic meaning: the average negative log-likelihood is the cross-entropy between the data and the model, in nats. Common losses are special cases: squared error is the negative log-likelihood under Gaussian noise, binary cross-entropy under a Bernoulli model, and categorical cross-entropy under a categorical model.

</details>

<details>
<summary><strong>Model B beats model A by 0.8 accuracy points on the same 1,000-example test set. Do you ship B?</strong></summary>

Not on that evidence alone. At accuracy near 0.87 with $n = 1000$, the standard error is $\sqrt{0.87 \times 0.13 / 1000} \approx 0.0106$, so a 95% interval is roughly ±2.1 points, and a 0.8-point gap sits well inside the noise. Because both models are scored on the same examples, use a paired analysis: look only at examples where they disagree and run McNemar's test, or bootstrap the accuracy difference by resampling test examples. Paired comparisons are much tighter than two independent intervals. Also ask: was this test set used during tuning (then it is optimistic for whichever model was tuned more), does B win on the slices that matter, and do latency, cost, and calibration favor it? If the gap matters commercially, collect more test data or run an online A/B test.

</details>

<details>
<summary><strong>Users who adopt feature X churn 30% less. The PM wants to push every user into X. What do you say?</strong></summary>

That is a correlation in observational data, and it can't tell us what pushing users into X would do. The likely confounder is engagement: highly engaged users both discover X and stay, the same way temperature drives both ice-cream sales and swimming incidents. There may also be selection (people who need X differ from those who don't) and reverse causation (users already planning to leave stop exploring features). To estimate the causal effect, run a randomized experiment: prompt a random half of eligible users toward X and compare churn between the two arms, not between adopters and non-adopters. If an experiment is impossible, causal-inference methods such as matching on pre-treatment engagement or difference-in-differences around a launch can help, but they rest on assumptions you must state. Expect the true effect to be much smaller than 30%.

</details>
