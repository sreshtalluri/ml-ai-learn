<!-- GENERATED from experimentation-and-ab-testing.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Experimentation and A/B testing

Covers the lesson [Experimentation and A/B testing](../lessons/22-ml-system-design/02-experimentation-and-ab-testing.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/experimentation-and-ab-testing/) grades these interactively and tracks a review queue.

## 1. Multiple choice (medium)

An A/B test reports $p = 0.02$ for the new ranker. Which statement is correct?

- **A.** There is a 98% chance the new ranker is better
- **B.** There is a 2% chance the new ranker has no effect
- **C.** If the new ranker had no effect, a difference at least this large would appear about 2% of the time
- **D.** The new ranker improves the metric by 2%

<details>
<summary>Answer</summary>

**C.** If the new ranker had no effect, a difference at least this large would appear about 2% of the time

A p-value is the probability of data this extreme assuming the null hypothesis is true. It is not the probability of either hypothesis.

- **A:** That would be a posterior probability, which needs a prior on how often changes work.
- **B:** Same mistake in reverse. The p-value is computed assuming the null is true.
- **D:** The p-value says nothing about effect size. Use the estimate and its confidence interval.

</details>

## 2. Calculation (medium)

Baseline conversion 20%, relative MDE 10% (20% to 22%), $\alpha = 0.05$ two-sided, power 0.8. Using $n = (z_{0.975} + z_{0.8})^2 [p_A(1-p_A) + p_B(1-p_B)]/\delta^2$ with $(1.96 + 0.8416)^2 = 7.8489$, how many users per arm (round up)?

<details>
<summary>Answer</summary>

**6507** (within ±2)

Variance term $0.20 \times 0.80 + 0.22 \times 0.78 = 0.16 + 0.1716 = 0.3316$. $\delta^2 = 0.02^2 = 0.0004$. $n = 7.8489 \times 0.3316 / 0.0004 = 6{,}506.8$, so 6,507.

</details>

## 3. Calculation (medium)

Control converts 1,500 of 15,000; treatment 1,650 of 15,000. The pooled standard error is 0.00354. What is the z-statistic? Give 2 decimals.

<details>
<summary>Answer</summary>

**2.82** (within ±0.02)

$z = (0.110 - 0.100)/0.00354 = 0.010/0.00354 = 2.82$ (2.825 unrounded), giving a two-sided p-value of about 0.0047.

</details>

## 4. Multiple choice (easy)

You halve the minimum detectable effect and keep everything else the same. Roughly what happens to the required sample size?

- **A.** It halves
- **B.** It stays the same
- **C.** It doubles
- **D.** It roughly quadruples

<details>
<summary>Answer</summary>

**D.** It roughly quadruples

$n \propto 1/\delta^2$, so halving $\delta$ multiplies $n$ by about 4. The variance term changes slightly too, so it is "roughly".

- **A:** Sample size moves in the opposite direction to the MDE, and with the square.
- **B:** The MDE is in the denominator of the formula.
- **C:** Doubling would be true if $n \propto 1/\delta$. Noise shrinks with $\sqrt{n}$, so you need 4× the users.

</details>

## 5. Multiple choice (hard)

A 50/50 experiment on 200,000 users ends with 101,000 in control and 99,000 in treatment. Treatment looks like a big win. What should you do first?

- **A.** Ship it, the imbalance is only 1 percentage point
- **B.** Reweight the arms to 50/50 and recompute the result
- **C.** Treat it as a sample ratio mismatch ($\chi^2 = 20$, p far below 0.001), distrust the result, and find where treatment users are being lost
- **D.** Extend the test for another week

<details>
<summary>Answer</summary>

**C.** Treat it as a sample ratio mismatch ($\chi^2 = 20$, p far below 0.001), distrust the result, and find where treatment users are being lost

$\chi^2 = 2 \times 1000^2 / 100{,}000 = 20$. An imbalance that large is essentially impossible by chance, so the analyzed users are not the randomized ones.

- **A:** At this sample size, one point is about 4.5 standard deviations from 50/50.
- **B:** Reweighting cannot recover which kinds of users went missing, so the bias stays.
- **D:** More data from a broken assignment or logging pipeline is still broken.

</details>

## 6. Multiple choice (medium)

A PM checks the dashboard every day and plans to ship the moment $p < 0.05$. In the lesson's A/A simulation with 14 daily looks, roughly what false-positive rate does this produce?

- **A.** About 5%
- **B.** About 22%
- **C.** About 51%
- **D.** About 0.4%

<details>
<summary>Answer</summary>

**B.** About 22%

The simulation gave 0.216. Every look is another chance for noise to cross the threshold, but the looks are correlated, so the rate is well below the independent-tests bound of $1 - 0.95^{14} = 0.51$.

- **A:** Five percent holds only with a single pre-planned look.
- **B:** That bound assumes 14 independent tests. Cumulative data makes the looks correlated.
- **D:** Peeking raises the false-positive rate. It never lowers it.

</details>

## 7. Select all that apply (hard)

In which experiments does user-level randomization give a biased estimate because of interference? Select all that apply.

- **A.** A new surge-pricing model for riders in a ride-hailing marketplace
- **B.** A new font on the settings page
- **C.** A feature that makes users share more posts with friends
- **D.** Ads bidding changes where both arms spend from the same advertiser budget

<details>
<summary>Answer</summary>

**A, C, D**

Shared driver supply, social spillover, and shared budgets all let treatment affect control outcomes. A font on a settings page affects only the user who sees it.

- **B:** Each user's experience depends only on their own assignment.

</details>

## 8. Calculation (medium)

A test needs 40,000 users per arm. You apply CUPED with a pre-period covariate whose correlation with the outcome is $\rho = 0.6$. About how many users per arm do you need now?

<details>
<summary>Answer</summary>

**25600** (within ±10)

Variance is multiplied by $1 - \rho^2 = 1 - 0.36 = 0.64$, so $n = 0.64 \times 40{,}000 = 25{,}600$.

</details>

## 9. Multiple choice (medium)

A redesigned feed shows +6% engagement in week 1, +3% in week 2, and +1% in week 3. What is the most likely explanation?

- **A.** Primacy effect
- **B.** Novelty effect, so the long-run lift is probably small
- **C.** Sample ratio mismatch
- **D.** The test was underpowered in week 3

<details>
<summary>Answer</summary>

**B.** Novelty effect, so the long-run lift is probably small

An effect that decays over time as users get used to the change is a novelty effect. Estimate the long-run effect from later weeks or from new users only.

- **A:** Primacy is the reverse pattern, where users resist change at first and the effect grows.
- **B:** SRM is about assignment counts, not a trend in the effect.
- **C:** Power does not create a steady downward trend.

</details>

## 10. Reflection (hard)

You changed the system prompt of an LLM support assistant. Offline, an LLM judge prefers the new version 62% of the time. Describe how you would decide whether to ship.

<details>
<summary>Answer</summary>

**Model answer.** Treat the offline win as a gate, not a verdict. Check that the judge agrees with human labels on a sample, and that safety and regression suites still pass. Then run an A/B test randomized by user, with a primary metric such as resolution without human escalation. Add guardrails for cost per conversation, latency, thumbs-down rate, and safety flags, and make sure caches are keyed by arm. Fix the sample size and duration in advance, check SRM, and look at the effect over time for novelty.

Offline LLM evals measure a proxy on a fixed set. The online test measures real user outcomes and cost.

</details>
