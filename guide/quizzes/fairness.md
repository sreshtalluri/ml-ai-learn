<!-- GENERATED from fairness.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Fairness

Covers the lesson [Fairness](../lessons/21-safety-security/02-fairness.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/fairness/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

Which condition is demographic parity?

- **A.** Equal selection rates, $P(\hat Y = 1 \mid A = a)$, across groups
- **B.** Equal true positive and false positive rates across groups
- **C.** Equal precision across groups
- **D.** Equal base rates across groups

<details>
<summary>Answer</summary>

**A.** Equal selection rates, $P(\hat Y = 1 \mid A = a)$, across groups

Demographic parity equalizes how often each group gets a positive decision, regardless of the true outcome.

- **A:** Correct.
- **B:** That is equalized odds.
- **C:** That is predictive parity.
- **D:** Base rates are a property of the population, not a constraint on the model.

</details>

## 2. Match (easy)

Match each criterion to the quantity it equalizes across groups.

| Concept | Options |
|---|---|
| Demographic parity | Precision (PPV) |
| Equal opportunity | True positive rate and false positive rate |
| Equalized odds | True positive rate |
| Predictive parity | Selection rate |

<details>
<summary>Answer</summary>

- Demographic parity → Selection rate
- Equal opportunity → True positive rate
- Equalized odds → True positive rate and false positive rate
- Predictive parity → Precision (PPV)

Each criterion holds a different ratio from the per-group confusion matrix equal.

</details>

## 3. Multiple choice (medium)

Two groups have different base rates. For an imperfect classifier, which statement is true?

- **A.** All group criteria can be satisfied with enough data
- **B.** Equal precision and equal error rates (FPR and FNR) cannot all hold at once
- **C.** Demographic parity implies equalized odds
- **D.** Removing the group attribute satisfies all criteria

<details>
<summary>Answer</summary>

**B.** Equal precision and equal error rates (FPR and FNR) cannot all hold at once

$\text{FPR} = \frac{p}{1-p}\cdot\frac{1-\text{PPV}}{\text{PPV}}\cdot\text{TPR}$ ties them together; different $p$ forces at least one gap (Chouldechova 2017).

- **A:** More data does not change the arithmetic linking the rates to the base rate.
- **B:** Correct.
- **C:** With different base rates, equal selection rates usually force unequal error rates.
- **D:** The disparity usually survives through proxies, and you lose the ability to measure it.

</details>

## 4. Calculation (medium)

A group of 100 has 40 true positives in reality. The model has TPR 0.75 and FPR 0.10 in this group. What is its precision (PPV)? Give a decimal to three places.

<details>
<summary>Answer</summary>

**0.833** (within ±0.002)

$\text{TP} = 0.75 \times 40 = 30$. $\text{FP} = 0.10 \times 60 = 6$. $\text{PPV} = 30 / 36 = 0.833$.

</details>

## 5. Calculation (easy)

Group A is selected at 0.36 and group B at 0.48. What is the selection-rate ratio (lower over higher)?

<details>
<summary>Answer</summary>

**0.75** (within ±0.005)

$0.36 / 0.48 = 0.75$, below the four-fifths (0.8) rule of thumb used for adverse impact in US employment.

</details>

## 6. Multiple choice (medium)

A team removed gender from a hiring model's features and claims the model is now fair. What is the best response?

- **A.** Agree; the model cannot discriminate on a feature it never sees
- **B.** Ask for a per-group audit, because proxies like name, school, and work history can carry the same information
- **C.** Remove all features to be safe
- **D.** Retrain with a higher learning rate

<details>
<summary>Answer</summary>

**B.** Ask for a per-group audit, because proxies like name, school, and work history can carry the same information

Fairness through unawareness rarely removes disparity, and it makes measurement harder. Collect the attribute for auditing and compute per-group rates.

- **A:** Correlated features let the model reconstruct much of the removed attribute.
- **B:** Correct.
- **C:** A model with no features is useless and still not "fair" by any useful criterion.
- **D:** Optimization settings don't address proxy features.

</details>

## 7. Multiple choice (medium)

For a model that flags transactions and blocks customers' cards for review, which criterion best targets the main harm to customers?

- **A.** Equal false positive rates across groups
- **B.** Equal selection rates across groups
- **C.** Equal precision across groups
- **D.** Equal base rates

<details>
<summary>Answer</summary>

**A.** Equal false positive rates across groups

The direct harm is innocent customers being blocked. Equal FPR means innocent customers in each group face the same chance of that.

- **A:** Correct.
- **B:** If fraud rates differ by group, equal flag rates would add false flags in one group or miss fraud in another.
- **C:** Precision matters for review cost, but the customer harm is false flags among the innocent.
- **D:** Base rates describe the world; the model cannot set them.

</details>

## 8. Multiple choice (hard)

In the causal model X = U + 2A, a person has A = 1 and X = 7. Under counterfactual fairness, what is their counterfactual X if A had been 0?

- **A.** 7
- **B.** 5
- **C.** 3
- **D.** 9

<details>
<summary>Answer</summary>

**B.** 5

Abduction gives $U = 7 - 2 = 5$. Flip $A$ to 0 and recompute, $X = 5 + 0 = 5$. A predictor using raw $X$ would change; one using $U$ would not.

- **A:** That assumes A has no effect on X.
- **B:** Correct.
- **C:** That subtracts the effect twice.
- **D:** That adds the effect instead of removing it.

</details>

## 9. Multiple choice (medium)

An audit shows a TPR of 0.80 for one group based on 40 positives and 0.86 for another based on 4,000. What should the report say?

- **A.** The model is clearly unfair to the first group
- **B.** The gap is within noise for the small group (95% interval roughly ±0.12), so report intervals and collect more data before concluding
- **C.** The second group is being favoured by 6 points
- **D.** TPR should not be reported per group

<details>
<summary>Answer</summary>

**B.** The gap is within noise for the small group (95% interval roughly ±0.12), so report intervals and collect more data before concluding

Small groups give noisy rates. Report confidence intervals and avoid chasing gaps smaller than the noise.

- **A:** A 6-point gap on 40 positives is not distinguishable from chance.
- **B:** Correct.
- **C:** Same problem; the difference is not statistically clear.
- **D:** Per-group TPR is exactly what equal opportunity requires.

</details>
