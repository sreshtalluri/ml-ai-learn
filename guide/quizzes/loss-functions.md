<!-- GENERATED from loss-functions.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Loss functions

Covers the lesson [Loss functions](../lessons/10-neural-networks/02-loss-functions.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/loss-functions/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What does a loss function represent in neural-network training?

- **A.** The learning-rate schedule
- **B.** A differentiable measure of how wrong the predictions are, which the optimizer minimizes
- **C.** The number of incorrect predictions
- **D.** The size of the training set

<details>
<summary>Answer</summary>

**B.** A differentiable measure of how wrong the predictions are, which the optimizer minimizes

The loss maps predictions and targets to one scalar. The optimizer follows its gradient, so it must be differentiable (almost everywhere).

- **A:** The schedule controls step size; it isn't a measure of error.
- **B:** Correct.
- **C:** That is an error count, a metric with zero gradient almost everywhere, so it can't be optimized directly with gradient descent.
- **D:** Dataset size has nothing to do with the loss definition.

</details>

## 2. Multiple choice (medium)

Why is cross-entropy preferred over MSE for training a sigmoid or softmax classifier?

- **A.** Cross-entropy doesn't need labels
- **B.** Its gradient at the logit is p − y, while MSE's keeps a factor p(1 − p) that nearly vanishes when the model is confidently wrong
- **C.** MSE only works for regression and gives an error on classification data
- **D.** Cross-entropy is faster to compute

<details>
<summary>Answer</summary>

**B.** Its gradient at the logit is p − y, while MSE's keeps a factor p(1 − p) that nearly vanishes when the model is confidently wrong

At logit −5 for a positive example, cross-entropy pushes with −0.993 and MSE with −0.013.

- **A:** It needs labels just as much.
- **B:** Correct.
- **C:** MSE runs fine on 0/1 labels; it just learns poorly from confidently wrong predictions.
- **D:** The cost difference is negligible and not the reason.

</details>

## 3. Multiple choice (hard)

On a balanced binary dataset where the features carry no information, which constant prediction minimizes the loss?

- **A.** 0.5 under MSE, but 0 or 1 under cross-entropy
- **B.** 0.5 under both MSE and cross-entropy
- **C.** 0 under MSE, 0.5 under cross-entropy
- **D.** Cross-entropy has no minimum for a constant prediction

<details>
<summary>Answer</summary>

**B.** 0.5 under both MSE and cross-entropy

Both losses are proper scoring rules, so the best constant is the base rate, 0.5. MSE is then 0.25 and cross-entropy $\ln 2 = 0.693$. The losses differ in how hard they correct a committed model, not at this starting point.

- **A:** Predicting 1 for every example gives infinite cross-entropy on the half labelled 0.
- **B:** Correct.
- **C:** Predicting 0 gives MSE 0.5, worse than 0.25.
- **D:** $-[0.5\ln c + 0.5\ln(1-c)]$ is minimized at $c = 0.5$.

</details>

## 4. Calculation (medium)

A sigmoid classifier outputs logit $z = -5$ for an example with label 1, so $p = 0.0067$. What is the MSE gradient at the logit, $2(p - y)\,p(1 - p)$? Give 4 decimals.

<details>
<summary>Answer</summary>

**-0.0132** (within ±0.0003)

$2 \times (0.0067 - 1) \times 0.0067 \times 0.9933 = 2 \times (-0.9933) \times 0.006655 = -0.0132$. Cross-entropy's gradient is $p - y = -0.9933$, about 75 times larger.

</details>

## 5. Calculation (medium)

Residuals are $[-0.5, 0.5, 0, 10]$. What is the mean Huber loss with $\delta = 1$?

<details>
<summary>Answer</summary>

**2.4375** (within ±0.001)

Small residuals: $\tfrac12 \times 0.25 = 0.125$ each. Outlier: $1 \times (10 - 0.5) = 9.5$. Mean: $(0.125 + 0.125 + 0 + 9.5)/4 = 2.4375$. MSE on the same residuals is 25.125.

</details>

## 6. Multiple choice (medium)

With label smoothing $\varepsilon = 0.1$ and 10 classes, what is the target probability for the true class?

- **A.** 1.0
- **B.** 0.9
- **C.** 0.91
- **D.** 0.99

<details>
<summary>Answer</summary>

**C.** 0.91

$(1 - \varepsilon) + \varepsilon/K = 0.9 + 0.01 = 0.91$. Each other class gets 0.01.

- **A:** That is the unsmoothed one-hot target.
- **B:** This forgets that the true class also receives its share $\varepsilon/K$ of the spread mass.
- **C:** Correct.
- **D:** That would correspond to $\varepsilon \approx 0.011$.

</details>

## 7. Multiple choice (hard)

A 10-class model's predictions stay near uniform and the loss plateaus around 1.5. The code is `F.cross_entropy(F.softmax(logits, -1), y)`. What's the bug?

- **A.** The learning rate is too low
- **B.** [object Object]
- **C.** The labels should be one-hot
- **D.** Cross-entropy is the wrong loss for 10 classes

<details>
<summary>Answer</summary>

**B.** [object Object]

Inputs that differ by at most 1 give a softmax whose largest value is at most $e/(e + 9) \approx 0.23$, so the loss can't go below about 1.46. Pass raw logits.

- **A:** No learning rate fixes a loss floor built into the computation.
- **B:** Correct.
- **C:** F.cross_entropy takes integer class indices; one-hot isn't needed.
- **D:** Softmax cross-entropy is the standard multi-class loss.

</details>

## 8. Multiple choice (medium)

Each image can have any subset of 20 tags. Which output and loss do you use?

- **A.** 20 logits, softmax cross-entropy
- **B.** 20 logits, binary cross-entropy with logits per tag
- **C.** 1 logit, MSE
- **D.** 20 logits, InfoNCE

<details>
<summary>Answer</summary>

**B.** 20 logits, binary cross-entropy with logits per tag

Tags are independent yes/no decisions, so each gets its own sigmoid and BCE. Softmax would force the tags to compete for one unit of probability.

- **A:** Softmax assumes exactly one correct class per example.
- **B:** Correct.
- **C:** One number can't represent 20 independent tags.
- **D:** InfoNCE learns embeddings by contrasting pairs; it doesn't predict a tag set.

</details>

## 9. Multiple choice (hard)

You train a fraud model (0.5% positives) with `pos_weight=199`. A teammate wants to show the outputs to analysts as "probability of fraud". What should you do first?

- **A.** Nothing; the outputs are probabilities because they pass through a sigmoid
- **B.** Recalibrate on held-out data (for example Platt or isotonic), because the weighting shifted predicted probabilities upward
- **C.** Switch to MSE so the outputs are calibrated
- **D.** Divide every output by 199

<details>
<summary>Answer</summary>

**B.** Recalibrate on held-out data (for example Platt or isotonic), because the weighting shifted predicted probabilities upward

Reweighting changes the loss's optimum from the true conditional probability to an inflated one. Calibrate before interpreting, or train unweighted and move the threshold.

- **A:** A sigmoid output is in [0, 1], but it isn't calibrated after reweighting.
- **B:** Correct.
- **C:** Changing the loss doesn't undo the class weighting and brings back weak gradients.
- **D:** The relation between weighted and true probabilities is not a simple division; it's an odds rescaling, and calibrating on data is safer.

</details>
