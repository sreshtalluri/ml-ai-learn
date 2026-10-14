<!-- GENERATED from hyperparameter-tuning.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Hyperparameter tuning

Covers the lesson [Hyperparameter tuning](../lessons/24-applied-ml/03-hyperparameter-tuning.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/hyperparameter-tuning/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

Which of these is a hyperparameter rather than a parameter?

- **A.** A weight in a logistic regression
- **B.** A split threshold chosen by a decision tree during training
- **C.** The maximum depth allowed for the tree
- **D.** A bias term in a neural network layer

<details>
<summary>Answer</summary>

**C.** The maximum depth allowed for the tree

Max depth is set before training and constrains what is learned. The others are learned from data.

- **A:** Weights are learned by minimizing the loss.
- **B:** The tree learns thresholds from the data.
- **C:** Correct.
- **D:** Biases are learned along with weights.

</details>

## 2. Multiple choice (medium)

Why does random search usually find better hyperparameters than grid search with the same number of trials?

- **A.** Random search uses a smarter acquisition function
- **B.** Only a few hyperparameters usually matter, and random search tries more distinct values of each
- **C.** Grid search cannot run in parallel
- **D.** Random search trains each model for longer

<details>
<summary>Answer</summary>

**B.** Only a few hyperparameters usually matter, and random search tries more distinct values of each

A 3 by 3 grid tries 3 values of the important knob; 9 random trials try 9.

- **A:** Random search ignores past results; acquisition functions belong to Bayesian optimization.
- **B:** Correct.
- **C:** Grid search parallelizes just as easily.
- **D:** Both use the same training procedure per trial.

</details>

## 3. Calculation (medium)

The top 5% of a search space is "good enough". With 30 independent random trials, what is the probability that at least one lands in it? (3 decimals)

<details>
<summary>Answer</summary>

**0.785** (within ±0.002)

$1 - 0.95^{30} = 1 - 0.2146 = 0.785$.

</details>

## 4. Calculation (medium)

Successive halving starts 27 configurations at 1 epoch with $\eta = 3$ and stops at 9 epochs (budgets 1, 3, 9). How many epochs does it use in total?

<details>
<summary>Answer</summary>

**81**

27 × 1 + 9 × 3 + 3 × 9 = 27 + 27 + 27 = 81, versus 243 to train all 27 for 9 epochs.

</details>

## 5. Multiple choice (easy)

You sample a learning rate uniformly between 0.0001 and 1. About what fraction of samples fall below 0.01?

- **A.** About 50%
- **B.** About 1%
- **C.** About 10%
- **D.** About 99%

<details>
<summary>Answer</summary>

**B.** About 1%

The interval [0.0001, 0.01] is about 1% of [0.0001, 1] in linear length, though it spans 2 of the 4 decades. Sample log-uniformly instead, so each decade gets equal share.

- **A:** That would be true on a log scale, which is the point of using one.
- **B:** Correct.
- **C:** 0.1 would be the cutoff for about 10%.
- **D:** That is the fraction above 0.01.

</details>

## 6. Multiple choice (hard)

In expected improvement, two candidates have the same predicted mean, equal to the current best. Candidate A has σ = 0.04, candidate B has σ = 0.01. Which does EI prefer?

- **A.** A, because EI grows with uncertainty when the means tie
- **B.** B, because it is more certain
- **C.** Neither; EI is zero for both because the mean does not beat the best
- **D.** It depends only on the order they were sampled

<details>
<summary>Answer</summary>

**A.** A, because EI grows with uncertainty when the means tie

With mean equal to the best (ignoring ξ), z = 0 and EI = σ φ(0) = 0.3989σ, so A scores 0.016 and B 0.004. This is the exploration term.

- **A:** Correct.
- **B:** Certainty is rewarded only through the mean; with equal means, more uncertainty means more upside.
- **C:** EI is positive whenever σ > 0, because there is some chance of improvement.
- **D:** EI depends on the surrogate's mean and σ, not sampling order.

</details>

## 7. Multiple choice (hard)

A search over 200 configurations reports best_score_ = 0.91. The test set, touched once, gives 0.87. Preprocessing is inside a Pipeline and the split is correct. What is the most likely explanation?

- **A.** The test set is corrupted
- **B.** Selection optimism; the best of 200 noisy CV scores is biased upward
- **C.** The model is underfitting
- **D.** Random search is broken; use grid search

<details>
<summary>Answer</summary>

**B.** Selection optimism; the best of 200 noisy CV scores is biased upward

Choosing the max over many noisy estimates selects for luck. Nested CV estimates the true performance of the whole tuning procedure.

- **A:** Possible, but a gap of this size is expected after a large search, so check optimism first.
- **B:** Correct.
- **C:** Underfitting would hurt both scores similarly; the gap is the clue.
- **D:** Grid search has the same optimism, and more trials make it worse, not better.

</details>

## 8. Multiple choice (medium)

You tune a gradient-boosting model whose training takes 20 minutes. Which plan is most efficient?

- **A.** Grid search over learning rate, depth, number of trees, subsample, and column sample, 4 values each
- **B.** Fix a large number of trees with early stopping; random or TPE search over learning rate (log scale), depth, and regularization with a pruner
- **C.** Tune number of trees first with a fine grid, then everything else
- **D.** Use defaults; tuning never helps boosting

<details>
<summary>Answer</summary>

**B.** Fix a large number of trees with early stopping; random or TPE search over learning rate (log scale), depth, and regularization with a pruner

Early stopping chooses the number of trees for free, log-scale sampling covers the learning rate properly, and pruning stops bad trials early.

- **A:** 4^5 = 1,024 trials × 20 minutes is about 14 days.
- **B:** Correct.
- **C:** The best number of trees depends on the learning rate, so tuning it alone first is wasted.
- **D:** Defaults are a good start, but the learning rate and regularization usually matter.

</details>

## 9. Arrange in order (medium)

Order the steps of nested cross-validation for one outer fold.

- Average the outer-fold scores across all outer folds
- Score that model on the outer test fold
- Refit the best configuration on all the remaining data
- Run the hyperparameter search with inner cross-validation on the remaining data
- Hold out the outer test fold

<details>
<summary>Answer</summary>

1. Hold out the outer test fold
2. Run the hyperparameter search with inner cross-validation on the remaining data
3. Refit the best configuration on all the remaining data
4. Score that model on the outer test fold
5. Average the outer-fold scores across all outer folds

The outer test fold never influences which configuration is chosen, so the averaged outer scores estimate how the whole tuning procedure generalizes.

</details>

## 10. Reflection (hard)

Why can successive halving discard the configuration that would have been best at full budget, and how does Hyperband mitigate it?

<details>
<summary>Answer</summary>

**Model answer.** Early scores are only a proxy. Configurations with small learning rates or strong regularization learn slowly and rank low after one epoch even if they finish best, and noise can also cut a good config. Hyperband runs several successive-halving brackets with different starting budgets, including one that trains a few configurations at full budget from the start, so slow starters get a chance.

Learning curves can cross; early ranking assumes they don't.

</details>
