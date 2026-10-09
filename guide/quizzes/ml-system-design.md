<!-- GENERATED from ml-system-design.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: ML system design

Covers the lesson [ML system design](../lessons/22-ml-system-design/01-ml-system-design.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/ml-system-design/) grades these interactively and tracks a review queue.

## 1. Arrange in order (easy)

Put these ML system design steps in the order you should cover them in an interview.

- Evaluation, launch, and monitoring
- Training pipeline and serving
- Baseline, then model choice
- Data, labels, and features
- Frame the ML task and choose metrics
- Clarify the goal and constraints

<details>
<summary>Answer</summary>

1. Clarify the goal and constraints
2. Frame the ML task and choose metrics
3. Data, labels, and features
4. Baseline, then model choice
5. Training pipeline and serving
6. Evaluation, launch, and monitoring

Goal and metrics come before any model, because they decide what "better" means. Serving and launch come after the model exists, and monitoring closes the loop.

</details>

## 2. Calculation (medium)

Using the lesson's synthetic funnel (overhead 20 ms; candidate generation $12 + 0.005$ ms per candidate; ranking $15 + 0.06$ ms per candidate; re-ranking $3 + 0.1$ ms per item, 100 items), what is the total latency in ms if you pass $k_1 = 2{,}000$ candidates to the ranker?

<details>
<summary>Answer</summary>

**190** (within ±0.5)

Candidate generation $12 + 2000 \times 0.005 = 22$ ms; ranking $15 + 2000 \times 0.06 = 135$ ms; re-ranking $3 + 100 \times 0.1 = 13$ ms. Total $20 + 22 + 135 + 13 = 190$ ms, 40 ms over a 150 ms budget.

</details>

## 3. Calculation (hard)

A click model was trained on all positives and 20% of negatives ($w = 0.2$). It outputs $q = 0.5$ for an impression. What is the calibrated click probability? Give 4 decimals.

<details>
<summary>Answer</summary>

**0.1667** (within ±0.001)

$p = q / (q + (1-q)/w) = 0.5 / (0.5 + 0.5/0.2) = 0.5 / (0.5 + 2.5) = 0.5 / 3 = 0.1667$. In odds: $1 \times 0.2 = 0.2$, so $p = 0.2/1.2 = 0.1667$.

</details>

## 4. Calculation (medium)

Each request costs 75 worker-ms of compute across the funnel. Peak traffic is 20,000 requests per second. How many workers are fully busy at peak (before any headroom)?

<details>
<summary>Answer</summary>

**1500**

$75 \times 20{,}000 = 1{,}500{,}000$ worker-ms per second, divided by 1,000 ms per worker-second, is 1,500 workers.

</details>

## 5. Multiple choice (hard)

A new ranker improves offline AUC from 0.78 to 0.81 but loses the A/B test. Logging shows that the feature "user clicks in the last hour" averages 4.1 in training data and 0.0 for 30% of online requests. What is the most likely cause?

- **A.** The model is overfitting and needs more regularization
- **B.** Training-serving skew, because the real-time feature is missing or defaulting online
- **C.** The A/B test needs more users
- **D.** AUC is the wrong offline metric for ranking

<details>
<summary>Answer</summary>

**B.** Training-serving skew, because the real-time feature is missing or defaulting online

A feature that is populated offline but zero for a large share of online requests is classic training-serving skew. The model leans on a signal it never actually receives in production.

- **A:** Overfitting would show up as a gap between training and validation metrics offline. It does not explain a feature that is zero only online.
- **C:** More users would sharpen the estimate of a real loss. It would not fix a broken feature.
- **D:** AUC has limits, but the logs point at a concrete data bug that explains the loss directly.

</details>

## 6. Select all that apply (medium)

You are predicting whether a user will like a video at impression time. Which features leak the label? Select all that apply.

- **A.** The user's total number of likes, computed from today's snapshot and joined to last month's impressions
- **B.** Watch time of this video in this session
- **C.** The video's like rate over the 7 days before the impression
- **D.** The user's account age at impression time

<details>
<summary>Answer</summary>

**A, B**

Today's total likes include likes that happened after the impression, possibly this one, so that is time travel. Session watch time is only known after the impression and strongly correlates with liking. Both features that are computed as of impression time are legitimate.

- **C:** Computed from data before the impression, so it is point-in-time correct.
- **D:** Known at impression time and unrelated to the outcome.

</details>

## 7. Multiple choice (medium)

Why is a two-tower model used for candidate generation but rarely as the final ranker?

- **A.** It cannot be trained on implicit feedback
- **B.** User and item only interact through one dot product, so it cannot use cross features; that same restriction is what lets item vectors be precomputed and searched with ANN
- **C.** Its embeddings are too large to store
- **D.** It only works for text

<details>
<summary>Answer</summary>

**B.** User and item only interact through one dot product, so it cannot use cross features; that same restriction is what lets item vectors be precomputed and searched with ANN

The late interaction is the trade-off. Precomputed item vectors plus ANN give sublinear retrieval, but the model cannot express "this user × this creator" signals that a ranker uses.

- **A:** Two-tower models are usually trained on implicit feedback such as clicks and watches.
- **C:** 10M items × 64 floats is a few GB, which is routinely served, and quantization shrinks it further.
- **D:** Towers can encode any features, including IDs, categorical data, and images.

</details>

## 8. Multiple choice (medium)

A ranker trained naively on click logs keeps promoting whatever the previous model put in slot 1, and new items rarely surface. Which change addresses the root cause?

- **A.** Train longer
- **B.** Model position explicitly (position as a feature fixed at serving time, or inverse propensity weighting) and reserve a small exploration slice
- **C.** Increase the number of items shown per page
- **D.** Switch from GBDT to a deep network

<details>
<summary>Answer</summary>

**B.** Model position explicitly (position as a feature fixed at serving time, or inverse propensity weighting) and reserve a small exploration slice

Clicks partly reflect position, not relevance. Modeling position and adding exploration breaks the feedback loop and gives new items a chance to collect labels.

- **A:** More training learns the biased signal better.
- **C:** More slots dilute the bias but do not remove it.
- **D:** Any model class trained on the same biased labels learns the same bias.

</details>

## 9. Match (medium)

Match each system to its most distinctive design constraint.

| Concept | Options |
|---|---|
| Fraud detection | batch prediction is fine because freshness is measured in days |
| Feed recommendation | relevance is defined per query and needs graded human judgments |
| Search ranking | position bias and popularity feedback loops |
| Weekly email recommendations | labels arrive weeks later and adversaries adapt |

<details>
<summary>Answer</summary>

- Fraud detection → labels arrive weeks later and adversaries adapt
- Feed recommendation → position bias and popularity feedback loops
- Search ranking → relevance is defined per query and needs graded human judgments
- Weekly email recommendations → batch prediction is fine because freshness is measured in days

Each case changes which part of the framework dominates. In fraud it is labels and drift, in feeds it is feedback loops, in search it is query-level relevance, and in a weekly email it is serving mode.

</details>

## 10. Reflection (hard)

In two or three sentences, explain how you would choose $k_1$, the number of candidates passed from retrieval to the ranker, for a feed with a 150 ms p99 budget.

<details>
<summary>Answer</summary>

**Model answer.** Measure candidate recall@k1 (the fraction of items users later engaged with that retrieval returned) on a time-based holdout, and measure ranker latency per candidate under production load. Pick the largest k1 whose p99 latency, including headroom for peaks, fits the ranker's share of the budget, then check that recall has saturated around it. If it has not, make the ranker cheaper (distillation, fewer features) or add a light pre-ranker rather than blowing the budget.

k1 trades recall against latency and compute. Both sides need to be measured, not guessed.

</details>
