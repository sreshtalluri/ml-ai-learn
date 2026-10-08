<!-- GENERATED from learning-paradigms.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Learning paradigms

Covers the lesson [Learning paradigms](../lessons/01-ml-vocabulary/02-learning-paradigms.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/learning-paradigms/) grades these interactively and tracks a review queue.

## 1. Match (easy)

Match each problem to its most natural paradigm.

| Concept | Options |
|---|---|
| Predict house prices from past sales | Reinforcement learning |
| Group customers by purchasing behavior, no predefined segments | Self-supervised learning |
| Pretrain a language model on web text | Unsupervised clustering |
| Teach a robot arm to stack blocks from success signals | Supervised regression |

<details>
<summary>Answer</summary>

- Predict house prices from past sales → Supervised regression
- Group customers by purchasing behavior, no predefined segments → Unsupervised clustering
- Pretrain a language model on web text → Self-supervised learning
- Teach a robot arm to stack blocks from success signals → Reinforcement learning

Ask what feedback exists: known prices (labels), no labels, labels hidden in the text itself, or rewards after actions.

</details>

## 2. Calculation (easy)

Next-token prediction on a single sequence of 50 tokens creates how many (context, target) training pairs?

<details>
<summary>Answer</summary>

**49**

Every token except the first can be a target, using all earlier tokens as context: $50 - 1 = 49$.

</details>

## 3. Multiple choice (medium)

Fraud is confirmed by chargebacks that arrive up to 60 days later. Which framing is best for a first production model?

- **A.** Unsupervised clustering, because fraud labels are imperfect.
- **B.** Supervised classification on confirmed historical labels, validated with a time-based split.
- **C.** Reinforcement learning, because blocking a card is an action.
- **D.** Supervised regression predicting the transaction amount.

<details>
<summary>Answer</summary>

**B.** Supervised classification on confirmed historical labels, validated with a time-based split.

Reliable (if delayed) labels exist, so supervised learning is the strongest baseline. A time split mirrors deployment: train on the past, predict the future. Unsupervised anomaly scores make useful extra features.

- **A:** Imperfect labels are still far more informative than none.
- **B:** Correct.
- **C:** RL is unnecessary when logged outcomes give direct supervision, and exploring by letting fraud through is costly.
- **D:** The amount is an input, not the target.

</details>

## 4. Multiple choice (medium)

K-means on user behavior finds 5 clusters. A slide says "we discovered 5 types of users." What is the most accurate statement?

- **A.** Correct; K-means finds the true number of groups.
- **B.** K-means returns as many clusters as you ask for; the 5 clusters are hypotheses that need validation.
- **C.** The clusters are definitely meaningless.
- **D.** It should be 4 types, because cluster numbering starts at 0.

<details>
<summary>Answer</summary>

**B.** K-means returns as many clusters as you ask for; the 5 clusters are hypotheses that need validation.

K was chosen by a person, and K-means returns K groups even for random data. Check stability, interpretability, and usefulness before naming them.

- **A:** K-means does not choose K.
- **B:** Correct.
- **C:** They may be useful; they are unverified, not meaningless.
- **D:** Numbering has nothing to do with the number of clusters.

</details>

## 5. Select all that apply (hard)

You are predicting at signup whether a customer will churn within 90 days. Which features would leak information unavailable at prediction time? Select all that apply.

- **A.** Number of support tickets filed in the first 90 days
- **B.** Acquisition channel (ad, referral, organic)
- **C.** Whether the account was closed
- **D.** Plan chosen at signup

<details>
<summary>Answer</summary>

**A, C**

Only information available at the moment of prediction (signup) is allowed. Tickets filed during the next 90 days and account closure happen after signup. Closure is practically the label.

- **A:** Correct. It is measured over the very window you are predicting.
- **B:** Known at signup, so it is fine.
- **C:** Correct. This is close to the target itself.
- **D:** Known at signup, so it is fine.

</details>

## 6. Arrange in order (medium)

Order the problem-formulation checklist.

- Build a simple baseline before a complex model
- Choose a metric that matches the cost of errors
- List what information is available at prediction time
- Check that a trustworthy target exists
- Identify the prediction unit
- Define the decision the system must support

<details>
<summary>Answer</summary>

1. Define the decision the system must support
2. Identify the prediction unit
3. Check that a trustworthy target exists
4. List what information is available at prediction time
5. Choose a metric that matches the cost of errors
6. Build a simple baseline before a complex model

Decision first, model last.

</details>

## 7. Multiple choice (hard)

In self-training, the model labels unlabeled data with its own confident predictions and retrains. What is the main risk?

- **A.** It always lowers accuracy.
- **B.** Confident mistakes become training labels, so the model can reinforce its own errors (confirmation bias).
- **C.** It needs more labels than supervised learning.
- **D.** It only works for regression.

<details>
<summary>Answer</summary>

**B.** Confident mistakes become training labels, so the model can reinforce its own errors (confirmation bias).

Pseudo-labels are only as good as the model producing them. Use a high confidence threshold, monitor against a held-out labeled set, and stop if validation accuracy drops.

- **A:** It often helps when classes are well separated; it is not guaranteed either way.
- **B:** Correct.
- **C:** Its purpose is to use fewer labels.
- **D:** It works for classification, and variants exist for regression.

</details>

## 8. Reflection (medium)

Why can LLMs be pretrained on trillions of tokens when even the largest human-labeled datasets have only millions of examples?

<details>
<summary>Answer</summary>

**Model answer.** Because pretraining is self-supervised: the target for each position is simply the next token in the text, so every token of raw text is a free label. No human annotation is needed, so the only limits are the amount of text and compute.

Self-supervision turns raw data into supervised examples at no labeling cost.

</details>
