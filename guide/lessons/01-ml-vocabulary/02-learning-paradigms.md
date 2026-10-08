---
title: Learning paradigms
summary: Tell supervised, unsupervised, self-supervised, semi-supervised, and reinforcement learning apart by the feedback they learn from, and choose one for a real problem.
skill: classical-ml
minutes: 30
prerequisites: [ml-vocabulary]
related: [ml-workflow, k-means, tokenization-and-pretraining]
---

# Learning paradigms

> **Mental model.** The paradigm is decided by the *feedback signal* in your data. Correct answers for every example: supervised. No answers: unsupervised. Answers hidden inside the data itself: self-supervised. A few answers: semi-supervised. Rewards for actions: reinforcement learning.

**You will learn to**
- Define each paradigm by its data signal and goal.
- Choose a paradigm for a concrete problem, and explain what would change the choice.
- Show how self-supervised learning creates labels from raw data (the basis of LLM pretraining).
- Recognize when a clustering result should not be treated as a real category.
- Apply the problem-formulation checklist before picking any algorithm.

**Why it matters.** Picking an algorithm before framing the problem is the most common way ML projects fail. The paradigm decides what data you need to collect, what you can measure, and what "working" even means. Problem formulation usually matters more than model novelty.

## 1. Intuition

Think about how people learn.

- A student with an answer key checks each practice problem and corrects mistakes. That is **supervised learning**: examples paired with correct answers (labels).
- Someone sorting a pile of unlabeled photos into piles that "look alike" has no answer key. That is **unsupervised learning**: find structure without being told what it is.
- A child learns language by hearing sentences and guessing the next word, then hearing the actual word. Nobody wrote labels; the data checks itself. That is **self-supervised learning**, and it is how large language models are pretrained.
- A student with a few worked examples and a big stack of unanswered problems learns from both. That is **semi-supervised learning**.
- Learning to ride a bike: no one labels each muscle movement; you get feedback (balance, falling) after sequences of actions. That is **reinforcement learning**.

## 2. Visualization

<!-- lab:paradigm-guide -->
![Left: a supervised classifier draws a boundary between labeled blue and orange points. Middle: K-means finds two clusters in the same points with no labels. Right: only six points are labeled; self-training uses the 194 unlabeled points to place a boundary.](../../figures/learning-paradigms.png)

*The same synthetic 2D data, three ways. With all 200 labels, logistic regression scores 0.98. With no labels, K-means' two clusters happen to match the hidden classes 97% of the time, because the classes are two well-separated blobs (on real data they often don't match). With only 6 labels, training on those 6 scores 0.96 and self-training on all 200 reaches 0.98. This data is easy; on harder data the gap from unlabeled examples is often larger, and sometimes negative.*

*Interactive version: answer three questions about your problem, or load one of seven scenarios (house prices, fraud, customer segments, anomaly detection, next-token prediction, recommendation, robot control). [Open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/paradigm-guide/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Before loading a scenario, decide which paradigm fits fraud detection. Then load it and compare.
2. Change only the first answer (the feedback signal) and watch the recommendation change. Why does the signal matter more than the output type?

## 3. The math

### Symbols

| Symbol | Meaning |
|---|---|
| $x$ | input features of one example |
| $y$ | its label (target) |
| $f_\theta$ | a model with parameters $\theta$ |
| $\ell(\hat{y}, y)$ | per-example loss comparing prediction and label |
| $\mathcal{D}$ | the dataset |
| $s, a, r$ | state, action, reward (reinforcement learning) |
| $\pi$ | a policy: a rule mapping states to actions |
| $\gamma$ | discount factor in $[0, 1)$ |

### Each paradigm as an objective

**Supervised:** labeled pairs $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$. Minimize average loss.

```math
\min_\theta \; \frac{1}{n}\sum_{i=1}^n \ell\big(f_\theta(x_i),\, y_i\big)
```

Regression when $y$ is a number; classification when $y$ is a category.

**Unsupervised:** inputs only, $\mathcal{D} = \{x_i\}$. Optimize a structural objective instead of agreement with labels, for example K-means' within-cluster distance $\sum_i \lVert x_i - \mu_{c(i)} \rVert^2$, or PCA's retained variance.

**Self-supervised:** build $(x, y)$ pairs from unlabeled data with a rule. For language, the target at position $t$ is token $t$ and the input is everything before it:

```math
\max_\theta \; \sum_{t} \log P_\theta(\text{token}_t \mid \text{token}_{<t})
```

The math is supervised; the labels are free.

**Semi-supervised:** a small labeled set $\mathcal{D}_L$ plus a large unlabeled set $\mathcal{D}_U$. One simple recipe (self-training): fit on $\mathcal{D}_L$, predict on $\mathcal{D}_U$, add confident predictions as pseudo-labels, refit.

**Reinforcement learning:** an agent in state $s_t$ takes action $a_t$, receives reward $r_t$, and moves to $s_{t+1}$. Find a policy that maximizes expected discounted return:

```math
\max_\pi \; \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]
```

### Worked example: making self-supervised labels

From the sentence "the model predicts the next token from the previous tokens" (10 words), next-word prediction creates 9 training pairs with no human labeling:

| Input (context) | Target |
|---|---|
| the | model |
| the model | predicts |
| the model predicts | the |
| the model predicts the | next |
| … | … |
| the model predicts the next token from the previous | tokens |

A sentence of $n$ tokens yields $n - 1$ examples. A trillion-token corpus yields about a trillion examples, which is why self-supervision scales so far beyond human labeling.

### Worked example: choosing a paradigm

**Problem:** flag fraudulent card transactions.

1. *Decision:* block, review, or approve each transaction.
2. *Prediction unit:* one transaction.
3. *Trustworthy target?* Yes, but delayed: chargebacks confirm fraud weeks later. About 0.1% are fraud.
4. *Information at prediction time:* amount, merchant, device, recent account history, but not the chargeback outcome.
5. *Metric:* precision and recall at a threshold chosen from the cost of a missed fraud versus a blocked good customer; not accuracy (99.9% for "never fraud").
6. *Baseline:* rules such as "amount > 5× the customer's median."

**Choice:** supervised classification on confirmed labels, with a time-based split. Add an unsupervised anomaly score as a feature to catch new fraud patterns that have no labels yet.

## 4. Implementation

```python
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

# Supervised: features and labels
clf = LogisticRegression().fit(X, y)

# Unsupervised: features only
clusters = KMeans(n_clusters=2, n_init=10).fit_predict(X)

# Semi-supervised: -1 marks unlabeled rows
semi = SelfTrainingClassifier(LogisticRegression(), threshold=0.9).fit(X, y_partial)

# Self-supervised: build (context, next-token) pairs from raw text
tokens = text.split()
pairs = [(tokens[:i], tokens[i]) for i in range(1, len(tokens))]
```

Runnable script with the figure and all four comparisons: [`code/01-ml-vocabulary/paradigms.py`](../../code/01-ml-vocabulary/paradigms.py).

## 5. Engineering

**Labels are the expensive part.** Supervised learning is the most predictable paradigm *if* labels exist and are trustworthy. Ask: who produces the labels, how noisy are they, how delayed, and do they drift? Weak labels (heuristics), active learning (label the most informative examples), and pretrained representations all reduce labeling cost.

**Self-supervision as a first stage.** Modern practice often pretrains a representation with self-supervision on huge unlabeled data, then adapts it with a small labeled set (fine-tuning) or uses it as-is (embeddings, prompting). That is the LLM recipe.

**Reinforcement learning is expensive.** It needs an environment or simulator, many interactions, and careful reward design (agents exploit loopholes in rewards). Many "RL problems" are better framed as supervised learning on logged decisions first. In LLMs, RL appears in preference optimization (RLHF), where a reward model is trained on human preference labels.

**Paradigms combine.** Recommendation systems mix self-supervised embeddings, supervised ranking on clicks, and sometimes RL-style exploration. Anomaly detection is unsupervised until labeled incidents accumulate, then becomes supervised.

> [!IMPORTANT]
> **A discovered cluster is not a verified category.** Unsupervised methods always return structure, even in random data. Treat clusters as hypotheses to validate with domain experts and downstream outcomes.

> [!WARNING]
> **Failure modes.** Labels that leak the future (a "was refunded" feature in a fraud model); proxy labels that measure the wrong thing (clicks instead of satisfaction); pseudo-labels that amplify the model's own mistakes in semi-supervised training; reward hacking in RL.

### Common mistakes

- Choosing an algorithm before defining the target, the prediction unit, and the metric.
- Calling a problem "unsupervised" when labels could be collected cheaply.
- Evaluating an unsupervised model only by its own objective (inertia), not by usefulness.
- Assuming self-supervised pretraining removes the need for evaluation data. You still need labeled examples to measure the task.

## 6. Knowledge check

<!-- quiz:learning-paradigms -->
**[Take the learning paradigms quiz](../../quizzes/learning-paradigms.md)**: classify scenarios, spot leaky labels, and frame problems.
<!-- /quiz -->

**Practice exercise.** For each problem, name the paradigm and the feedback signal: (a) predicting tomorrow's electricity demand from past demand and weather; (b) grouping support tickets into topics nobody has defined; (c) training an image model to predict a hidden patch of each image from the visible patches.

<details>
<summary>Solution</summary>

(a) Supervised regression; the label is the actual demand, which arrives the next day. Use a time split.
(b) Unsupervised clustering (or topic modeling); no labels. Validate topics with the support team.
(c) Self-supervised; the hidden patch is the label, created from the image itself.
</details>

**Implementation challenge.** Using the script's synthetic data, vary the number of labeled points from 2 to 50 and plot accuracy for "train on labeled points only" versus `SelfTrainingClassifier`. Then make the blobs overlap more (`cluster_std=2.5`) and see whether self-training still helps.

## Summary

- The feedback signal decides the paradigm: labels, no labels, self-made labels, few labels, or rewards.
- Supervised learning minimizes loss against labels; unsupervised optimizes a structural objective; self-supervised manufactures labels from raw data.
- LLMs are pretrained with self-supervision (next-token prediction) and then adapted with supervised and preference-based methods.
- Frame the problem first: decision, prediction unit, trustworthy target, available information, metric, baseline.
- Clusters are hypotheses, not facts.

**Next:** [The end-to-end ML workflow](../02-ml-workflow/01-ml-workflow.md)

**Related:** [K-means](../07-unsupervised/01-k-means.md) · [Tokenization and pretraining](../15-llms/01-tokenization-and-pretraining.md) · [The language of ML](01-ml-vocabulary.md)
