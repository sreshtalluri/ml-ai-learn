---
name: Naive Bayes
tags: [supervised, classification, nlp, low-latency, small-data, interpretable]
lessons: [naive-bayes, probability-and-statistics]
labs: []
---

# Naive Bayes

## Problem type

Supervised classification, especially text.

## Input

Word counts or TF-IDF (multinomial), binary indicators (Bernoulli), or continuous features (Gaussian).

## Output

Posterior probability per class.

## Mental model

Start from how common each class is, multiply by how typical each observed feature is of that class, and pick the class with the bigger product. Features are assumed independent given the class.

## Core objective

```math
\hat{c} = \arg\max_c \Big[\ln P(c) + \sum_j \ln P(x_j \mid c)\Big]
```

Parameters are estimated by counting (maximum likelihood with Laplace smoothing).

## Training process

One pass over the data to count class frequencies and per-class feature statistics.

## Preprocessing

Tokenize and count (text); smoothing $\alpha$; for Gaussian NB, transform skewed features.

## Assumptions

Conditional independence of features given the class (rarely true); the chosen likelihood family fits the features.

## Key hyperparameters

Smoothing $\alpha$; variant choice; class priors.

## Good use cases

Spam and topic classification baselines; tiny datasets; streaming updates.

## Poor use cases

When calibrated probabilities matter; strongly dependent features; complex interactions.

## Strengths

Extremely fast to train and predict; works with little data; handles many features; incremental.

## Weaknesses

Overconfident probabilities; usually beaten by logistic regression or linear SVMs with enough data.

## Computational cost

Training $O(np)$; prediction $O(Cp)$.

## Evaluation metrics

Accuracy, macro-F1, log loss (expect poor calibration).

## Failure modes

Zero probabilities without smoothing; underflow without logs; double-counting correlated evidence.

## Minimal implementation

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
nb = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0)).fit(train_texts, train_labels)
```

## Compared with neighbors

- **Logistic regression:** discriminative, better calibrated, usually more accurate with enough data.
- **Linear SVM:** strong text baseline without probability outputs.

## Learn more

[Naive Bayes](../lessons/05-instance-and-probabilistic/02-naive-bayes.md)
