<!-- GENERATED from topic-modeling.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Topic modeling

Covers the lesson [Topic modeling](../lessons/09-classical-nlp/04-topic-modeling.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/topic-modeling/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

How does LDA model a document?

- **A.** As a single topic chosen from a fixed list
- **B.** As a mixture over topics, where each topic is a probability distribution over words
- **C.** As a dense embedding vector
- **D.** As a sequence of hidden states

<details>
<summary>Answer</summary>

**B.** As a mixture over topics, where each topic is a probability distribution over words

Each document has a topic mixture $\theta_d$; each word's topic is drawn from it, and the word from that topic's $\phi_k$.

- **A:** That's the one-topic-per-document assumption of clustering methods like BERTopic's default, not LDA.
- **C:** LDA works on word counts; embeddings belong to BERTopic and similar methods.
- **D:** That describes an HMM or RNN, which models word order. LDA ignores order.

</details>

## 2. Multiple choice (medium)

Why does scikit-learn's LDA expect a CountVectorizer matrix rather than TF-IDF?

- **A.** LDA cannot store floating-point numbers
- **B.** LDA's likelihood is defined over integer word counts from a bag of words; TF-IDF weights break that assumption
- **C.** TF-IDF uses too much memory
- **D.** TF-IDF is slower to compute

<details>
<summary>Answer</summary>

**B.** LDA's likelihood is defined over integer word counts from a bag of words; TF-IDF weights break that assumption

The generative story draws one topic and one word per token, so its sufficient statistics are counts.

- **A:** The library would accept floats; the issue is the model's assumption, not the data type.
- **C:** Both are sparse matrices of the same shape.
- **D:** Speed is not the reason; TF-IDF is cheap to compute.

</details>

## 3. Calculation (medium)

Collapsed Gibbs update with $K = 2$, $V = 20$, $\alpha = 0.5$, $\beta = 0.1$. With the current token removed: $n_{d0} = 3$, $n_{d1} = 3$; $n_{0w} = 1$, $n_{1w} = 5$; $n_0 = n_1 = 98$. What is $P(z = 1)$? Give 3 decimals.

<details>
<summary>Answer</summary>

**0.823** (within ±0.002)

Denominators: $98 + 20 \times 0.1 = 100$ for both. Topic 0: $3.5 \times 1.1 / 100 = 0.0385$. Topic 1: $3.5 \times 5.1 / 100 = 0.1785$. $P(z=1) = 0.1785 / (0.0385 + 0.1785) = 0.1785 / 0.2170 = 0.823$.

</details>

## 4. Calculation (easy)

You fit 8 topics and take the top 6 words of each. Among those 48 words, 28 are unique. What is the topic diversity?

<details>
<summary>Answer</summary>

**0.583** (within ±0.002)

Diversity is unique words divided by total listed words, $28/48 = 0.583$. Low diversity suggests duplicate topics, so $K$ may be too large.

</details>

## 5. Multiple choice (medium)

You lower LDA's document-topic prior $\alpha$ from 1.0 to 0.05. What do you expect?

- **A.** Each document is explained by fewer topics (sparser mixtures)
- **B.** Each topic uses more words
- **C.** The number of topics changes
- **D.** Nothing; $\alpha$ only affects speed

<details>
<summary>Answer</summary>

**A.** Each document is explained by fewer topics (sparser mixtures)

In the Gibbs update, $(n_{dk} + \alpha)$ with small $\alpha$ strongly favors topics the document already uses.

- **B:** That's controlled by $\beta$, the topic-word prior.
- **C:** $K$ is fixed by you; $\alpha$ shapes how documents spread over those $K$ topics.
- **D:** $\alpha$ changes the posterior, not just runtime.

</details>

## 6. Multiple choice (easy)

In BERTopic output, what does topic id -1 mean?

- **A.** The largest topic
- **B.** Documents the density-based clustering (HDBSCAN) did not assign to any cluster
- **C.** Stop-word documents
- **D.** A reserved topic for empty strings

<details>
<summary>Answer</summary>

**B.** Documents the density-based clustering (HDBSCAN) did not assign to any cluster

HDBSCAN labels low-density points as noise. Watch the share of -1; if large, lower `min_topic_size` or use `reduce_outliers`.

- **A:** Topic 0 is usually the largest; -1 is special.
- **C:** Stop words are handled in the vectorizer, not by the clusterer.
- **D:** Empty strings are not a special category.

</details>

## 7. Multiple choice (hard)

An LDA model on support emails has a topic with top words "regards, thanks, sent, iphone, please". What is the best fix?

- **A.** Increase the number of topics until it disappears
- **B.** Remove signatures and boilerplate before vectorizing, add domain stop words, and lower max_df
- **C.** Switch to TF-IDF input
- **D.** Label it "customer politeness" and report it

<details>
<summary>Answer</summary>

**B.** Remove signatures and boilerplate before vectorizing, add domain stop words, and lower max_df

LDA found a real co-occurrence pattern (email boilerplate). Fix preprocessing, then rerun and read documents.

- **A:** More topics splits the junk rather than removing it.
- **C:** LDA expects counts, and boilerplate would still co-occur.
- **D:** That names an artifact as if it were a theme.

</details>

## 8. Multiple choice (medium)

Why does BERTopic usually beat LDA on short texts like tweets and product titles?

- **A.** Short texts contain fewer topics
- **B.** Sentence embeddings capture meaning even when documents share few words, while bag-of-words co-occurrence is too sparse
- **C.** BERTopic is always faster
- **D.** LDA cannot process documents under 100 words

<details>
<summary>Answer</summary>

**B.** Sentence embeddings capture meaning even when documents share few words, while bag-of-words co-occurrence is too sparse

LDA relies on word co-occurrence within documents; with 10 words there's little evidence. Embeddings place paraphrases near each other.

- **A:** The number of topics in a corpus doesn't depend on document length.
- **C:** Embedding every document usually makes BERTopic slower than LDA.
- **D:** LDA runs on any length; it's just noisier on short ones.

</details>

## 9. Multiple choice (hard)

A product team wants a weekly dashboard of themes in 5,000 long policy reports, on CPU only, with stable topic names. Which plan fits?

- **A.** Retrain BERTopic every week and show the topic ids directly
- **B.** Fit LDA or NMF once with fixed preprocessing, validate topics by reading documents, name them, freeze the model, and assign new reports with transform
- **C.** Use k-means on TF-IDF with K equal to the number of reports
- **D.** Use perplexity alone to pick K, then ship

<details>
<summary>Answer</summary>

**B.** Fit LDA or NMF once with fixed preprocessing, validate topics by reading documents, name them, freeze the model, and assign new reports with transform

Long documents suit mixed-membership models; freezing the model keeps ids and names stable across weeks.

- **A:** Retraining reshuffles topic ids, so a dashboard keyed on ids breaks, and long reports exceed embedding context unless chunked.
- **C:** One cluster per document is not a topic model.
- **D:** Perplexity often disagrees with human judgments of topic quality.

</details>

## 10. Select all that apply (medium)

Which are reasonable signals when choosing the number of topics $K$? Select all that apply.

- **A.** Topic coherence (for example $C_v$ or NPMI) across a range of K
- **B.** Topic diversity of top words
- **C.** Reading 10 to 20 documents per topic
- **D.** Picking the K that gives the lowest training loss on the same data

<details>
<summary>Answer</summary>

**A, B, C**

Combine automatic coherence and diversity with human reading. Fit on the training data alone always improves with more topics.

</details>
