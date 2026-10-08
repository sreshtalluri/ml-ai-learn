<!-- GENERATED from text-to-vectors.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: From text to vectors

Covers the lesson [From text to vectors](../lessons/09-classical-nlp/01-text-to-vectors.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/text-to-vectors/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

$N = 1000$ documents; a term appears in 10 of them. What is its IDF, $\ln(N/\text{DF})$? (3 decimals)

<details>
<summary>Answer</summary>

**4.605** (within ±0.002)

$\ln(100) = 4.605$.

</details>

## 2. Calculation (easy)

Same term, occurring 2 times in document $d$. What is its TF-IDF in $d$? (3 decimals)

<details>
<summary>Answer</summary>

**9.21** (within ±0.005)

$2 \times 4.605 = 9.210$.

</details>

## 3. Multiple choice (easy)

A word appears in every document of the corpus. What is its TF-IDF in any document (course formula)?

- **A.** Equal to its count
- **B.** 0
- **C.** 1
- **D.** Undefined

<details>
<summary>Answer</summary>

**B.** 0

$\ln(N/N) = \ln 1 = 0$, so it contributes nothing.

</details>

## 4. Calculation (medium)

What is the cosine similarity between $[1, 0, 1]$ and $[1, 1, 0]$?

<details>
<summary>Answer</summary>

**0.5** (within ±0.0001)

Dot product 1; each norm is $\sqrt{2}$; $1/2 = 0.5$.

</details>

## 5. Select all that apply (medium)

Which are genuine limitations of TF-IDF features? Select all that apply.

- **A.** Synonyms like "car" and "automobile" are unrelated features.
- **B.** Word order beyond the n-gram window is lost.
- **C.** It cannot be trained quickly on large corpora.
- **D.** Words unseen at training time are ignored.

<details>
<summary>Answer</summary>

**A, B, D**

TF-IDF is very fast; its weaknesses are semantics, order, and a frozen vocabulary.

</details>

## 6. Reflection (medium)

A sentiment classifier using TF-IDF with stop-word removal misclassifies "this movie was not good" as positive. Why, and what would you change?

<details>
<summary>Answer</summary>

**Model answer.** "not" is in many stop-word lists, so it was removed and the model saw only "movie good." Keep negations (customize the stop list or skip stop-word removal) and add bigrams so "not good" becomes its own feature.

Aggressive cleaning can remove meaning; n-grams capture short phrases.

</details>
