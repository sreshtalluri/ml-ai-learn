<!-- GENERATED from llm-evaluation.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Evaluating LLM systems

Covers the lesson [Evaluating LLM systems](../lessons/17-llm-evaluation/01-llm-evaluation.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/llm-evaluation/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

Relevance of the top 4 results is $[1, 0, 0, 1]$ and there are 4 relevant documents in total. What is recall@4?

<details>
<summary>Answer</summary>

**0.5** (within ±0.0001)

2 relevant retrieved out of 4 relevant overall.

</details>

## 2. Calculation (medium)

Three queries have their first relevant result at ranks 1, 4, and 2. What is the MRR? (3 decimals)

<details>
<summary>Answer</summary>

**0.583** (within ±0.002)

$(1 + 0.25 + 0.5)/3 = 0.583$.

</details>

## 3. Calculation (hard)

Relevance $[0, 1]$ in the top 2, with one relevant document. What is nDCG@2? (3 decimals)

<details>
<summary>Answer</summary>

**0.631** (within ±0.002)

DCG $= 1/\log_2 3 = 0.631$; IDCG $= 1/\log_2 2 = 1$; ratio 0.631.

</details>

## 4. Calculation (medium)

Reference "the warranty lasts one year" (5 tokens). Answer "one year warranty" (3 tokens). Token F1? (3 decimals)

<details>
<summary>Answer</summary>

**0.75** (within ±0.002)

Overlap 3 tokens (one, year, warranty). Precision 3/3 = 1, recall 3/5 = 0.6, F1 = 2(1)(0.6)/1.6 = 0.75.

</details>

## 5. Select all that apply (medium)

Which are known biases of LLM-as-judge? Select all that apply.

- **A.** Preferring the answer shown first (position bias)
- **B.** Preferring longer answers (verbosity bias)
- **C.** Favoring outputs from its own model family
- **D.** Perfect agreement with human raters

<details>
<summary>Answer</summary>

**A, B, C**

Randomize order, use rubrics and references, and check agreement with humans.

</details>

## 6. Reflection (hard)

A new prompt raises accuracy from 0.80 to 0.84 on a 50-case eval set. Your manager wants to ship it today. What do you check first?

<details>
<summary>Answer</summary>

**Model answer.** Whether the gain is real: compare per case (how many fixed versus broken) and compute a paired bootstrap interval, since 2 net cases out of 50 is easily noise. Check which failure categories changed and whether anything regressed (safety, format, refusals). Check latency and cost per request. If it holds up, expand the eval set in the affected categories or run an online A/B test before a full rollout.

Paired, per-case comparisons plus operational metrics, before declaring a win.

</details>
