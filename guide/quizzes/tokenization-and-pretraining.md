<!-- GENERATED from tokenization-and-pretraining.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Tokenization and pretraining

Covers the lesson [Tokenization and pretraining](../lessons/15-llms/01-tokenization-and-pretraining.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/tokenization-and-pretraining/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

The model assigns probability 0.25 to the actual next token. What is the cross-entropy for that token (natural log, 3 decimals)?

<details>
<summary>Answer</summary>

**1.386** (within ±0.002)

$-\ln 0.25 = \ln 4 = 1.386$.

</details>

## 2. Calculation (medium)

A model's mean next-token loss on a test set is $\ln 20$. What is its perplexity?

<details>
<summary>Answer</summary>

**20** (within ±0.001)

Perplexity is $e^{\text{loss}} = e^{\ln 20} = 20$.

</details>

## 3. Multiple choice (medium)

Word counts: "hug" ×10, "pug" ×5, "hugs" ×5. Starting from characters, which pair does BPE merge first?

- **A.** h u
- **B.** u g (appears 20 times)
- **C.** g s
- **D.** p u

<details>
<summary>Answer</summary>

**B.** u g (appears 20 times)

"u g" appears in every word: 10 + 5 + 5 = 20 times, more than "h u" (15).

</details>

## 4. Multiple choice (medium)

When computing the language-modeling loss, why are targets shifted one position relative to inputs?

- **A.** To save memory.
- **B.** Because the model at position k predicts the token at position k + 1.
- **C.** To skip the first token, which is always wrong.
- **D.** It isn't necessary.

<details>
<summary>Answer</summary>

**B.** Because the model at position k predicts the token at position k + 1.

Inputs are tokens 1..T−1; targets are tokens 2..T.

</details>

## 5. Select all that apply (medium)

Which statements about tokens are true? Select all that apply.

- **A.** API cost and context limits are measured in tokens.
- **B.** The same sentence always has the same number of tokens in every model.
- **C.** Non-English text and code often take more tokens per word than English prose.
- **D.** Perplexities from models with different tokenizers aren't directly comparable.

<details>
<summary>Answer</summary>

**A, C, D**

Different tokenizers segment text differently, which changes counts and the meaning of per-token loss.

</details>

## 6. Reflection (hard)

Explain why pretraining on next-token prediction can produce a model that states false things confidently.

<details>
<summary>Answer</summary>

**Model answer.** The objective rewards assigning high probability to text that tends to appear, not to text that is true. Fluent, plausible-sounding continuations can be likely even when wrong, especially for rare facts the model saw little of. The model has no built-in mechanism to check claims against a source, and confident phrasing is itself just likely text, not a calibrated signal of certainty.

Likely text is not the same as true text; that is why grounding, tools, and verification exist.

</details>
