<!-- GENERATED from decoding.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Decoding

Covers the lesson [Decoding](../lessons/15-llms/02-decoding.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/decoding/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

Logits $[\ln 4, 0]$ at temperature $T = 0.5$. What probability does the first token get? (3 decimals)

<details>
<summary>Answer</summary>

**0.941** (within ±0.002)

Scaled logits $[2\ln 4, 0] = [\ln 16, 0]$; exponentials $[16, 1]$; $16/17 = 0.941$. At $T = 1$ it would be $4/5 = 0.8$.

</details>

## 2. Calculation (easy)

Probabilities $[0.4, 0.3, 0.2, 0.1]$ with top-k = 2. What is the renormalized probability of the first token? (4 decimals)

<details>
<summary>Answer</summary>

**0.5714** (within ±0.001)

Keep $[0.4, 0.3]$, sum 0.7; $0.4/0.7 = 0.571$.

</details>

## 3. Calculation (medium)

Same probabilities $[0.4, 0.3, 0.2, 0.1]$ with top-p = 0.85. How many tokens are kept?

<details>
<summary>Answer</summary>

**3**

Cumulative 0.4, 0.7, 0.9; the first three are needed to reach 0.85.

</details>

## 4. Match (medium)

Match each task to a sensible decoding setting.

| Concept | Options |
|---|---|
| Extract fields into JSON | Temperature about 0.7, top-p about 0.9 |
| Brainstorm 20 product names | Higher temperature sampling |
| General chat assistant | Greedy (temperature near 0) with a schema |

<details>
<summary>Answer</summary>

- Extract fields into JSON → Greedy (temperature near 0) with a schema
- Brainstorm 20 product names → Higher temperature sampling
- General chat assistant → Temperature about 0.7, top-p about 0.9

Determinism for structure, diversity for creativity, a balance for chat.

</details>

## 5. Multiple choice (medium)

A model answers a factual question wrongly. You set temperature to 0. What happens?

- **A.** It now answers correctly.
- **B.** It gives its single most likely answer every time, which may still be wrong.
- **C.** It refuses to answer.
- **D.** It produces random tokens.

<details>
<summary>Answer</summary>

**B.** It gives its single most likely answer every time, which may still be wrong.

Temperature controls randomness, not knowledge. Grounding (RAG) or tools address correctness.

</details>

## 6. Reflection (hard)

Why does beam search often produce bland, generic text in open-ended generation, even though it finds higher-probability sequences than sampling?

<details>
<summary>Answer</summary>

**Model answer.** Beam search maximizes total sequence probability, and the most probable continuations are generic and safe (common phrases, short answers, repetition), while interesting human text includes many moderately unlikely tokens. Maximizing likelihood over long outputs pushes toward the "average" continuation, so it's better suited to tasks with one correct output, like translation, than to open-ended writing.

High likelihood is not the same as high quality for open-ended text.

</details>
