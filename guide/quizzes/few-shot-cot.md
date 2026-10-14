<!-- GENERATED from few-shot-cot.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Few-shot and chain-of-thought

Covers the lesson [Few-shot and chain-of-thought](../lessons/28-prompting-and-context/02-few-shot-cot.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/few-shot-cot/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What is the difference between zero-shot and few-shot prompting?

- **A.** Zero-shot gives only the instruction; few-shot adds input-output examples before the real input
- **B.** Zero-shot doesn't use a system prompt
- **C.** Few-shot uses a different, smaller model
- **D.** Few-shot trains the model on the examples

<details>
<summary>Answer</summary>

**A.** Zero-shot gives only the instruction; few-shot adds input-output examples before the real input

Few-shot examples are part of the context. They demonstrate format and style; no weights change.

- **A:** Correct.
- **B:** Either style can have a system prompt.
- **C:** The model is the same; only the context differs.
- **D:** In-context examples don't update weights; that would be fine-tuning.

</details>

## 2. Multiple choice (medium)

Why does chain-of-thought improve accuracy on multi-step problems even though the model's weights don't change?

- **A.** It activates a hidden reasoning mode in the weights
- **B.** Writing intermediate results as tokens lets later tokens condition on them, so each step is a small prediction instead of one big leap
- **C.** It makes the model use a larger vocabulary
- **D.** It reduces the number of tokens the model reads

<details>
<summary>Answer</summary>

**B.** Writing intermediate results as tokens lets later tokens condition on them, so each step is a small prediction instead of one big leap

A transformer does a fixed amount of work per token. Writing steps spreads the computation over more tokens and puts intermediate results in the context.

- **A:** There's no special mode; the mechanism is extra generated context.
- **B:** Correct.
- **C:** The vocabulary is fixed.
- **D:** CoT increases tokens; that's its cost.

</details>

## 3. Calculation (medium)

Each reasoning step fails independently with probability 0.1. What is the probability a 5-step chain is fully correct? (3 decimals)

<details>
<summary>Answer</summary>

**0.59** (within ±0.002)

$(1 - 0.1)^5 = 0.9^5$. $0.9^2 = 0.81$, $0.81^2 = 0.6561$, $0.6561 \times 0.9 = 0.5905$.

</details>

## 4. Calculation (hard)

Each chain is right with probability 0.6. With 3 independent chains, what is the probability that at least 2 are right? (3 decimals)

<details>
<summary>Answer</summary>

**0.648** (within ±0.002)

$k = 2$: $3 \times 0.6^2 \times 0.4 = 3 \times 0.36 \times 0.4 = 0.432$. $k = 3$: $0.6^3 = 0.216$. Sum $0.648$, up from 0.6 for one chain.

</details>

## 5. Multiple choice (hard)

In which situation does self-consistency voting make accuracy worse than a single chain?

- **A.** When single-chain accuracy is 0.9 and errors are random
- **B.** When the model usually makes the same wrong answer, so wrong chains agree and outvote the right one
- **C.** When you use 5 chains instead of 3
- **D.** When the answers are numbers

<details>
<summary>Answer</summary>

**B.** When the model usually makes the same wrong answer, so wrong chains agree and outvote the right one

Voting amplifies whatever the majority does. With agreeing wrong chains and p = 0.4, five-way voting gives about 0.32.

- **A:** High accuracy with random errors is where voting helps most.
- **B:** Correct.
- **C:** More chains help when errors scatter; the count alone doesn't cause harm.
- **D:** Short comparable answers like numbers are what voting needs.

</details>

## 6. Multiple choice (medium)

A teammate runs self-consistency with 5 chains at temperature 0 and sees no accuracy gain. Why?

- **A.** Five chains is too few
- **B.** Greedy decoding returns the same chain every time, so all five votes are identical
- **C.** Temperature 0 disables chain-of-thought
- **D.** The extractor can't parse greedy outputs

<details>
<summary>Answer</summary>

**B.** Greedy decoding returns the same chain every time, so all five votes are identical

Self-consistency needs diverse samples; at temperature 0 there is nothing to vote over.

- **A:** The count isn't the issue; identical samples give one vote five times.
- **B:** Correct.
- **C:** The model still writes its reasoning at temperature 0.
- **D:** Parsing doesn't depend on temperature.

</details>

## 7. Multiple choice (medium)

Your few-shot classifier labels most inputs with the label of the last example. What is the best first fix?

- **A.** Add more examples with the same label
- **B.** Balance examples across labels, vary their order, and retrieve examples similar to each input
- **C.** Remove the instruction and keep only examples
- **D.** Raise the temperature

<details>
<summary>Answer</summary>

**B.** Balance examples across labels, vary their order, and retrieve examples similar to each input

Models can be biased by label frequency and recency in examples. Balance, order, and relevance address the cause.

- **A:** That strengthens the bias.
- **B:** Correct.
- **C:** Instructions and label definitions help reduce the bias; removing them makes it worse.
- **D:** Random noise doesn't fix a systematic bias.

</details>

## 8. Multiple choice (medium)

You switch to a reasoning model that thinks before answering. What should you do with your hand-written "Let's think step by step" scaffolding?

- **A.** Keep it and add more worked reasoning examples
- **B.** Remove it, state the goal, constraints, and output format clearly, and control effort with the reasoning or thinking budget setting
- **C.** Replace it with "answer immediately"
- **D.** Run self-consistency with 20 chains on top

<details>
<summary>Answer</summary>

**B.** Remove it, state the goal, constraints, and output format clearly, and control effort with the reasoning or thinking budget setting

Reasoning models already do chain-of-thought internally; extra scaffolding is redundant and can hurt.

- **A:** Redundant tokens, and rigid examples can constrain the model's own reasoning.
- **B:** Correct.
- **C:** That fights the model's design and usually lowers accuracy on hard tasks.
- **D:** Very expensive; measure a single call first.

</details>

## 9. Match (medium)

Match each task with the technique that fits it best.

| Concept | Options |
|---|---|
| Extract invoice fields into an exact JSON layout | Tree-of-thought search |
| Grade-school word problem on a non-reasoning model | Self-consistency voting |
| High-stakes numeric answer, no way to verify it | Chain-of-thought |
| Puzzle where single chains often dead-end | Few-shot examples plus structured output |

<details>
<summary>Answer</summary>

- Extract invoice fields into an exact JSON layout → Few-shot examples plus structured output
- Grade-school word problem on a non-reasoning model → Chain-of-thought
- High-stakes numeric answer, no way to verify it → Self-consistency voting
- Puzzle where single chains often dead-end → Tree-of-thought search

Examples fix format, CoT lowers per-step error, voting removes random errors, and tree search explores alternatives.

</details>
