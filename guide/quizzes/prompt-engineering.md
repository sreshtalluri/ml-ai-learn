<!-- GENERATED from prompt-engineering.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Prompt engineering

Covers the lesson [Prompt engineering](../lessons/28-prompting-and-context/01-prompt-engineering.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/prompt-engineering/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What is the most common reason a prompt gives mediocre or inconsistent results?

- **A.** It doesn't include enough examples
- **B.** It leaves the label set, format, length, or edge cases unspecified, so the model guesses
- **C.** It uses the wrong API endpoint
- **D.** It uses too many tokens

<details>
<summary>Answer</summary>

**B.** It leaves the label set, format, length, or edge cases unspecified, so the model guesses

Whatever the prompt leaves out, the model fills in with its most typical guess. Specifying labels, format, and edge-case rules removes those guesses.

- **A:** Examples help, but a clearly specified zero-shot prompt often works; missing specification is the more common cause.
- **B:** Correct.
- **C:** The endpoint does not change how well the instructions describe the task.
- **D:** Length alone isn't the problem; a short vague prompt and a long contradictory one both fail.

</details>

## 2. Multiple choice (easy)

What does the system message usually hold in a chat API call?

- **A.** The model's architecture and weights
- **B.** Standing rules for the whole conversation, such as role, scope, and output format
- **C.** The API key and authentication
- **D.** A compressed copy of earlier conversation turns

<details>
<summary>Answer</summary>

**B.** Standing rules for the whole conversation, such as role, scope, and output format

The system message sets rules that apply to every turn. It is re-sent on every call, so it also costs tokens on every call.

- **A:** Weights are fixed; the system message is just text in the context.
- **B:** Correct.
- **C:** Authentication is an HTTP header, not part of the prompt.
- **D:** History compaction is a context-engineering technique; the system message is the standing instructions.

</details>

## 3. Multiple choice (medium)

A prompt says `Respond as JSON: {"sentiment": "<label>"}`. Most outputs look like `Here is the analysis:` followed by valid JSON. Your code calls `json.loads` on the whole output. What is the most robust fix?

- **A.** Raise the temperature so the model tries different phrasings
- **B.** Use the provider's structured-output or tool-calling feature with a JSON schema, and validate in code
- **C.** Add "please" to the instruction
- **D.** Strip the first line of every response

<details>
<summary>Answer</summary>

**B.** Use the provider's structured-output or tool-calling feature with a JSON schema, and validate in code

Schema-constrained decoding guarantees the output parses. Saying "output only the JSON" helps too, but validation in code is still needed.

- **A:** Higher temperature adds variety, including more preambles; it makes parsing less reliable.
- **B:** Correct.
- **C:** Politeness doesn't specify the format.
- **D:** A brittle patch that breaks the moment the preamble changes or disappears.

</details>

## 4. Multiple choice (easy)

Which temperature setting fits a classification prompt whose output feeds an automated pipeline?

- **A.** Low (near 0), so the most probable label is chosen consistently
- **B.** High (around 1.5), so the model explores more labels
- **C.** Whatever the default is; temperature doesn't affect classification
- **D.** Alternate between low and high on each call

<details>
<summary>Answer</summary>

**A.** Low (near 0), so the most probable label is chosen consistently

Low temperature concentrates probability on the top token. In the lesson's harness, the same prompt fell from 0.917 accuracy at T = 0 to about 0.70 at T = 1.0.

- **A:** Correct.
- **B:** More randomness gives noisier labels, not better ones.
- **C:** Temperature changes the sampling distribution for every task, including classification.
- **D:** This adds noise and makes results impossible to reproduce.

</details>

## 5. Calculation (medium)

A prompt passes 18 of 24 test cases. What is the standard error of the pass rate? (3 decimals)

<details>
<summary>Answer</summary>

**0.088** (within ±0.002)

$\hat p = 18/24 = 0.75$. $\hat p(1 - \hat p) = 0.75 \times 0.25 = 0.1875$. Divide by 24: $0.0078125$. Square root: $0.0884$.

</details>

## 6. Calculation (hard)

You want the standard error of a pass rate to be at most 0.05 even in the worst case (p = 0.5). What is the minimum number of test cases?

<details>
<summary>Answer</summary>

**100**

$\sqrt{0.5 \times 0.5 / n} \le 0.05 \Rightarrow 0.25 / n \le 0.0025 \Rightarrow n \ge 100$.

</details>

## 7. Multiple choice (medium)

You changed one word in a prompt, tried it on three examples, and all three looked better. On your 50-case test set the pass rate went from 0.88 to 0.90. What should you conclude?

- **A.** The change is an improvement; ship it
- **B.** The difference is within noise (about one case); look at which cases changed, and add cases before deciding
- **C.** The test set is wrong because it disagrees with the three examples
- **D.** The change made things worse

<details>
<summary>Answer</summary>

**B.** The difference is within noise (about one case); look at which cases changed, and add cases before deciding

A one-case change on 50 cases is far below the standard error (about 0.045). A paired look at fixed and broken cases, and a larger set, are needed.

- **A:** Three hand-picked examples plus a one-case gain is not evidence.
- **B:** Correct.
- **C:** Three examples are a far weaker sample than 50.
- **D:** It went up by one case; there's no evidence either way.

</details>

## 8. Multiple choice (medium)

A support bot answers policy questions incorrectly because the policies changed last month. Which fix addresses the cause?

- **A.** Add "You are a world-class policy expert" to the system prompt
- **B.** Retrieve the current policy documents and put the relevant ones in the context
- **C.** Lower the temperature to 0
- **D.** Add more output-format rules

<details>
<summary>Answer</summary>

**B.** Retrieve the current policy documents and put the relevant ones in the context

The model lacks current knowledge. Prompting changes behaviour, not facts; retrieval supplies the facts.

- **A:** A role biases style; it cannot supply knowledge the model never saw.
- **B:** Correct.
- **C:** Determinism makes it consistently wrong, not right.
- **D:** Format rules don't change what the model knows.

</details>

## 9. Multiple choice (medium)

Which practice makes it easiest to trace a bad production output back to its cause?

- **A.** Edit the prompt directly in the production config whenever needed
- **B.** Version prompts in git, run the test suite on every change, and log the prompt version with each call
- **C.** Keep several prompt variants and pick one at random per call
- **D.** Only change prompts during model upgrades

<details>
<summary>Answer</summary>

**B.** Version prompts in git, run the test suite on every change, and log the prompt version with each call

Prompts are code. Versioning, tests, and logging the version per call let you reproduce and attribute any output.

- **A:** Untracked edits make it impossible to know which prompt produced which output.
- **B:** Correct.
- **C:** Random variants without logging hide which one caused the failure.
- **D:** Bundling prompt and model changes means you can't tell which one caused a regression.

</details>
