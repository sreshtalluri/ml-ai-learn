<!-- GENERATED from scaling-laws.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Scaling laws

Covers the lesson [Scaling laws](../lessons/15-llms/06-scaling-laws.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/scaling-laws/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

Using $C \approx 6ND$, how many FLOPs does it take to train a 7B-parameter model on 2 trillion tokens? Give the answer in units of $10^{22}$ FLOPs.

<details>
<summary>Answer</summary>

**8.4** (within ±0.01)

$6 \cdot 7 \times 10^9 \cdot 2 \times 10^{12} = 84 \times 10^{21} = 8.4 \times 10^{22}$ FLOPs.

</details>

## 2. Calculation (medium)

With a budget of $C = 1.2 \times 10^{23}$ FLOPs and the "20 tokens per parameter" rule, how many billion parameters should the model have?

<details>
<summary>Answer</summary>

**31.6** (within ±0.2)

$D = 20N$ gives $C = 6N \cdot 20N = 120N^2$. $N^2 = 1.2 \times 10^{23}/120 = 10^{21}$, so $N = 3.16 \times 10^{10}$, about 31.6B parameters on 632B tokens.

</details>

## 3. Calculation (medium)

Using $L = E + A/N^\alpha + B/D^\beta$ with $E = 1.69$, suppose for some model $A/N^\alpha = 0.10$ and $B/D^\beta = 0.15$. If you double the training tokens ($\beta = 0.28$), what is the new predicted loss? (3 decimals)

<details>
<summary>Answer</summary>

**1.914** (within ±0.002)

Doubling $D$ divides the data term by $2^{0.28} = 1.214$: $0.15/1.214 = 0.1236$. New loss $1.69 + 0.10 + 0.1236 = 1.914$, down from $1.94$.

</details>

## 4. Multiple choice (medium)

GPT-3 has 175B parameters trained on 300B tokens. Under the Chinchilla analysis, what is its main inefficiency?

- **A.** It is undertrained; the same compute would give lower loss with a smaller model trained on more tokens.
- **B.** It is too small; more parameters would help most.
- **C.** It used too much data for its size.
- **D.** Its loss is already at the irreducible floor.

<details>
<summary>Answer</summary>

**A.** It is undertrained; the same compute would give lower loss with a smaller model trained on more tokens.

At under 2 tokens per parameter, the data term dominates its predicted loss (about 0.25 versus 0.06 for the model-size term). Chinchilla, 70B on 1.4T tokens, used similar compute more effectively.

- **B:** Adding parameters reduces the smaller of the two terms; the data term is the bottleneck.
- **C:** 300B tokens for 175B parameters is far below compute-optimal.
- **D:** The floor is 1.69 in this fit; its predicted loss is about 2.0.

</details>

## 5. Multiple choice (medium)

A company will serve its model to millions of users for two years. Why would it train an 8B model on 15T tokens instead of a compute-optimal larger model with the same training budget?

- **A.** Small models always reach lower loss.
- **B.** Inference cost scales with parameters and is paid on every request, so a small loss increase buys large lasting savings.
- **C.** Larger models cannot be trained on more than 1T tokens.
- **D.** Over-training reduces the irreducible loss.

<details>
<summary>Answer</summary>

**B.** Inference cost scales with parameters and is paid on every request, so a small loss increase buys large lasting savings.

Training cost is paid once; inference cost, about $2N$ FLOPs per token, is paid forever. Near the optimum the loss curve is flat, so moving to a smaller model costs little quality.

- **A:** At fixed compute, the compute-optimal model reaches the lowest loss by definition.
- **C:** There is no such limit; data availability is the practical constraint.
- **D:** $E$ is a property of the data, not of how long you train.

</details>

## 6. Multiple choice (hard)

Your team sees exact-match accuracy on a multi-step arithmetic benchmark go from 2% at 3B parameters to 55% at 30B and calls it an emergent ability. What do you check first?

- **A.** Whether a continuous metric, such as per-token log-likelihood of the correct answer, also jumps or improves smoothly.
- **B.** Whether the larger model used a higher temperature.
- **C.** Whether the tokenizer changed between runs.
- **D.** Nothing; a jump that large is proof of emergence.

<details>
<summary>Answer</summary>

**A.** Whether a continuous metric, such as per-token log-likelihood of the correct answer, also jumps or improves smoothly.

Exact match gives zero credit until every step is right, so smooth per-token improvements can appear as a sudden jump. Schaeffer et al. showed many reported emergent abilities become smooth under continuous metrics.

- **B:** Worth controlling, but sampling settings don't explain the metric artefact you should rule out first.
- **C:** Also worth confirming, but a shared tokenizer is the normal case within a model family.
- **D:** Large jumps on discontinuous metrics are exactly where artefacts hide.

</details>

## 7. Multiple choice (hard)

In the fitted law, $N^* \propto C^{\beta/(\alpha+\beta)}$ with $\alpha = 0.34$ and $\beta = 0.28$. What does that imply as compute grows?

- **A.** Optimal tokens per parameter stays exactly 20.
- **B.** Optimal tokens per parameter rises slowly with compute, because $D^*$ grows as $C^{0.55}$ and $N^*$ as $C^{0.45}$.
- **C.** Optimal tokens per parameter falls with compute.
- **D.** The optimal model size doesn't depend on compute.

<details>
<summary>Answer</summary>

**B.** Optimal tokens per parameter rises slowly with compute, because $D^*$ grows as $C^{0.55}$ and $N^*$ as $C^{0.45}$.

$D^*/N^* \propto C^{0.548 - 0.452} = C^{0.097}$. Over eight orders of magnitude that is about a 6× drift, from roughly 26 to 150 in the lesson's plot.

- **A:** Constant 20 corresponds to equal exponents (0.5 each), which Approaches 1 and 2 roughly found, not these constants.
- **C:** The data exponent is the larger one, so the ratio rises.
- **D:** $N^*$ grows as $C^{0.45}$.

</details>

## 8. Multiple choice (medium)

You plan a pretraining run with 100× your past largest run's compute. What is the most reliable way to choose $N$ and $D$?

- **A.** Use the published Chinchilla constants as they are.
- **B.** Train a sweep of small models on your own data and recipe, fit the law, extrapolate, and validate with one mid-size run.
- **C.** Pick the largest model that fits in memory.
- **D.** Copy the size of the latest open model.

<details>
<summary>Answer</summary>

**B.** Train a sweep of small models on your own data and recipe, fit the law, extrapolate, and validate with one mid-size run.

Constants depend on data, tokenizer, optimizer, and architecture. Fitting your own curve on cheap runs, then checking one larger point, is the standard approach.

- **A:** Constants from a different dataset and recipe can mislead badly when extrapolated 100×.
- **C:** Memory limits say nothing about the data you need to train it well.
- **D:** Another lab's model reflects their budget, data, and inference goals, not yours.

</details>

## 9. Select all that apply (medium)

Which statements are correct?

- **A.** The irreducible loss $E$ reflects the inherent unpredictability of the training text.
- **B.** $C \approx 6ND$ includes attention FLOPs that grow with context length.
- **C.** Scaling-law constants change when you change the data or tokenizer.
- **D.** For a mixture-of-experts model, compute scales with active rather than total parameters.

<details>
<summary>Answer</summary>

**A, C, D**

$6ND$ counts the matrix multiplies against the weights; attention score FLOPs are extra and matter at long context.

- **B:** The estimate ignores attention-score FLOPs; they are an extra term proportional to context length.

</details>
