<!-- GENERATED from fine-tuning-in-practice.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Fine-tuning in practice

Covers the lesson [Fine-tuning in practice](../lessons/15-llms/04-fine-tuning-in-practice.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/fine-tuning-in-practice/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

LoRA with rank $r = 16$ is applied to the gate, up, and down projections of every layer in a synthetic model with $d = 4096$, $d_{ff} = 11008$, and 32 layers. Attention is not targeted. How many trainable parameters?

<details>
<summary>Answer</summary>

**23199744**

Each MLP matrix: $16 \times (4096 + 11008) = 16 \times 15{,}104 = 241{,}664$. Three per layer: $724{,}992$. Times 32 layers: $23{,}199{,}744$.

</details>

## 2. Calculation (medium)

Full fine-tuning with mixed-precision AdamW (bf16 weights and gradients, fp32 master weights and two fp32 Adam moments). How many GB (decimal) are needed for a 3-billion-parameter model, before activations?

<details>
<summary>Answer</summary>

**48** (within ±0.5)

$2 + 2 + 4 + 4 + 4 = 16$ bytes per parameter. $16 \times 3 \times 10^9 = 48 \times 10^9$ bytes $= 48$ GB.

</details>

## 3. Calculation (hard)

DPO with $\beta = 0.1$. For one pair, the chosen response's log-ratio $\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ is $+1.5$ and the rejected one's is $-0.5$. What is the loss? (3 decimals)

<details>
<summary>Answer</summary>

**0.598** (within ±0.002)

Argument $= 0.1 \times (1.5 - (-0.5)) = 0.2$. $\sigma(0.2) = 1/(1 + e^{-0.2}) = 1/1.8187 = 0.5498$. Loss $= -\ln 0.5498 = 0.598$.

</details>

## 4. Multiple choice (medium)

An SFT run on (prompt, answer) pairs produces a model that sometimes continues by writing a new user question after its answer. Prompts are about 4 times longer than answers. What is the most likely data bug?

- **A.** The learning rate is too low.
- **B.** The loss is computed on prompt tokens as well as response tokens, and/or the end-of-turn token is missing from the targets.
- **C.** LoRA rank is too small.
- **D.** The model needs DPO instead of SFT.

<details>
<summary>Answer</summary>

**B.** The loss is computed on prompt tokens as well as response tokens, and/or the end-of-turn token is missing from the targets.

Without masking, most of the gradient teaches the model to write user turns. Without an EOS or end-of-turn target, it never learns to stop. Both produce this symptom.

- **A:** A low learning rate would make the model change less, not learn to write user turns.
- **B:** Correct.
- **C:** Rank limits capacity, it doesn't create this specific behavior.
- **D:** DPO shapes preferences; it doesn't fix a masking bug in the SFT data.

</details>

## 5. Multiple choice (medium)

A LoRA run on a 13B model in bf16 is out of memory on a 24 GB GPU. Which change frees the most memory?

- **A.** Halve the LoRA rank from 16 to 8.
- **B.** Load the frozen base in 4-bit NF4 (QLoRA).
- **C.** Switch from AdamW to SGD for the adapters.
- **D.** Remove the v projection from the target modules.

<details>
<summary>Answer</summary>

**B.** Load the frozen base in 4-bit NF4 (QLoRA).

The frozen bf16 base is about 26 GB by itself. NF4 brings it to about 7 GB. The adapter-related memory (rank, optimizer, targets) is well under 1 GB, so changing it barely helps.

- **A:** Adapter state is a few hundred MB; halving it saves very little.
- **B:** Correct.
- **C:** Optimizer state for adapters is already tiny.
- **D:** Same as rank, a small fraction of a small number.

</details>

## 6. Select all that apply (hard)

A fine-tuned model improved on the target task but lost general reasoning ability. Which changes are reasonable mitigations? Select all that apply.

- **A.** Mix a slice of general instruction data into the training set.
- **B.** Train for more epochs so the model converges.
- **C.** Lower the learning rate and use fewer epochs.
- **D.** Gate releases on a general-capability regression suite.
- **E.** Raise the LoRA rank to the maximum.

<details>
<summary>Answer</summary>

**A, C, D**

Replay data, gentler training, and regression gates address forgetting. More epochs and more capacity usually make it worse.

- **A:** Replay of general data is a standard mitigation.
- **B:** More epochs push further from the base model and increase forgetting.
- **C:** Smaller updates keep the model closer to the base.
- **D:** Gating catches regressions before users do.
- **E:** More capacity lets the model move further from its base, which typically increases forgetting.

</details>

## 7. Calculation (medium)

GRPO samples a group of 4 answers with verifier rewards $[1, 1, 1, 0]$. Using population standard deviation, what is the advantage of the failing answer? (3 decimals)

<details>
<summary>Answer</summary>

**-1.732** (within ±0.002)

Mean $= 0.75$. Deviations: $0.25, 0.25, 0.25, -0.75$. Variance $= (3 \times 0.0625 + 0.5625)/4 = 0.75/4 = 0.1875$. Std $= 0.4330$. Advantage $= -0.75 / 0.4330 = -1.732$.

</details>

## 8. Multiple choice (medium)

You have 20,000 prompts for SQL generation and an executor that checks whether each query returns the expected rows. The SFT model already writes valid SQL most of the time. What is a natural next step?

- **A.** DPO on human-written preference pairs you would need to collect.
- **B.** RL with verifiable rewards (for example GRPO) using the executor as the reward.
- **C.** Full fine-tuning on the prompts without answers.
- **D.** Increase decoding temperature.

<details>
<summary>Answer</summary>

**B.** RL with verifiable rewards (for example GRPO) using the executor as the reward.

A programmatic checker is exactly what RLVR needs. Sample several queries per prompt, reward the ones that return the right rows, and push their probability up.

- **A:** Possible, but you would have to build preference data when you already have a reward.
- **B:** Correct.
- **C:** There are no targets to learn from without answers.
- **D:** Temperature changes sampling, not the model's skill.

</details>

## 9. Reflection (medium)

Why is LoRA's $B$ matrix initialized to zero while $A$ is random, and what happens at inference after merging?

<details>
<summary>Answer</summary>

**Model answer.** With $B = 0$, the update $BA$ is zero, so training starts exactly at the pretrained model with no random perturbation. $A$ is random so that the gradient with respect to $B$ is non-zero and learning can start. After training, $W_0 + \frac{\alpha}{r}BA$ is computed once and stored, so the merged model has the same shape and latency as the base.

Zero init preserves the starting point; random A breaks symmetry; merging removes inference overhead.

</details>

## 10. Multiple choice (easy)

Which requirement is the weakest reason to fine-tune?

- **A.** The model must answer with today's inventory levels.
- **B.** A small model must match a large model's output on one narrow classification task.
- **C.** Outputs must always follow a company's tone and structure.
- **D.** Latency must drop by replacing long few-shot prompts.

<details>
<summary>Answer</summary>

**A.** The model must answer with today's inventory levels.

Changing facts belong in retrieval or tool calls. Fine-tuning shapes behavior, not live data.

- **A:** Inventory changes constantly; weights would be stale immediately.
- **B:** Distillation into a small model is a good use of fine-tuning.
- **C:** Style and structure are behaviors that fine-tuning shapes well.
- **D:** Fine-tuning can bake in behavior that long prompts were providing, saving tokens.

</details>
