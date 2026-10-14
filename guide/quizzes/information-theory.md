<!-- GENERATED from information-theory.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Information theory

Covers the lesson [Information theory](../lessons/00-foundations/05-information-theory.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/information-theory/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What does the entropy of a distribution measure?

- **A.** The number of bits needed to store one probability
- **B.** The error rate of a classifier
- **C.** The average surprise, or uncertainty, of outcomes drawn from it
- **D.** How far the distribution is from the uniform distribution

<details>
<summary>Answer</summary>

**C.** The average surprise, or uncertainty, of outcomes drawn from it

$H(P) = -\sum_x p(x)\log p(x)$ is the expected surprise. It is highest for a uniform distribution and 0 when one outcome is certain.

- **A:** Entropy is a property of the whole distribution, not a storage size for numbers.
- **B:** Error rate is a metric of a model on data; entropy needs no model.
- **C:** Correct.
- **D:** Related (entropy is maximal at uniform), but the distance to uniform is a KL divergence, $\log K - H(P)$, not the entropy itself.

</details>

## 2. Calculation (medium)

A coin lands heads with probability 0.75. What is its entropy in bits? (Give 3 decimals.)

<details>
<summary>Answer</summary>

**0.811** (within ±0.002)

$H = -(0.75\log_2 0.75 + 0.25\log_2 0.25) = -(0.75 \times -0.4150 + 0.25 \times -2) = 0.3113 + 0.5 = 0.811$ bits.

</details>

## 3. Multiple choice (medium)

Why is minimizing cross-entropy during training equivalent to minimizing KL divergence from the model to the data?

- **A.** They are the same formula
- **B.** $H(P, Q) = H(P) + D_{\text{KL}}(P \Vert Q)$, and $H(P)$ of the data does not depend on the model
- **C.** KL divergence is always 0 at convergence
- **D.** The model's entropy is 0 at convergence

<details>
<summary>Answer</summary>

**B.** $H(P, Q) = H(P) + D_{\text{KL}}(P \Vert Q)$, and $H(P)$ of the data does not depend on the model

The entropy of the labels is a constant during training, so any reduction in cross-entropy is a reduction in KL.

- **A:** They differ by $H(P)$, which is positive whenever labels are noisy or soft.
- **B:** Correct.
- **C:** On noisy data the model can't match $P$ exactly; KL shrinks but need not reach 0. The equivalence holds at every step anyway.
- **D:** The model's own entropy is not part of the identity; it's the data's entropy that cancels.

</details>

## 4. Multiple choice (easy)

A language model has perplexity 50 on a test set. What does that mean?

- **A.** It has 50 layers
- **B.** It makes 50 errors per sentence
- **C.** On average it is as uncertain as choosing uniformly among 50 tokens at each step
- **D.** It was trained for 50 epochs

<details>
<summary>Answer</summary>

**C.** On average it is as uncertain as choosing uniformly among 50 tokens at each step

Perplexity is $e^{\text{mean loss}}$. A uniform guess over 50 tokens has loss $\ln 50$ and perplexity exactly 50.

- **A:** Perplexity is computed from predicted probabilities, not architecture.
- **B:** It doesn't count errors; a model can be right with low confidence or wrong with low perplexity elsewhere.
- **C:** Correct.
- **D:** Training length doesn't appear in the formula.

</details>

## 5. Calculation (medium)

A model's mean cross-entropy on a test set is 2.0 nats per token. What is its perplexity? (Give 2 decimals.)

<details>
<summary>Answer</summary>

**7.39** (within ±0.01)

$\text{PPL} = e^{2.0} = 7.389$.

</details>

## 6. Calculation (medium)

Logits are $[2.0, 1.0, 0.1]$ and the true class is 0. What is the cross-entropy loss in nats? (Give 3 decimals.)

<details>
<summary>Answer</summary>

**0.417** (within ±0.002)

Subtract the max: $[0, -1, -1.9]$. Exponentiate: $[1, 0.3679, 0.1496]$, sum $1.5175$. $q_0 = 1/1.5175 = 0.6590$. Loss $= -\ln 0.6590 = 0.417$.

</details>

## 7. Multiple choice (medium)

A freshly initialized 10-class classifier reports a step-0 loss of 9.4 nats. What is the most likely explanation?

- **A.** This is normal; losses start high
- **B.** The final layer produces large logits, so the model is confidently wrong before training
- **C.** The learning rate is too low
- **D.** The model is overfitting

<details>
<summary>Answer</summary>

**B.** The final layer produces large logits, so the model is confidently wrong before training

A near-uniform model should start at $\ln 10 = 2.303$. A loss of 9.4 means the true class gets about $e^{-9.4} \approx 0.00008$, so the initial logits are far too large (or labels are misaligned).

- **A:** The expected start is 2.303 nats, not 9.4; the gap is a signal.
- **B:** Correct.
- **C:** The learning rate hasn't acted yet at step 0.
- **D:** No training has happened, so there is nothing to overfit.

</details>

## 8. Multiple choice (hard)

$P = [1, 0]$ and $Q = [0.5, 0.5]$. Which statement is true?

- **A.** $D_{\text{KL}}(P \Vert Q) = D_{\text{KL}}(Q \Vert P)$
- **B.** $D_{\text{KL}}(P \Vert Q) = \ln 2$ and $D_{\text{KL}}(Q \Vert P)$ is infinite
- **C.** Both are infinite
- **D.** Both are 0

<details>
<summary>Answer</summary>

**B.** $D_{\text{KL}}(P \Vert Q) = \ln 2$ and $D_{\text{KL}}(Q \Vert P)$ is infinite

$D(P\Vert Q) = 1 \cdot \ln(1/0.5) = \ln 2 = 0.693$. $D(Q\Vert P)$ has a term $0.5\ln(0.5/0)$, which is infinite: $Q$ puts mass where $P$ says impossible.

- **A:** KL is asymmetric, and this pair shows it starkly.
- **B:** Correct.
- **C:** $D(P\Vert Q)$ only sums where $p > 0$, and there $q = 0.5$, so it is finite.
- **D:** The distributions differ, so neither direction is 0.

</details>

## 9. Multiple choice (medium)

You want to screen features for a label where some relationships are U-shaped. Why prefer mutual information over Pearson correlation?

- **A.** MI is cheaper to compute
- **B.** MI is 0 only under independence, so it catches non-linear and non-monotonic dependence; correlation can be 0 for a U-shape
- **C.** Correlation only works on categorical data
- **D.** They always rank features the same way

<details>
<summary>Answer</summary>

**B.** MI is 0 only under independence, so it catches non-linear and non-monotonic dependence; correlation can be 0 for a U-shape

$y = x^2$ with $x$ symmetric around 0 has correlation 0 but high mutual information.

- **A:** MI needs density estimation or binning and is usually more expensive and noisier.
- **B:** Correct.
- **C:** Pearson correlation is defined for numeric variables; it's MI that handles categorical ones naturally.
- **D:** They disagree exactly on non-linear relationships like the U-shape.

</details>

## 10. Multiple choice (hard)

Model A (tokenizer with 32k tokens) has test perplexity 12; model B (tokenizer with 100k tokens) has 15 on the same text. What is the right conclusion?

- **A.** A is better because its perplexity is lower
- **B.** B is better because its vocabulary is larger
- **C.** You can't tell yet; normalize to a tokenizer-independent unit like bits per byte first
- **D.** They are equivalent because both are below 20

<details>
<summary>Answer</summary>

**C.** You can't tell yet; normalize to a tokenizer-independent unit like bits per byte first

Perplexity is per token. Different tokenizers split the same text into different numbers of tokens, so per-token numbers aren't comparable. Total loss divided by bytes (or words) is.

- **A:** Lower per-token perplexity can come from cutting text into more, easier tokens.
- **B:** Vocabulary size alone says nothing about prediction quality.
- **C:** Correct.
- **D:** There is no such threshold; the comparison itself is invalid without normalization.

</details>
