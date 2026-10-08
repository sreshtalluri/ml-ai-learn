<!-- GENERATED from self-attention.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Self-attention

Covers the lesson [Self-attention](../lessons/14-transformers/01-self-attention.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/self-attention/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

Scaled scores for one query are $[\ln 3,\; 0]$. What attention weight goes to the first key? (Hint: $e^{\ln 3} = 3$.)

<details>
<summary>Answer</summary>

**0.75** (within ±0.001)

Exponentials are $3$ and $1$, sum $4$. Weights $[3/4, 1/4] = [0.75, 0.25]$.

</details>

## 2. Fill in (medium)

Weights $[0.75, 0.25]$ and values $v_1 = [4, 0]$, $v_2 = [0, 8]$. What is the output vector? Answer as `[a, b]`.

<details>
<summary>Answer</summary>

**[3, 2]** or **[3,2]** or **3,2** or **(3, 2)** or **3, 2**

$0.75 \times [4, 0] + 0.25 \times [0, 8] = [3, 0] + [0, 2] = [3, 2]$.

</details>

## 3. Calculation (easy)

A head has $d_k = 64$. A raw query-key dot product is 24. What is the scaled score?

<details>
<summary>Answer</summary>

**3** (within ±0.001)

$24 / \sqrt{64} = 24 / 8 = 3$.

</details>

## 4. Match (medium)

A sequence of $n = 10$ tokens, $d_{\text{model}} = 32$, one head with $d_k = d_v = 8$. Match each tensor to its shape.

| Concept | Options |
|---|---|
| X (embeddings) | [10, 8] |
| W_Q | [10, 10] |
| Q K^T | [32, 8] |
| softmax(...) V | [10, 32] |

<details>
<summary>Answer</summary>

- X (embeddings) → [10, 32]
- W_Q → [32, 8]
- Q K^T → [10, 10]
- softmax(...) V → [10, 8]

$Q = XW_Q$ is $[10, 8]$; $QK^\top$ is $[10, 8] \times [8, 10] = [10, 10]$; multiplying the weights by $V$ ($[10, 8]$) gives $[10, 8]$.

</details>

## 5. Multiple choice (medium)

Why do decoder-only language models apply a causal mask during training?

- **A.** To reduce memory use by half.
- **B.** So each position predicts the next token using only earlier tokens, matching how generation works at inference.
- **C.** Because softmax cannot handle more than one key.
- **D.** To make attention weights sum to 1.

<details>
<summary>Answer</summary>

**B.** So each position predicts the next token using only earlier tokens, matching how generation works at inference.

Training predicts every next token in parallel. Without the mask, position $t$ could attend to token $t+1$, the very token it must predict, and the model would learn to copy instead of predict.

- **A:** Masked entries are still computed in a naive implementation; the purpose is correctness.
- **B:** Correct.
- **C:** Softmax works over any number of keys.
- **D:** Softmax always produces rows summing to 1, masked or not.

</details>

## 6. Multiple choice (hard)

You increase the context length from 4,000 to 16,000 tokens. Roughly how does the size of each attention score matrix change?

- **A.** It stays the same; it depends only on $d_k$.
- **B.** It grows 4×.
- **C.** It grows 16×.
- **D.** It grows 64×.

<details>
<summary>Answer</summary>

**C.** It grows 16×.

The matrix is $[n, n]$, so it scales with $n^2$: $(16{,}000/4{,}000)^2 = 16$.

- **A:** The score matrix is $n \times n$; $d_k$ affects compute per entry, not the matrix size.
- **B:** That would be linear scaling.
- **C:** Correct.
- **D:** That would be cubic scaling.

</details>

## 7. Arrange in order (easy)

Put the steps of scaled dot-product attention in order.

- Multiply the weights by the values
- Softmax each row into weights
- Apply the mask (for decoders)
- Divide scores by the square root of d_k
- Compute scores as query-key dot products
- Project embeddings into queries, keys, and values

<details>
<summary>Answer</summary>

1. Project embeddings into queries, keys, and values
2. Compute scores as query-key dot products
3. Divide scores by the square root of d_k
4. Apply the mask (for decoders)
5. Softmax each row into weights
6. Multiply the weights by the values

Project, compare, scale, mask, normalize, mix.

</details>

## 8. Reflection (hard)

A team trains a small decoder language model. Training loss drops almost to zero within an hour, but generated text is gibberish. What is the most likely bug, and how would you confirm it?

<details>
<summary>Answer</summary>

**Model answer.** The causal mask is missing or wrong, so each position can attend to the token it is supposed to predict and the model learns to copy it. Confirm by inspecting an attention weight matrix (upper triangle should be exactly zero) or by changing a future token and checking whether earlier positions' logits change. They must not.

Near-zero training loss with useless generation is the signature of label leakage through attention.

</details>
