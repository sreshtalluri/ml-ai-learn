<!-- GENERATED from transformer-architecture.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: The transformer architecture

Covers the lesson [The transformer architecture](../lessons/14-transformers/02-transformer-architecture.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/transformer-architecture/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

How many parameters (with biases) does a feed-forward sublayer have with $d = 256$ and $d_{\text{ff}} = 1024$?

<details>
<summary>Answer</summary>

**525568**

$2 \times 256 \times 1024 + 1024 + 256 = 524{,}288 + 1{,}280 = 525{,}568$.

</details>

## 2. Calculation (medium)

How many parameters (with biases) do the Q, K, V, and O projections have together with $d = 256$?

<details>
<summary>Answer</summary>

**263168**

$4 \times 256^2 + 4 \times 256 = 262{,}144 + 1{,}024 = 263{,}168$.

</details>

## 3. Multiple choice (medium)

What does a residual connection $x + F(x)$ buy you in a deep transformer?

- **A.** It halves the parameters.
- **B.** Gradients flow directly through the addition, and each block only needs to learn a change to its input.
- **C.** It removes the need for attention.
- **D.** It applies the causal mask.

<details>
<summary>Answer</summary>

**B.** Gradients flow directly through the addition, and each block only needs to learn a change to its input.

Residual paths make deep stacks trainable.

</details>

## 4. Match (easy)

Match each task to the most natural transformer family.

| Concept | Options |
|---|---|
| Producing sentence embeddings for search | Encoder-decoder |
| Open-ended chat responses | Decoder-only |
| Translating English to German | Encoder-only |

<details>
<summary>Answer</summary>

- Producing sentence embeddings for search → Encoder-only
- Open-ended chat responses → Decoder-only
- Translating English to German → Encoder-decoder

Bidirectional understanding, causal generation, or input-to-output transformation.

</details>

## 5. Multiple choice (medium)

In cross-attention inside an encoder-decoder model, where do queries, keys, and values come from?

- **A.** All from the encoder
- **B.** Queries from the decoder; keys and values from the encoder outputs
- **C.** Queries from the encoder; keys and values from the decoder
- **D.** All from the decoder

<details>
<summary>Answer</summary>

**B.** Queries from the decoder; keys and values from the encoder outputs

Each decoder position asks questions of the full encoded input.

</details>

## 6. Reflection (hard)

You double a model's context length from 8k to 16k tokens. Which costs grow, and roughly how?

<details>
<summary>Answer</summary>

**Model answer.** Weights don't change. Attention-score computation during prefill grows about 4× (quadratic in length), feed-forward compute grows about 2× (linear), and the KV cache memory per sequence doubles (linear), which often limits batch size. Per generated token with a KV cache, cost grows roughly linearly with the context already processed.

Separate quadratic attention terms from linear feed-forward and KV-cache terms.

</details>
