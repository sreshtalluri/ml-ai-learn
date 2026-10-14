<!-- GENERATED from open-models-architecture-walkthroughs.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Open models: architecture walkthroughs

Covers the lesson [Open models: architecture walkthroughs](../lessons/15-llms/07-open-models-architecture-walkthroughs.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/open-models-architecture-walkthroughs/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

A config has `num_hidden_layers: 28`, `num_attention_heads: 28`, `num_key_value_heads: 4`, `hidden_size: 3584`, no `head_dim` field, and every layer global. How many bytes of KV cache does one token need in BF16?

<details>
<summary>Answer</summary>

**57344**

$d = 3{,}584/28 = 128$. Per layer: $2 \cdot 4 \cdot 128 \cdot 2 = 2{,}048$ bytes. Times 28 layers: $57{,}344$ bytes (56 KiB). The 28 query heads do not enter the formula.

</details>

## 2. Calculation (medium)

A layer has $h = 2{,}048$, 16 query heads, 4 KV heads, head size 128, and no biases. How many attention parameters (Q, K, V, output) does it have, in millions? (2 decimals)

<details>
<summary>Answer</summary>

**10.49** (within ±0.01)

Q: $2{,}048 \cdot 16 \cdot 128 = 4{,}194{,}304$. K and V: $2 \cdot 2{,}048 \cdot 4 \cdot 128 = 2{,}097{,}152$. Output: $4{,}194{,}304$. Total $10{,}485{,}760 \approx 10.49$M.

</details>

## 3. Calculation (hard)

A synthetic MoE has shared parameters (attention, norms, embeddings) of 2B and 16 experts per model of 1.5B each (summed across layers), with top-2 routing. How many billion parameters are active per token?

<details>
<summary>Answer</summary>

**5**

Active = shared + $k$ experts $= 2 + 2 \cdot 1.5 = 5$B. Total $= 2 + 16 \cdot 1.5 = 26$B, which sets memory (52 GB in BF16).

</details>

## 4. Multiple choice (medium)

Why do SwiGLU models use an intermediate size near $\tfrac{8}{3}h$ instead of the $4h$ used by GPT-2's MLP?

- **A.** The gated MLP has three matrices instead of two, so $3 \cdot h \cdot \tfrac{8}{3}h = 8h^2$ matches the $2 \cdot h \cdot 4h$ parameters of the GELU MLP.
- **B.** SwiGLU needs a smaller hidden size to stay numerically stable.
- **C.** RoPE requires the MLP width to be a multiple of 3.
- **D.** It halves the KV cache.

<details>
<summary>Answer</summary>

**A.** The gated MLP has three matrices instead of two, so $3 \cdot h \cdot \tfrac{8}{3}h = 8h^2$ matches the $2 \cdot h \cdot 4h$ parameters of the GELU MLP.

Keeping parameters equal makes the comparison fair. Llama 3 rounds up to $3.5h$ for hardware-friendly sizes.

- **B:** Stability isn't the reason; the choice is about parameter budget.
- **C:** RoPE acts on queries and keys in attention, not on the MLP.
- **D:** The MLP doesn't affect the KV cache at all.

</details>

## 5. Multiple choice (hard)

Your parameter calculator gives 9.01B for Gemma 2 9B instead of the published 9.24B. Its config has `hidden_size: 3584`, `num_attention_heads: 16`, `head_dim: 256`. What's the most likely bug?

- **A.** You computed head size as $3{,}584/16 = 224$ instead of using `head_dim = 256`, undercounting every attention projection.
- **B.** You forgot that Gemma 2 is a mixture-of-experts model.
- **C.** You counted the output head separately although embeddings are tied.
- **D.** You used BF16 instead of FP32.

<details>
<summary>Answer</summary>

**A.** You computed head size as $3{,}584/16 = 224$ instead of using `head_dim = 256`, undercounting every attention projection.

With $d = 224$ the Q and output projections are $3{,}584 \times 3{,}584$ instead of $3{,}584 \times 4{,}096$, and K and V shrink too. Always prefer an explicit `head_dim`.

- **B:** Gemma 2 9B is dense.
- **C:** That would overcount, by about 0.9B, not undercount.
- **D:** Precision affects bytes, not the parameter count.

</details>

## 6. Multiple choice (hard)

A team serves a model whose config has `layer_types` alternating sliding (window 4,096) and full attention. At 32k context their engine's KV memory is twice what your calculation predicts. What's the likely cause?

- **A.** The engine doesn't implement the sliding-window layers' smaller cache and allocates full context for every layer.
- **B.** The model secretly uses MHA.
- **C.** RoPE doubles the cache at long context.
- **D.** The tokenizer produces twice as many tokens.

<details>
<summary>Answer</summary>

**A.** The engine doesn't implement the sliding-window layers' smaller cache and allocates full context for every layer.

Half the layers should keep 4,096 tokens instead of 32,768. An engine that treats them as global allocates nearly twice the memory. Check the engine's support for the architecture.

- **B:** `num_key_value_heads` in the config defines the cache; the engine reads the same config.
- **C:** RoPE changes how keys are computed, not how many are stored.
- **D:** Token count affects both your calculation and the engine's the same way.

</details>

## 7. Multiple choice (medium)

You need a model for one 80 GB GPU, serving many concurrent requests with 32k-token contexts, quality similar to a mid-size dense model. Which property should you weight most when comparing candidates of similar size?

- **A.** [object Object]
- **B.** Vocabulary size.
- **C.** Whether the model uses RMSNorm.
- **D.** Total parameter count only.

<details>
<summary>Answer</summary>

**A.** [object Object]

With many long requests, KV cache, not weights, limits concurrency. Two 8B models can differ several times in cache per token.

- **B:** Vocabulary affects embedding size and tokens per text, but not concurrency much.
- **C:** Every current model uses RMSNorm; it doesn't discriminate.
- **D:** Parameter count sets weight memory but ignores the cache, which dominates here.

</details>

## 8. Multiple choice (medium)

What does multi-head latent attention (MLA) store in the KV cache?

- **A.** A small compressed latent per token per layer, plus a shared RoPE key, from which per-head keys and values are reconstructed.
- **B.** Keys and values for every head, in FP8.
- **C.** Only the keys; values are recomputed from scratch.
- **D.** One key-value head shared across all layers.

<details>
<summary>Answer</summary>

**A.** A small compressed latent per token per layer, plus a shared RoPE key, from which per-head keys and values are reconstructed.

DeepSeek-V3 caches $512 + 64 = 576$ values per token per layer, about 71× less than full per-head keys and values for 128 heads.

- **B:** That describes KV quantization, which is orthogonal.
- **C:** Values are reconstructed from the latent like keys; neither is recomputed from the hidden states.
- **D:** Cross-layer sharing is a different technique; MLA caches one latent per layer.

</details>

## 9. Match (easy)

Match each component with what it replaced and why.

| Concept | Options |
|---|---|
| RMSNorm | Multi-head attention: fewer KV heads, smaller cache |
| RoPE | GELU MLP: gated, better loss per parameter |
| SwiGLU | Learned position table: relative positions, context can be extended |
| Grouped-query attention | LayerNorm without mean subtraction or bias: cheaper, equally stable |

<details>
<summary>Answer</summary>

- RMSNorm → LayerNorm without mean subtraction or bias: cheaper, equally stable
- RoPE → Learned position table: relative positions, context can be extended
- SwiGLU → GELU MLP: gated, better loss per parameter
- Grouped-query attention → Multi-head attention: fewer KV heads, smaller cache

These four appear in essentially every current open dense model.

</details>
