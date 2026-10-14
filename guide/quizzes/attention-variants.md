<!-- GENERATED from attention-variants.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Attention variants

Covers the lesson [Attention variants](../lessons/14-transformers/04-attention-variants.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/attention-variants/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

A 16-token sequence uses causal sliding-window attention with window $w = 4$ (the query itself included). How many query-key cells are attended?

<details>
<summary>Answer</summary>

**58**

Rows 0, 1, 2 see 1, 2, 3 keys: $6$. Rows 3 to 15 are 13 rows of 4: $52$. Total $58$. Formula: $nw - w(w-1)/2 = 64 - 6 = 58$, versus 136 for causal.

</details>

## 2. Calculation (medium)

A causal model stacks sliding-window layers with $w = 8$. What is the smallest number of layers after which the last token of a 64-token sequence can be influenced by the first token?

<details>
<summary>Answer</summary>

**9**

Reach after $L$ layers is $L(w-1)+1 = 7L + 1$. We need $7L + 1 \ge 64$, so $L \ge 9$. Eight layers give $57$, nine give $64$.

</details>

## 3. Calculation (hard)

A synthetic model has 48 layers, 8 KV heads, $d_{\text{head}} = 128$, BF16 (2 bytes). For one 65,536-token sequence, 8 layers are global and 40 use a sliding window of 1,024. How many GB ($10^9$ bytes) of KV cache does it need? (2 decimals)

<details>
<summary>Answer</summary>

**2.32** (within ±0.01)

Per token per layer: $2 \cdot 8 \cdot 128 \cdot 2 = 4{,}096$ bytes. Global: $8 \cdot 65{,}536 \cdot 4{,}096 = 2{,}147{,}483{,}648$. Local: $40 \cdot 1{,}024 \cdot 4{,}096 = 167{,}772{,}160$. Sum $2{,}315{,}255{,}808$ bytes $= 2.32$ GB. The 8 global layers carry 93% of it.

</details>

## 4. Multiple choice (medium)

In a 16-token causal mask with local window $w = 2$ plus every key $j$ with $j \bmod 4 = 3$, which keys can query 12 read?

- **A.** {11, 12}
- **B.** {3, 7, 11, 12}
- **C.** {0, 4, 8, 12}
- **D.** {3, 7, 11, 12, 15}

<details>
<summary>Answer</summary>

**B.** {3, 7, 11, 12}

The window gives $\{11, 12\}$. Strided keys with $j \le 12$ are 3, 7, 11. The union is $\{3, 7, 11, 12\}$.

- **A:** That is only the local window; it ignores the strided links.
- **C:** Strided keys satisfy $j \bmod 4 = 3$, not $j \bmod 4 = 0$, and key 11 from the window is missing.
- **D:** Key 15 is in the future of query 12, so a causal mask forbids it.

</details>

## 5. Multiple choice (hard)

An engineer adds a sliding-window boolean mask to a model's attention and passes it to a dense attention call. Training at 4k context is no faster. What is the most likely reason?

- **A.** The mask is applied after the softmax, so it has no effect.
- **B.** A dense kernel still computes all $n^2$ scores and then discards the masked ones; savings need a kernel that skips masked blocks.
- **C.** Sliding window only saves memory during inference, never compute.
- **D.** The window must be a power of two to be fast.

<details>
<summary>Answer</summary>

**B.** A dense kernel still computes all $n^2$ scores and then discards the masked ones; savings need a kernel that skips masked blocks.

Masking with $-\infty$ changes which weights are nonzero, not how many scores are computed. FlexAttention block masks or a sliding-window FlashAttention kernel skip whole blocks and give the speedup.

- **A:** Masks are added to the scores before the softmax; that is why masked keys get zero weight.
- **C:** With a block-skipping kernel, training compute also drops from about $n^2$ to $nw$ per layer.
- **D:** Block sizes matter for kernel efficiency, but any window size gives correct and potentially faster results with the right kernel.

</details>

## 6. Multiple choice (hard)

A team adds a Longformer-style global token at position 0 to a causal decoder: token 0 reads every position, and every position reads token 0. Training loss drops sharply, but generation quality is poor. Diagnose it.

- **A.** The global token leaks future tokens into token 0's representation, which later positions read, so training sees the answers it must predict.
- **B.** The global token increases the KV cache too much.
- **C.** Global tokens cannot be used with RoPE.
- **D.** The model needs a larger window to make use of the global token.

<details>
<summary>Answer</summary>

**A.** The global token leaks future tokens into token 0's representation, which later positions read, so training sees the answers it must predict.

Letting token 0 read the whole sequence breaks causality. Every position reads token 0, so information from future tokens reaches predictions during training. At generation time the future doesn't exist, so the shortcut vanishes. Global tokens belong in encoders, or must stay causal in decoders.

- **B:** One extra row and column adds little memory, and memory doesn't explain a train/generate gap.
- **C:** Positional encoding is independent of which pairs the mask allows.
- **D:** Window size doesn't fix a causality leak.

</details>

## 7. Multiple choice (medium)

You design a 128k-context decoder and must cut KV memory at long context while keeping good recall of single facts far back. Which design fits best?

- **A.** Every layer sliding window with $w = 1{,}024$.
- **B.** Every layer full causal attention.
- **C.** Mostly sliding-window layers, with a fraction of global layers interleaved, and a long-context retrieval evaluation to set the ratio.
- **D.** Full attention with global tokens added.

<details>
<summary>Answer</summary>

**C.** Mostly sliding-window layers, with a fraction of global layers interleaved, and a long-context retrieval evaluation to set the ratio.

Interleaving keeps direct long-range paths in the global layers while local layers keep memory small. Gemma 2 used 1:1 and Gemma 3 uses 5:1 local to global.

- **A:** Smallest cache, but every long-range path needs many hops and single-fact recall suffers.
- **B:** Best recall, but the KV cache grows with the full context in every layer.
- **D:** Adds cost to full attention and does nothing for memory.

</details>

## 8. Select all that apply (medium)

Which statements are correct?

- **A.** With $w \ge n$, causal sliding-window attention is identical to causal attention.
- **B.** FlashAttention is a sparse attention pattern that skips distant tokens.
- **C.** A sliding-window layer only needs to cache keys and values for the last $w$ tokens.
- **D.** Stacking $L$ sliding-window layers lets a token be influenced by up to $L(w-1)+1$ positions.
- **E.** Sliding-window attention costs $O(n^2)$ compute per layer.

<details>
<summary>Answer</summary>

**A, C, D**

FlashAttention computes exact full attention with less memory traffic; it is not a sparsity pattern. Sliding window costs $O(nw)$.

- **B:** FlashAttention is exact; it reorders the computation to avoid writing the score matrix to main memory.
- **E:** Each query scores at most $w$ keys, so the cost is $O(nw)$.

</details>

## 9. Arrange in order (easy)

Order these 16-token masks from fewest to most attended cells.

- Full bidirectional
- Causal
- Local radius 2 + global token 0
- Sliding window w=4
- Local w=2 + stride 4

<details>
<summary>Answer</summary>

1. Local w=2 + stride 4
2. Sliding window w=4
3. Local radius 2 + global token 0
4. Causal
5. Full bidirectional

These are the counts from the lesson's script on a 16-token sequence.

</details>
