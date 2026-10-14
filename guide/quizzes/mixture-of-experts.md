<!-- GENERATED from mixture-of-experts.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Mixture of experts

Covers the lesson [Mixture of experts](../lessons/15-llms/05-mixture-of-experts.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/mixture-of-experts/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

A token's router logits for four experts are $[2.0, 1.0, 0.5, -1.0]$. With top-2 routing and gates renormalized over the chosen experts, what is the gate on expert 0? (4 decimals)

<details>
<summary>Answer</summary>

**0.7311** (within ±0.0005)

Softmax: $e^{2} = 7.389$, $e^{1} = 2.718$, $e^{0.5} = 1.649$, $e^{-1} = 0.368$, sum $12.124$. $p_0 = 0.6095$, $p_1 = 0.2242$. Gate $= 0.6095/(0.6095 + 0.2242) = 0.7311$, which equals $\sigma(2.0 - 1.0)$.

</details>

## 2. Calculation (medium)

An MoE layer has 64 experts and top-2 routing. A batch of 4,096 tokens uses capacity factor 1.25. How many slots does each expert get?

<details>
<summary>Answer</summary>

**160**

$Tk = 8{,}192$ assignments; fair share $8{,}192/64 = 128$; capacity $\lceil 1.25 \times 128 \rceil = 160$.

</details>

## 3. Calculation (medium)

12 tokens, 8 experts, top-2, capacity factor 1. The router sends $[8, 1, 2, 4, 1, 3, 3, 2]$ assignments to experts 0 to 7. How many assignments are dropped?

<details>
<summary>Answer</summary>

**6**

Capacity $\lceil 24/8 \rceil = 3$. Expert 0 overflows by $8 - 3 = 5$, expert 3 by $4 - 3 = 1$. Total $6$ of 24.

</details>

## 4. Multiple choice (medium)

A model has 4B shared parameters and 16 experts of 2B each, with top-2 routing. Which statement is correct?

- **A.** It needs memory for 8B parameters and compute for 36B.
- **B.** It needs memory for 36B parameters (72 GB in BF16) and compute for about 8B per token.
- **C.** It needs memory and compute for 8B parameters.
- **D.** It needs memory and compute for 36B parameters.

<details>
<summary>Answer</summary>

**B.** It needs memory for 36B parameters (72 GB in BF16) and compute for about 8B per token.

Every expert must be resident: $4 + 16 \cdot 2 = 36$B. Each token uses the shared part plus two experts: $4 + 2 \cdot 2 = 8$B.

- **A:** [object Object]
- **C:** Experts not chosen for this token are still needed for other tokens, so they must stay in memory.
- **D:** Only the chosen experts are computed per token.

</details>

## 5. Multiple choice (hard)

During MoE training, per-layer logs show 3 of 32 experts receive 70% of tokens and the drop rate is 20%. What is the most likely cause and fix?

- **A.** The learning rate is too low; raise it.
- **B.** Routing collapse; enable or strengthen load balancing (auxiliary loss or bias updates) and monitor per-expert load.
- **C.** Too many experts; reduce to 4.
- **D.** The capacity factor is too high; lower it.

<details>
<summary>Answer</summary>

**B.** Routing collapse; enable or strengthen load balancing (auxiliary loss or bias updates) and monitor per-expert load.

Early favourites get more gradient and get chosen more, a feedback loop. Balancing pushes the router toward even use; drops then fall as well.

- **A:** A higher learning rate tends to speed up collapse rather than fix it.
- **C:** Fewer experts throws away capacity; the problem is the router, not the count.
- **D:** Lowering capacity would increase drops.

</details>

## 6. Multiple choice (medium)

In auxiliary-loss-free balancing (DeepSeek-V3), what does the per-expert bias change?

- **A.** Both which experts are selected and their gate weights.
- **B.** Only which experts are selected; gates come from the unbiased router probabilities.
- **C.** Only the gate weights.
- **D.** The expert weights directly.

<details>
<summary>Answer</summary>

**B.** Only which experts are selected; gates come from the unbiased router probabilities.

Selection uses $z_e + b_e$; gates use the softmax of $z$ alone. The bias is updated by a rule, not by gradients, so it doesn't distort outputs or add a loss term.

- **A:** If the bias changed gates, balancing would directly distort the layer's output.
- **C:** Gates don't change which experts get load, so this couldn't balance anything.
- **D:** Expert weights are trained by the language-modeling loss as usual.

</details>

## 7. Multiple choice (hard)

You must deploy a model on a single 24 GB GPU for a single-user desktop assistant. You can choose a dense 8B model or an MoE with 47B total and 13B active parameters of similar quality per FLOP. Which fits better?

- **A.** The MoE, because it computes only 13B per token.
- **B.** The dense 8B model, because the MoE's 47B parameters need about 94 GB in BF16 (or about 24 GB even at 4 bits), leaving no room for the KV cache.
- **C.** The MoE, because experts not in use can be skipped from memory.
- **D.** Either; memory doesn't depend on architecture.

<details>
<summary>Answer</summary>

**B.** The dense 8B model, because the MoE's 47B parameters need about 94 GB in BF16 (or about 24 GB even at 4 bits), leaving no room for the KV cache.

Memory follows total parameters. On memory-bound edge hardware MoE's advantage disappears.

- **A:** Active compute is low, but the weights still have to fit.
- **C:** Different tokens use different experts, so all must be available; offloading to CPU makes decode very slow.
- **D:** Memory scales with total parameter count, which is far larger for the MoE.

</details>

## 8. Select all that apply (medium)

Which statements are correct?

- **A.** Mixtral 8x7B has about 46.7B parameters, not 56B, because attention and embeddings are shared across experts.
- **B.** With top-2 routing, adding more experts raises per-token compute.
- **C.** Expert parallelism adds all-to-all communication to every MoE layer.
- **D.** The Switch auxiliary loss equals 1 when routing is perfectly uniform.
- **E.** Dropped tokens cause an error and stop training.

<details>
<summary>Answer</summary>

**A, C, D**

Per-token compute depends on $k$, not $E$. Dropped tokens skip the expert and pass through the residual connection silently, which is why you must log the drop rate.

- **B:** Each token still runs exactly 2 experts; more experts adds memory.
- **E:** Drops are silent; that is what makes them dangerous.

</details>

## 9. Arrange in order (easy)

Order the steps of an MoE layer's forward pass for one token.

- Run the chosen experts and sum their outputs weighted by the gates
- Renormalize the chosen probabilities into gates
- Keep the top-k experts
- Softmax the logits into probabilities
- Compute router logits with a linear layer

<details>
<summary>Answer</summary>

1. Compute router logits with a linear layer
2. Softmax the logits into probabilities
3. Keep the top-k experts
4. Renormalize the chosen probabilities into gates
5. Run the chosen experts and sum their outputs weighted by the gates

Router, softmax, top-k, renormalize, weighted sum. Experts outside the top-k are never computed.

</details>
