---
title: Mixture of experts
summary: Route each token to a few of many feed-forward experts, compute gates, capacity, and dropped tokens, balance expert load, and separate total parameters (memory) from active parameters (compute).
skill: llms
minutes: 40
prerequisites: [transformer-architecture, tokenization-and-pretraining]
related: [llm-inference, serving-llms, open-models-architecture-walkthroughs, scaling-laws]
adapted_from: [phases/07-transformers-deep-dive/11-mixture-of-experts]
---

# Mixture of experts

> **Mental model.** A hospital doesn't send every patient to every specialist. A triage nurse (the router) looks at each patient (a token) and sends them to the two most relevant specialists (experts). The hospital employs many specialists (lots of parameters), but each patient only sees two (little compute). The hard part is triage: if everyone gets sent to cardiology, the queue overflows while the other doctors sit idle.

**You will learn to**
- Describe a mixture-of-experts (MoE) layer: router, top-$k$ selection, renormalized gates, weighted sum.
- Compute gates for one token, expert capacity, and the number of dropped assignments.
- Explain load balancing with an auxiliary loss and with a selection bias (auxiliary-loss-free balancing).
- Compute total versus active parameters and say which one sets memory and which sets compute.
- Judge when MoE is worth its serving complexity.

**Why it matters.** Many of the largest open-weight models are MoE: Mixtral 8x7B has about 46.7B parameters but uses about 12.9B per token; DeepSeek-V3 has 671B parameters and activates 37B per token. MoE lets a model store far more knowledge than it computes with on each token. Interviewers ask how routing works, why experts collapse, and what MoE costs in memory and communication.

## 1. Intuition

In a [transformer block](../14-transformers/02-transformer-architecture.md), the feed-forward network (FFN) holds about two-thirds of the parameters, and every token passes through all of them. An MoE layer replaces that one FFN with $E$ smaller or equally sized FFNs, the **experts**, plus a tiny **router**:

1. The router is one linear layer. It gives each token a score (a logit) for every expert.
2. A softmax turns scores into probabilities. The token keeps only the top $k$ experts (often $k = 1$, 2, or 8).
3. The chosen probabilities are rescaled to sum to 1. These are the **gates**.
4. The token goes through its $k$ experts, and the outputs are added, weighted by the gates.

Attention, embeddings, and norms stay shared. Only the FFN is split.

Two numbers now describe the model. **Total parameters** include every expert; all of them must sit in memory. **Active parameters** are what one token actually uses: the shared parts plus $k$ experts. Compute per token follows active parameters, so an MoE model can have the knowledge capacity of a big model at the per-token cost of a small one.

The catch is **load balance**. A freshly initialized router tends to favour a few experts; those get more training signal, get better, and get picked even more. Left alone, most experts go unused. Real systems also give each expert a fixed number of slots per batch, its **capacity**, so they can run as fixed-size matrix multiplies. Tokens routed to a full expert are **dropped** for that expert: they skip it and pass through on the residual connection. So imbalance wastes parameters and loses computation. Every MoE recipe includes some way to push the router toward even load.

## 2. Visualization

<!-- lab:moe-routing -->
![Left: bar chart of token assignments for 8 experts with 512 synthetic tokens and top-2 routing. Without balancing, experts 0 and 7 get 260 and 238 assignments, above the capacity line at 160, while expert 4 gets 39. With bias balancing, every expert gets between 124 and 135. Right: the ratio of maximum to mean load falls from 2.03 to about 1.05 over 40 bias-update steps and then stays near 1.](../../figures/mixture-of-experts.png)

*Synthetic tokens and a synthetic router with a deliberate skew. Bias balancing nudges a per-expert offset that only changes which experts are selected, and the load evens out without touching the loss.*

*Interactive version: [open the lab on the website](https://sreshtalluri.github.io/ml-ai-learn/labs/moe-routing/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Start from the default: 12 tokens, 8 experts, top-2, capacity factor 1. How many slots does each expert get? (3.) How many of the 24 assignments are dropped? (6.)
2. Turn on load balancing. Predict whether drops go to zero. (They fall to 2; the busiest expert drops from 8 requests to 4.)
3. Turn balancing off and raise the capacity factor to 2. Why are there still drops, and what does the extra capacity cost?
4. Move top-$k$ from 2 to 1 and back. What happens to active parameters and to the busiest expert?

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $x$ | one token's hidden state | $[d]$ |
| $E$ | number of routed experts | scalar |
| $k$ | experts chosen per token | scalar |
| $W_r$ | router weights | $[d, E]$ |
| $z = xW_r$ | router logits | $[E]$ |
| $p = \operatorname{softmax}(z)$ | router probabilities | $[E]$ |
| $\mathcal{K}$ | indices of the top $k$ experts | set of size $k$ |
| $g_e$ | gate for chosen expert $e$ | scalar |
| $f_e(x)$ | output of expert $e$ (an FFN) | $[d]$ |
| $T$ | tokens in the batch | scalar |
| $c$ | capacity factor | scalar |
| $C$ | slots per expert | scalar |

### The layer

```math
y = \sum_{e \in \mathcal{K}} g_e\, f_e(x), \qquad g_e = \frac{p_e}{\sum_{j \in \mathcal{K}} p_j}, \qquad \mathcal{K} = \operatorname{top-}k(z)
```

Some models (DeepSeek-V3) add **shared experts** that every token uses: $y = \sum_{s} f_s(x) + \sum_{e \in \mathcal{K}} g_e f_e(x)$.

### Capacity

```math
C = \left\lceil c \cdot \frac{T k}{E} \right\rceil
```

$Tk/E$ is the fair share: the number of assignments each expert would get if routing were perfectly even. The capacity factor $c$ adds headroom.

### Load-balancing loss (Switch Transformer)

Let $F_e$ be the fraction of tokens whose first choice is expert $e$, and $P_e$ the mean router probability for $e$ over the batch:

```math
\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{e=1}^{E} F_e P_e
```

With perfectly even routing, $F_e = P_e = 1/E$, and the sum is $E \cdot E \cdot (1/E^2) = 1$. Imbalance pushes it above 1. $\alpha$ is small (0.01 in [Switch Transformer](https://arxiv.org/abs/2101.03961)) so balancing doesn't overwhelm the language-modeling loss.

**Auxiliary-loss-free balancing** ([Wang et al., 2024](https://arxiv.org/abs/2408.15664), used in DeepSeek-V3) instead keeps a bias $b_e$ per expert, selects with $z_e + b_e$, but computes gates from the unbiased $p$. After each step, overloaded experts get $b_e \mathrel{-}= \gamma$ and underloaded ones $b_e \mathrel{+}= \gamma$. No extra gradient touches the model.

### Worked example: one token

Four experts, $k = 2$, logits $z = [2.0, 1.0, 0.5, -1.0]$.

1. Exponentials: $e^{2.0} = 7.389$, $e^{1.0} = 2.718$, $e^{0.5} = 1.649$, $e^{-1.0} = 0.368$. Sum $= 12.124$.
2. Probabilities: $7.389/12.124 = 0.6095$, $2.718/12.124 = 0.2242$, $0.1360$, $0.0303$.
3. Top-2: experts 0 and 1. Their probabilities sum to $0.6095 + 0.2242 = 0.8337$.
4. Gates: $0.6095/0.8337 = 0.7311$ and $0.2242/0.8337 = 0.2689$.
5. Output: $y = 0.7311\, f_0(x) + 0.2689\, f_1(x)$. Experts 2 and 3 are never computed.

(With $k = 2$ the gates depend only on the logit gap: $\sigma(2.0 - 1.0) = \sigma(1) = 0.7311$.)

### Worked example: capacity and drops

The lab's default: $T = 12$ tokens, $E = 8$, $k = 2$, $c = 1$.

- Assignments: $T k = 24$. Fair share $24/8 = 3$. Capacity $C = \lceil 1 \cdot 3 \rceil = 3$.
- The synthetic router sends requests $[8, 1, 2, 4, 1, 3, 3, 2]$ to experts 0 to 7 (sum 24).
- Overflow: expert 0 has $8 - 3 = 5$ too many, expert 3 has $4 - 3 = 1$. Dropped: $5 + 1 = 6$ of 24 (25%).
- Imbalance: max over mean $= 8/3 = 2.67$.
- With bias balancing the requests become $[3, 4, 4, 2, 3, 3, 3, 2]$: 2 dropped, max over mean $4/3 = 1.33$.
- Raising $c$ to 2 without balancing gives $C = 6$: only expert 0 overflows ($8 - 6 = 2$ dropped), but all 8 experts now reserve 6 slots for 24 assignments, so half the slots ($48 - 24 = 24$) sit empty.

### Worked example: total versus active parameters

A synthetic model with 4B shared parameters (attention, embeddings, router) and 16 experts of 2B each, top-2:

- Total: $4 + 16 \times 2 = 36$B. In BF16 (2 bytes each): $72$ GB, all resident.
- Active: $4 + 2 \times 2 = 8$B per token, about $2 \times 8 \times 10^9 = 16$ GFLOPs per token for a forward pass.

So it computes like an 8B dense model but needs the memory of a 36B one. Real models: Mixtral 8x7B, 8 experts and top-2, about 46.7B total and 12.9B active (attention is shared, so "8x7B" is not 56B). DeepSeek-V3, 1 shared expert plus top-8 of 256 routed, 671B total and 37B active.

## 4. Implementation

### Build it from scratch

The full script is [`code/15-llms/mixture_of_experts.py`](../../code/15-llms/mixture_of_experts.py) (NumPy only, synthetic tokens and weights). It routes 512 tokens through 8 experts with top-2, then measures drops and balances the load.

The router is a matrix multiply, a softmax, and a top-$k$. Gates are renormalized over the chosen experts only. The optional `bias` changes which experts are picked but never the gate values.

```python
def route(X, W_router, k, bias=None):
    logits = X @ W_router                                    # [T, E]
    probs = softmax(logits)
    pick = logits if bias is None else logits + bias         # bias steers selection only
    top = np.argsort(-pick, axis=1)[:, :k]                   # [T, k] chosen experts
    g = np.take_along_axis(probs, top, axis=1)
    return top, g / g.sum(axis=1, keepdims=True), probs      # gates renormalized over the k chosen
```

*Predict:* with $k = 1$, what is every gate? (Exactly 1. Top-1 routing then needs another way to pass a gradient to the router; Switch Transformer multiplies the expert output by the unrenormalized probability $p_e$ for that reason.)

The forward pass groups tokens by expert, so each expert runs one batched multiply over just its tokens. That grouping is what real kernels do, and it is why each expert needs a fixed slot count.

```python
def moe_forward(X, top, gates):
    out = np.zeros_like(X)
    for slot in range(top.shape[1]):
        for e in range(len(experts)):
            rows = top[:, slot] == e                         # tokens sent to expert e in this slot
            out[rows] += gates[rows, slot, None] * (X[rows] @ experts[e])
    return out                                               # [T, d]
```

*Predict:* how many expert multiplies does a token do with $k = 2$ and $E = 8$? (Two, whatever $E$ is. Raising $E$ adds memory, not per-token compute.)

Capacity turns imbalance into dropped work. Tokens claim slots in order; once an expert is full, later tokens skip it.

```python
def dropped(top, E, cf):
    cap = int(np.ceil(cf * top.size / E))                    # slots per expert
    used = np.zeros(E, dtype=int)
    lost = 0
    for t in range(top.shape[0]):                            # tokens claim slots in order
        for e in top[t]:
            if used[e] < cap:
                used[e] += 1
            else:
                lost += 1                                    # this token skips expert e
    return cap, lost
```

*Predict:* because tokens claim slots in order, which tokens get dropped more often, early or late ones in the batch? (Late ones, which is one reason some systems prioritize by gate value instead.)

Two balancing methods. The Switch loss is added to the training loss and pushes router gradients toward even use. The bias method needs no gradient: nudge each expert's selection bias against its load.

```python
def switch_aux_loss(top, probs, E):
    f = np.bincount(top[:, 0], minlength=E) / len(top)       # fraction routed (first choice)
    P = probs.mean(axis=0)                                   # mean router probability
    return E * float(f @ P)                                  # 1.0 when perfectly uniform

bias, gamma, history = np.zeros(E), 0.02, []
for step in range(100):                                      # aux-loss-free balancing (DeepSeek-V3)
    t_b, _, _ = route(X, W_router, k, bias)
    load_b = np.bincount(t_b.ravel(), minlength=E)
    history.append(load_b.max() / load_b.mean())
    bias -= gamma * np.sign(load_b - load_b.mean())          # overloaded down, underloaded up
```

*Predict:* why does the load ratio end up oscillating around 1.05 instead of reaching exactly 1? (A fixed step $\gamma$ overshoots: an expert flips between slightly over and slightly under. A smaller $\gamma$ narrows the band but converges slower.)

### Use the library

You rarely write the router yourself. Hugging Face loads MoE models like any other, and the config tells you the shape:

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
print(cfg.num_local_experts, cfg.num_experts_per_tok)    # 8 experts, top-2
print(cfg.router_aux_loss_coef)                          # Switch-style balancing coefficient
```

Passing `output_router_logits=True` to a Mixtral forward pass returns the router logits per layer, so you can measure expert load on your own data. For serving, vLLM and SGLang implement fused MoE kernels and **expert parallelism**: experts are spread across GPUs and tokens are sent to whichever GPU holds their expert through an all-to-all exchange. What the library adds over the scratch version: grouped GEMM kernels that run all experts in one launch, capacity handling (or dropless routing that sizes buffers dynamically), the communication schedule, and a router z-loss that keeps logits small for numerical stability ([ST-MoE](https://arxiv.org/abs/2202.08906)).

### Run it

From `guide/`:

```bash
uv run code/15-llms/mixture_of_experts.py
```

```text
router probs: [0.6095 0.2242 0.136  0.0303]  chosen: [0 1]  renormalized gates: [0.7311 0.2689]

=== 512 synthetic tokens, 8 experts, top-2 ===
output shape: (512, 16)  requests per expert: [260 101  71 104  39 116  95 238]  (fair share 128)
  capacity factor 1.0 : 128 slots/expert, 242 of 1024 assignments dropped
  capacity factor 1.25: 160 slots/expert, 178 of 1024 assignments dropped
  capacity factor 2.0 : 256 slots/expert,   4 of 1024 assignments dropped

Switch aux loss (1.0 = balanced): 1.345
max/mean load before, after 100 bias updates: 2.03 -> 1.05
requests per expert after: [126 125 135 130 124 127 125 132]
dropped at capacity factor 1.25 after balancing: 0 of 1024

synthetic 4B shared + 16 x 2B experts, top-2: total 36B (72 GB in BF16), active 8B, 16 GFLOPs per token
```

Before balancing, capacity factor 2 is needed to keep drops near zero, which doubles the reserved slots. After balancing, factor 1.25 drops nothing.

## 5. Engineering

**When MoE pays off.** Large-scale training and high-throughput serving, where you want more model quality per FLOP and can afford the memory. At high batch sizes the router spreads tokens over nearly all experts, so the whole model gets used efficiently.

**When it doesn't.** Edge and single-GPU deployment, where memory is the binding constraint: you pay for all experts to compute with a few. At batch size 1, decode reads only the active experts' weights, so it's fast, but you still had to load everything. Small models: the extra complexity often isn't worth it below a few billion parameters.

**Memory and communication.** Memory follows total parameters; quantization (FP8, INT4) is common for MoE serving. With expert parallelism, every MoE layer adds two all-to-all exchanges (send tokens to experts, bring results back), so interconnect bandwidth and latency often dominate. Placement matters: keep the busiest experts spread across GPUs.

**Design choices in current models.**

| Choice | Classic (Switch, GShard, Mixtral) | Fine-grained (DeepSeek-V3) |
|---|---|---|
| Expert size | a full FFN | a fraction of an FFN |
| $E$, $k$ | 8 to 64 experts, top-1 or top-2 | 256 routed, top-8 |
| Shared experts | none | 1, always on |
| Balancing | auxiliary loss | selection bias, plus a tiny sequence-level loss |

**Fine-tuning MoE models.** Routing can drift and collapse on narrow fine-tuning data. Keep the balancing loss on, use a lower learning rate for the router (or freeze it), and monitor per-expert load.

### Common mistakes

- Quoting active parameters as the memory requirement. Memory follows total parameters.
- Reading "8x7B" as 56B: attention and embeddings are shared, so Mixtral is about 46.7B.
- Ignoring dropped tokens: a high drop rate silently weakens the model; log it per layer.
- Turning off the load-balancing term during fine-tuning and watching a few experts take over.
- Assuming MoE decode is as cheap as a dense model of the active size at every batch size; at large batch nearly every expert is read each step.

## 6. Knowledge check

<!-- quiz:mixture-of-experts -->
**[Take the mixture of experts quiz](../../quizzes/mixture-of-experts.md)**
<!-- /quiz -->

**Practice exercise.** A layer has $E = 64$ experts, top-$k = 2$, and processes a batch of $T = 4{,}096$ tokens with capacity factor 1.25. What is the capacity per expert? If one expert receives 300 assignments, how many of its assignments are dropped?

<details>
<summary>Solution</summary>

Assignments $Tk = 4{,}096 \times 2 = 8{,}192$. Fair share $8{,}192/64 = 128$. Capacity $\lceil 1.25 \times 128 \rceil = 160$. That expert drops $300 - 160 = 140$ assignments, 47% of what it was sent. Those tokens get only their other expert's output (or none, if both were full) and otherwise pass through the residual connection.
</details>

**Implementation challenge.** Add shared experts to the script: one always-on expert whose output is added for every token, plus top-$k$ routed experts. Then write `params(shared, per_expert, E, k, n_shared)` and check it on DeepSeek-V3's shape (1 shared plus top-8 of 256 routed): what fraction of the routed expert parameters does each token use?

<details>
<summary>Solution</summary>

```python
def params(shared, per_expert, E, k, n_shared=0):
    total = shared + (E + n_shared) * per_expert
    active = shared + (k + n_shared) * per_expert
    return total, active

# in moe_forward, before the routed loop:
out += X @ shared_expert
```

Each token uses $8/256 = 3.1\%$ of the routed experts, plus the shared expert. Compare Mixtral's $2/8 = 25\%$. Fine-grained routing makes the model much sparser per token, and the shared expert holds knowledge every token needs so routed experts can specialize.
</details>

## Summary

- An MoE layer replaces the FFN with $E$ experts and a router; each token uses its top $k$, weighted by renormalized gates.
- Total parameters set memory; active parameters (shared plus $k$ experts) set compute per token.
- Capacity $C = \lceil c\,Tk/E \rceil$ caps each expert; overflow is dropped, so imbalance costs quality.
- Balancing uses an auxiliary loss (Switch) or a selection bias that leaves the loss untouched (DeepSeek-V3).
- Serving MoE means expert parallelism, all-to-all communication, and memory for every expert.

**Next:** [Scaling laws](06-scaling-laws.md)

**Related:** [LLM inference](../18-llm-inference/01-llm-inference.md) · [Serving LLMs](../18-llm-inference/02-serving-llms.md) · [Open models: architecture walkthroughs](07-open-models-architecture-walkthroughs.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Mixture of Experts (MoE)](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/11-mixture-of-experts).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Walk me through a mixture-of-experts layer.</strong></summary>

It replaces the feed-forward block of a transformer layer; attention stays shared. A router, which is just a linear layer, produces one logit per expert for each token. A softmax turns those into probabilities, the token keeps its top k experts, and their probabilities are renormalized to sum to 1 to form gates. The token runs through only those k expert FFNs and the outputs are summed with the gate weights. For example, logits 2.0, 1.0, 0.5, and −1.0 with top-2 give gates 0.73 and 0.27 on the first two experts; the other two are never computed. In practice tokens are grouped by expert so each expert does one batched multiply, and each expert has a capacity, a fixed number of slots per batch. Some models add shared experts that every token uses. The result: parameters scale with the number of experts, compute per token scales with k.

</details>

<details>
<summary><strong>Why do MoE models need load balancing, and how do the main methods differ?</strong></summary>

The router is learned, and early on it tends to favour a few experts. Those experts get more gradient, improve, and get picked even more, so many experts end up unused, which wastes parameters. With fixed capacity, overloaded experts also drop tokens, which hurts quality. Switch Transformer adds an auxiliary loss, α times E times the sum over experts of the fraction of tokens routed there times the mean router probability. It equals 1 when routing is uniform and rises with imbalance. It works, but it's a second objective that can pull against language modeling, and α needs tuning. DeepSeek-V3's auxiliary-loss-free method adds a per-expert bias used only for selection: overloaded experts get their bias lowered and underloaded ones raised after each step. Gates still come from the unbiased probabilities, so balancing doesn't distort the model's outputs or add gradients.

</details>

<details>
<summary><strong>After fine-tuning an MoE model on a narrow domain, quality drops and two experts handle most tokens. What happened and what do you do?</strong></summary>

The router collapsed. Narrow data looks similar to the router, so it sends most tokens to the same few experts, and those experts get almost all the updates. With capacity limits, overloaded experts also drop tokens, so many tokens get less computation than in pretraining. First, measure: log per-layer expert load and drop rate on validation data before and after fine-tuning. Fixes include keeping the load-balancing loss or bias updates active during fine-tuning, freezing the router or giving it a much lower learning rate, raising the capacity factor or using dropless routing, mixing some general data into the fine-tuning set, and using LoRA on the experts instead of full updates. Then compare against the base model on both the domain evaluation and a general one.

</details>

<details>
<summary><strong>A synthetic MoE has 4B shared parameters and 16 experts of 2B, top-2. Can it serve on one 80 GB GPU in BF16?</strong></summary>

Total parameters are 4 plus 16 times 2, which is 36B, and all of them must be resident, so weights alone take 72 GB in BF16. That leaves about 8 GB on an 80 GB GPU for the KV cache, activations, and runtime overhead, which is too little for real traffic at any useful context length or batch. Active parameters are 4 plus 2 times 2, which is 8B, about 16 GFLOPs per token, so compute looks like an 8B dense model, but memory looks like a 36B one. Options: quantize weights to FP8, which brings them to 36 GB and leaves room for cache; split across two GPUs with expert or tensor parallelism; or choose a dense 8B model if memory is the main constraint. I'd decide on measured throughput and quality at the target context length.

</details>
