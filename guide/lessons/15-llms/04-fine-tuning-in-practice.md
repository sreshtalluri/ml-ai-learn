---
title: Fine-tuning in practice
summary: Run the post-training pipeline end to end (SFT data, loss masking, LoRA and QLoRA, DPO, GRPO-style RL) and size the GPU memory a fine-tuning job needs before you launch it.
skill: llms
minutes: 45
prerequisites: [adapting-llms, tokenization-and-pretraining, gradient-descent]
related: [transformer-architecture, llm-evaluation, training-and-regularization]
---

# Fine-tuning in practice

**Mental model.** Fine-tuning is a short, targeted continuation of training. The base model already knows language and the world. You show it a small, clean set of examples of the behavior you want, and you nudge its weights toward that behavior without wrecking what it already does well. Most of the engineering is in three places: the data, the memory budget, and the regression tests.

**You will learn to**
- Describe the modern post-training pipeline: pretraining, supervised fine-tuning (SFT), preference tuning, and reinforcement learning with verifiable rewards.
- Build SFT examples with a chat template and mask the loss so only response tokens are trained.
- Compute the GPU memory of full fine-tuning, LoRA, and QLoRA from bytes per parameter.
- Compute LoRA's trainable parameters for any set of target matrices, and explain its initialization and merging.
- Explain the DPO loss and GRPO advantages with numbers, and when each is used.
- Detect catastrophic forgetting with regression evals, and recognize when fine-tuning is the wrong tool.

**Why it matters.** The [adapting LLMs](03-adapting-llms.md) lesson answered *whether* to fine-tune. This lesson is for the day you decide yes. A fine-tuning run that does not fit on your GPU, trains on the prompt instead of the answer, or silently breaks the model's general skills wastes days. Interviewers ask about memory math, LoRA details, and DPO because they separate people who have run these jobs from people who have read about them.

## 1. Intuition

**The post-training pipeline.** Modern assistants are built in stages. Each stage uses a different kind of data and a different loss.

1. **Pretraining.** Next-token prediction on trillions of tokens of web, code, and books. The model learns language, facts, and reasoning patterns. It produces continuations, not answers.
2. **Supervised fine-tuning (SFT).** Train on (prompt, ideal response) pairs, often tens of thousands to a few million. The model learns the assistant format: follow instructions, use the chat template, stop at the right place.
3. **Preference tuning.** Train on (prompt, better response, worse response) triples. The model learns what humans (or a judge model) prefer: helpfulness, tone, refusals. RLHF with a reward model and PPO was the original recipe. [DPO](../../glossary.md#dpo) and its variants do it with a simple classification-style loss.
4. **Reinforcement learning with verifiable rewards (RLVR).** For tasks with a checkable answer (math with a known result, code with unit tests, a JSON output that must parse), sample several answers, score them with a program, and push up the probability of the ones that pass. GRPO is a popular algorithm here. This stage is where much of the recent progress on step-by-step reasoning has come from.

As an application engineer you usually start from a model that has already been through all four stages. Your own fine-tuning is typically a small SFT run, sometimes followed by DPO on your preference data, sometimes RL against your own verifier.

**SFT data is a product.** The model learns exactly what the examples show, including their mistakes. A few thousand clean, diverse, correctly formatted examples usually beat a much larger set of noisy ones. Duplicates teach the model to overweight those cases. Examples that overlap your evaluation set make your scores meaningless (contamination).

**Why memory is the first wall.** Training needs far more memory than inference. For every weight you train, the optimizer keeps extra state. Full fine-tuning of a 7B model needs roughly 16 bytes per parameter before activations: over 100 GB. LoRA freezes the base model and trains small add-on matrices, so the extra state shrinks to well under 1 GB. [QLoRA](../../glossary.md#qlora) also stores the frozen base in 4 bits, so a 7B model fits on a 24 GB consumer GPU.

**Forgetting.** Push a model hard on a narrow task and it can lose skills it had before: general reasoning, other languages, safety behavior. This is [catastrophic forgetting](../../glossary.md#catastrophic-forgetting). You only notice it if you test for it.

## 2. Visualization

<!-- lab:lora -->
![Left: horizontal stacked bars of estimated training memory for a synthetic 7B-style model. Full fine-tuning totals about 108.6 GB, dominated by optimizer states (purple), well past the 80 GB line. LoRA with a 16-bit base totals about 14.9 GB, almost all frozen weights. QLoRA with a 4-bit base totals about 5.3 GB. Right: log-log plot of LoRA trainable parameters as a share of the model against rank, two parallel lines for q,k,v,o only and for all seven linear layers, both rising linearly with rank.](../../figures/fine-tuning-in-practice.png)
*Synthetic 7B-style configuration (hidden size 4096, 32 layers, MLP width 11008, vocabulary 32000), 2,048 tokens per micro-batch, gradient checkpointing on. These are arithmetic estimates of training state, not measurements; real jobs add framework overhead.*
*Interactive version: [open the lab on the website](https://sreshtalluri.github.io/ml-ai-learn/labs/lora/).*
<!-- /lab -->

**Try it** (predict first, then check):
1. On the 7B preset, compare the full fine-tune bar with the LoRA bar below it. Predict which segment shrinks most. Then check whether it was weights, gradients, or optimizer states.
2. Double the LoRA rank from 16 to 32. Predict the change in trainable parameters and in total memory. Why does total memory barely move?
3. Pick the 70B preset with the 48 GB GPU line. Predict which of the three methods fits. Then turn off gradient checkpointing and raise the tokens per micro-batch.
4. Change the targets from "all linear" to "q, v only". Predict the drop in trainable parameters for the 7B preset before you look.

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $N$ | total model parameters | scalar |
| $d$ | hidden size (model width) | scalar |
| $d_{ff}$ | MLP intermediate width | scalar |
| $L$ | number of transformer layers | scalar |
| $W_0$ | frozen pretrained weight matrix | $[d_{\text{out}}, d_{\text{in}}]$ |
| $A$ | LoRA down-projection, random init | $[r, d_{\text{in}}]$ |
| $B$ | LoRA up-projection, zero init | $[d_{\text{out}}, r]$ |
| $r$ | LoRA rank | scalar |
| $\alpha$ | LoRA scaling hyperparameter | scalar |
| $x$ | prompt tokens | sequence |
| $y_w, y_l$ | chosen (preferred) and rejected responses | sequences |
| $\pi_\theta, \pi_{\text{ref}}$ | model being trained, frozen reference model | distributions over tokens |
| $\beta$ | DPO strength of the pull toward $\pi_{\text{ref}}$ | scalar |
| $m_t$ | loss mask on token $t$ (1 = response, 0 = prompt) | $\{0,1\}$ |

### SFT loss with masking

An SFT example is one sequence: the prompt rendered in the model's **chat template** (special tokens that mark system, user, and assistant turns) followed by the response. The loss is next-token cross-entropy, but only on response tokens:

```math
\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_t m_t}\sum_{t} m_t \log \pi_\theta(\text{token}_t \mid \text{token}_{<t})
```

With a 300-token prompt and a 100-token response, the mask has 300 zeros then 100 ones, so the loss averages over 100 tokens. Without the mask, three quarters of the gradient would teach the model to write user prompts. In PyTorch libraries the mask is implemented by setting the label to `-100` for prompt positions.

### Full fine-tuning memory

With mixed-precision AdamW, each trained parameter costs:

| Item | Bytes per parameter |
|---|---|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master copy of the weights | 4 |
| Adam first moment $m$ (fp32) | 4 |
| Adam second moment $v$ (fp32) | 4 |
| **Total** | **16** |

Plus activations, which scale with tokens per micro-batch rather than with $N$. Optimizer states are 12 of the 16 bytes. That is why sharding them across GPUs (ZeRO, FSDP) or storing them in 8 bits are the main levers for full fine-tuning.

### LoRA

```math
W = W_0 + \frac{\alpha}{r} B A, \qquad \text{trainable parameters per matrix} = r\,(d_{\text{in}} + d_{\text{out}})
```

- **Initialization.** $A$ is random and $B$ is zero, so $BA = 0$ and step 0 is exactly the pretrained model. Gradients still flow into $B$ because $A$ is non-zero.
- **Scaling.** $\alpha/r$ keeps the update's size roughly stable when you change $r$, so you can change rank without fully retuning the learning rate. Common settings are $\alpha = r$ or $\alpha = 2r$. (rsLoRA uses $\alpha/\sqrt{r}$ for high ranks.)
- **Which matrices.** Each transformer layer has seven linear maps: attention $q, k, v, o$ and MLP gate, up, down. Targeting all of them usually adapts better than attention-only, at a few times the trainable parameters.
- **Merging.** At deployment, compute $W_0 + \frac{\alpha}{r}BA$ once. The merged model has the same shape and speed as the base. Or keep adapters separate and hot-swap many small adapters on one shared base.

### QLoRA

QLoRA stores the frozen base weights in **NF4**, a 4-bit format whose 16 levels are placed at quantiles of a normal distribution (pretrained weights are roughly normal). Each block of 64 weights shares one scale; double quantization compresses those scales too, for about 4.127 bits per weight in total. In the forward pass, each block is dequantized to bf16 on the fly and multiplied. The LoRA matrices $A, B$ stay in 16 bits and are the only things trained. Gradients flow *through* the quantized base to reach them.

### Worked example: a synthetic 7B-style model

Configuration (synthetic, Llama-like shape): $d = 4096$, $L = 32$, $d_{ff} = 11008$, vocabulary 32000, full multi-head attention so $k$ and $v$ are $4096 \times 4096$.

**Step 1: parameters per layer.**

| Matrix | Shape | Parameters |
|---|---|---|
| $q, k, v, o$ (each) | $4096 \times 4096$ | $16{,}777{,}216$ |
| gate, up (each) | $4096 \times 11008$ | $45{,}088{,}768$ |
| down | $11008 \times 4096$ | $45{,}088{,}768$ |

Per layer: $4 \times 16{,}777{,}216 + 3 \times 45{,}088{,}768 = 67{,}108{,}864 + 135{,}266{,}304 = 202{,}375{,}168$.
All layers: $202{,}375{,}168 \times 32 = 6{,}476{,}005{,}376$.
Embedding plus untied output head: $2 \times 32000 \times 4096 = 262{,}144{,}000$.
Total: $N = 6{,}476{,}005{,}376 + 262{,}144{,}000 = 6{,}738{,}149{,}376 \approx 6.74$ billion. (Norm weights add about 0.3 million; ignored.)

**Step 2: LoRA with $r = 16$ on all seven linear layers.**
- $q, k, v, o$: $16 \times (4096 + 4096) = 131{,}072$ each, so $4 \times 131{,}072 = 524{,}288$.
- gate, up, down: $16 \times (4096 + 11008) = 16 \times 15{,}104 = 241{,}664$ each, so $3 \times 241{,}664 = 724{,}992$.
- Per layer: $524{,}288 + 724{,}992 = 1{,}249{,}280$. All layers: $1{,}249{,}280 \times 32 = 39{,}976{,}960$.
- Share of the model: $39{,}976{,}960 / 6{,}738{,}149{,}376 = 0.593\%$.
- Adapter file in bf16: $39{,}976{,}960 \times 2 = 79{,}953{,}920$ bytes $\approx 80$ MB.

(Attention-only with $r = 8$, the earlier lesson's example, gives $8 \times 8192 \times 4 \times 32 = 8{,}388{,}608$, or 0.124%.)

**Step 3: activations (rough).** A common estimate for one layer's saved activations in 16-bit, with a fused attention kernel, is about $34 \cdot s \cdot d$ bytes for $s$ tokens. For $s = 2048$: $34 \times 2048 \times 4096 = 285{,}212{,}672$ bytes per layer. Across 32 layers without checkpointing: $9.13$ GB. With **gradient checkpointing** we keep only each layer's input ($2 \cdot s \cdot d = 16{,}777{,}216$ bytes, times 32 $= 536{,}870{,}912$) plus one layer's full activations during recompute ($285{,}212{,}672$): $822{,}083{,}584$ bytes $\approx 0.82$ GB.

**Step 4: memory per method** (decimal GB, checkpointing on).

| Item | Full FT | LoRA (bf16 base) | QLoRA (NF4 base) |
|---|---|---|---|
| Frozen or trained base weights | $2N = 13.48$ | $2N = 13.48$ | $\frac{4.127}{8} \times 6{,}476{,}005{,}376 = 3.34$, plus embeddings in bf16 $2 \times 262{,}144{,}000 = 0.52$, total $3.87$ |
| Adapter weights $2T$ | none | $0.08$ | $0.08$ |
| Gradients | $2N = 13.48$ | $2T = 0.08$ | $0.08$ |
| Optimizer states | $12N = 80.86$ | $12T = 0.48$ | $0.48$ |
| Activations | $0.82$ | $0.82$ | $0.82$ |
| **Total** | **108.63** | **14.94** | **5.33** |

where $T = 39{,}976{,}960$ trainable LoRA parameters. Check the full fine-tuning total: $16 \times 6{,}738{,}149{,}376 = 107{,}810{,}390{,}016$ bytes, plus $822{,}083{,}584$, gives $108{,}632{,}473{,}600$ bytes $= 108.63$ GB. Full fine-tuning does not fit on one 80 GB GPU. LoRA fits on a 24 GB GPU with room for larger batches. QLoRA fits on a 24 GB GPU with a lot of headroom, or lets a much larger model fit on a 48 GB or 80 GB GPU.

> [!NOTE]
> Rank barely matters for memory. Doubling $r$ to 32 doubles $T$ to 79.95 million, which adds $16 \times 39{,}976{,}960 \approx 0.64$ GB. The frozen base dominates. That is why QLoRA's 4-bit base, not a smaller rank, is what moves the bar.

### DPO: preference tuning without a reward model

The RLHF recipe trains a **reward model** on preference pairs, then uses PPO to maximize reward with a KL penalty that keeps the policy near a reference model. PPO needs four models in memory (policy, reference, reward, value) and is fiddly to tune. DPO shows that the same objective can be optimized directly from the pairs:

```math
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big(\beta\big[\underbrace{\log \tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\text{chosen log-ratio}} - \underbrace{\log \tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\text{rejected log-ratio}}\big]\Big)
```

Each log-probability is summed over the response's tokens. Intuition: raise the chosen response's probability *relative to where the reference model had it*, and lower the rejected one's. $\beta$ controls how far the policy may move: small $\beta$ allows bigger moves.

Worked numbers with $\beta = 0.1$:
- At step 0 the policy equals the reference, so both log-ratios are 0. The argument is $0$, $\sigma(0) = 0.5$, loss $= -\ln 0.5 = 0.693$.
- Later, chosen log-ratio $= +2.0$ and rejected $= -1.0$. Argument $= 0.1 \times (2.0 - (-1.0)) = 0.3$. $\sigma(0.3) = 1/(1 + e^{-0.3}) = 1/(1 + 0.7408) = 0.5744$. Loss $= -\ln 0.5744 = 0.554$.

DPO only needs the policy and a frozen reference (whose log-probs can be precomputed). A known failure: both log-ratios can fall together while their gap grows, so the chosen response's absolute probability drops. Monitor both, not just the loss.

### GRPO and RL with verifiable rewards

GRPO (group relative policy optimization) drops PPO's value network. For each prompt it samples a group of $G$ responses, scores each with a reward $r_i$ (often a program: did the tests pass?), and uses the group-normalized reward as the advantage:

```math
\hat{A}_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}
```

Example: $G = 4$, rewards $[1, 0, 0, 1]$. Mean $= 0.5$. Standard deviation $= \sqrt{\frac{1}{4}(0.25 + 0.25 + 0.25 + 0.25)} = 0.5$. Advantages $= [1, -1, -1, 1]$. Tokens of passing answers are pushed up, failing ones pushed down, with a PPO-style clipped ratio and a KL term to the reference. If all four pass or all four fail, the standard deviation is 0 and the prompt gives no learning signal, so prompt difficulty matters.

### Distillation

**[Distillation](../../glossary.md#distillation)** trains a small student to imitate a large teacher. Sequence-level distillation is just SFT on teacher-generated responses. Logit-level distillation minimizes the KL divergence between teacher and student token distributions, which carries more information per token but requires a shared tokenizer and access to the teacher's logits. Distillation is the usual way to make a small, cheap model match a big one on a narrow task.

## 4. Implementation

```python
# SFT with loss masking and QLoRA (pip install transformers peft bitsandbytes trl)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tok = AutoTokenizer.from_pretrained(BASE)
def encode(ex):
    prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + ex["messages"][-1]["content"] + tok.eos_token
    ids = tok(full, add_special_tokens=False)["input_ids"]
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    return {"input_ids": ids, "labels": [-100] * n_prompt + ids[n_prompt:]}  # mask prompt tokens

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb)
model = prepare_model_for_kbit_training(model)  # enables gradient checkpointing
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
```

Libraries such as TRL wrap this (`SFTTrainer` with assistant-only loss, `DPOTrainer`, `GRPOTrainer`), but the masking above is what they do. After training, `model.merge_and_unload()` folds the adapter into a 16-bit copy of the base for deployment.

Runnable script (parameter counts, LoRA arithmetic, memory table, DPO and GRPO numbers, figure; no GPU needed): [`code/15-llms/fine_tuning_memory.py`](../../code/15-llms/fine_tuning_memory.py).

## 5. Engineering

**Data pipeline.** Render every example with the *exact* chat template the model will see at inference (a mismatch between training and serving templates is a classic silent bug). Deduplicate exact and near-duplicate examples (for example MinHash over n-grams). Check contamination: remove training examples that overlap your eval set by long n-gram matches. Strip secrets and personal data. Read a random sample of 50 examples yourself before every run.

**Hyperparameters that matter.** For LoRA, learning rates are typically around $10^{-4}$ (higher than full fine-tuning, which is typically around $10^{-5}$), 1 to 3 epochs, warmup plus cosine or linear decay. More epochs on a small set overfits fast: the training loss keeps dropping while held-out quality and general skills fall. Pack multiple short examples into one sequence (with attention masks that keep them separate) to avoid wasting compute on padding.

**Memory levers, in order.** Gradient checkpointing (trades about one extra forward pass for most activation memory). LoRA instead of full fine-tuning. QLoRA's 4-bit base. Smaller micro-batch with gradient accumulation. For full fine-tuning: 8-bit optimizers, ZeRO/FSDP sharding across GPUs, CPU offload.

**Regression evals.** Before and after every run, score three suites: (1) the target task, (2) a general-capability suite (reasoning, instruction following, other languages you care about), (3) safety and refusal behavior. Ship only if (1) improves and (2) and (3) stay within a tolerance you set in advance. Mitigations for forgetting: mix a slice of general instruction data into training (replay), lower the learning rate, fewer epochs, LoRA with modest rank.

**Serving.** A merged model is a normal checkpoint. Unmerged adapters let one GPU serve many customers' fine-tunes (multi-LoRA serving batches requests for different adapters together). Adapters are tied to the exact base checkpoint: when the base model is upgraded, you must retrain them.

**When fine-tuning is the wrong tool.** Facts that change (use [RAG](../16-rag/01-rag-pipeline.md) or tools). Fewer than a few hundred good examples and no eval set. A behavior that a better prompt or structured outputs already gets right. Needing citations to sources. A base model your provider will deprecate in months.

> [!WARNING]
> **Failure modes.** Loss computed on prompt tokens; train/serve chat-template mismatch; missing EOS token so the model never stops; eval examples leaked into training; overfitting from too many epochs; forgetting general skills or safety behavior; DPO lowering the chosen response's probability; QLoRA adapter merged into a 4-bit base and requantized, which loses quality (merge into a 16-bit copy instead).

### Common mistakes

- Estimating training memory as "model size times 2" and forgetting gradients and optimizer states.
- Believing a lower LoRA rank is the main memory saver. The frozen base dominates.
- Comparing fine-tuned and base models on the training distribution only.
- Fine-tuning to add knowledge, then being surprised by confident wrong answers on facts just outside the training set.
- Mixing up DPO's $\beta$ with a learning rate. $\beta$ is the strength of the pull toward the reference.
- Running RL with a verifier that can be gamed (for example, tests that pass on empty output). The model will find the loophole.

## 6. Knowledge check

<!-- quiz:fine-tuning-in-practice -->
**[Take the fine-tuning in practice quiz](../../quizzes/fine-tuning-in-practice.md)**
<!-- /quiz -->

**Practice exercise.** A synthetic model has $d = 2048$, $L = 24$, $d_{ff} = 5504$, full multi-head attention. You apply LoRA with $r = 8$ to $q$ and $v$ only. How many trainable parameters? How much memory does full fine-tuning need for 2 billion total parameters (excluding activations)?

<details>
<summary>Solution</summary>

Each of $q$, $v$: $8 \times (2048 + 2048) = 32{,}768$. Per layer: $2 \times 32{,}768 = 65{,}536$. Total: $65{,}536 \times 24 = 1{,}572{,}864$ trainable parameters. ($d_{ff}$ does not matter because MLP layers are not targeted.)

Full fine-tuning: $16 \times 2 \times 10^9 = 32 \times 10^9$ bytes $= 32$ GB before activations.

</details>

**Implementation challenge.** Write `encode(example)` for a two-turn conversation (user, assistant, user, assistant) that masks every token except the two assistant replies. Test it by decoding only the positions whose label is not `-100` and checking you get exactly the two replies plus end-of-turn tokens.

<details>
<summary>Solution sketch</summary>

Render the conversation prefix after each turn with the chat template, tokenize each prefix, and record the token index where each assistant turn starts and ends (the prefix length before and after appending that reply). Set labels to `-100` everywhere, then copy `input_ids` into `labels` for those spans. Tokenizing prefixes separately can differ from tokenizing the full string at boundaries, so assert that each prefix's ids are a prefix of the full ids. Libraries expose this as an "assistant tokens only" option driven by markers in the chat template.

</details>

## Summary

- Post-training runs pretraining, then SFT, then preference tuning (RLHF or DPO), then RL with verifiable rewards (for example GRPO). Your fine-tune is usually a small SFT or DPO run on top.
- SFT trains next-token cross-entropy on response tokens only, using the serving chat template. Data quality, dedup, and contamination checks matter more than volume.
- Full fine-tuning with mixed-precision AdamW costs about 16 bytes per parameter plus activations: about 108.6 GB for the synthetic 7B-style model.
- LoRA trains $r(d_{\text{in}} + d_{\text{out}})$ parameters per matrix (0.59% at $r = 16$ on all linear layers), starts at $BA = 0$, and merges into $W_0$ for free inference. QLoRA stores the base in NF4 and brings the job to about 5.3 GB.
- DPO optimizes preferences directly against a frozen reference; GRPO normalizes verifier rewards within a group of samples. Always run regression evals for forgetting.

**Next:** [The RAG pipeline](../16-rag/01-rag-pipeline.md)

**Related:** [Adapting LLMs](03-adapting-llms.md) · [Evaluating LLM systems](../17-llm-evaluation/01-llm-evaluation.md) · [Training and regularization](../12-training-regularization/01-training-and-regularization.md) · [Cheat sheet: fine-tuning](../../cheatsheets/fine-tuning.md)

## Interview angle

<details>
<summary><strong>How much GPU memory does it take to fully fine-tune a 7B model, and where does it go?</strong></summary>

Roughly 16 bytes per parameter before activations, so about 110 GB for 7B: more than one 80 GB GPU. With mixed-precision AdamW you hold bf16 weights (2 bytes) and bf16 gradients (2 bytes), plus an fp32 master copy of the weights and Adam's two moment estimates (4 bytes each, 12 total). Optimizer state is three quarters of the bill. Activations come on top and scale with tokens per micro-batch; gradient checkpointing cuts them to under a gigabyte at 2K tokens by recomputing layers in the backward pass. The fixes are sharding (ZeRO or FSDP across GPUs), 8-bit optimizers, or not training all the weights: LoRA trains under 1% of parameters, so gradients and optimizer state nearly vanish and the frozen bf16 base (about 13.5 GB) dominates. QLoRA stores that base in 4 bits and gets the job to around 5 GB.

</details>

<details>
<summary><strong>Explain LoRA. Why is B initialized to zero, and what does the alpha over r factor do?</strong></summary>

LoRA freezes a weight matrix $W_0$ and learns a low-rank update, so the effective weight is $W_0 + \frac{\alpha}{r}BA$ with $A$ of shape $[r, d_{\text{in}}]$ and $B$ of shape $[d_{\text{out}}, r]$. Trainable parameters per matrix drop from $d_{\text{in}} d_{\text{out}}$ to $r(d_{\text{in}} + d_{\text{out}})$: for a 4096-square projection at $r = 16$, 131K instead of 16.8M. $B$ starts at zero so $BA = 0$ and training begins exactly at the pretrained model, with no random perturbation. $A$ is random so gradients still reach $B$. The $\alpha/r$ factor keeps the update's magnitude roughly stable as you change rank, so you can sweep $r$ without retuning the learning rate from scratch. At deploy time you can merge $BA$ into $W_0$ for zero extra latency, or keep adapters separate to serve many fine-tunes on one base.

</details>

<details>
<summary><strong>When would you use DPO instead of PPO-based RLHF, and when would you reach for GRPO?</strong></summary>

Use DPO when you have preference pairs (chosen and rejected responses) and want something stable and cheap. It optimizes the same KL-regularized objective as RLHF but directly, as a logistic loss on the gap between the chosen and rejected log-ratios against a frozen reference. No reward model, no value network, no sampling during training, so it needs two models instead of four. PPO-style RLHF still makes sense when you have a good reward model and want on-policy exploration beyond the fixed pairs. GRPO fits tasks with a verifiable reward: math answers, unit tests, schema validity. It samples a group of answers per prompt and uses the reward minus the group mean, divided by the group std, as the advantage, avoiding a value network. Its weakness is reward hacking: if the verifier has a loophole, the model will find it.

</details>

<details>
<summary><strong>After fine-tuning on support tickets, task accuracy went up but users say the model got worse at everything else. What do you check?</strong></summary>

This is catastrophic forgetting until proven otherwise. First, confirm it with numbers: run the same general-capability and safety suites on the base and fine-tuned checkpoints, not just the ticket task. Then check the training setup. Was the loss masked to responses only, and was the chat template identical to serving? A template mismatch can degrade everything. How many epochs, and at what learning rate? A small set run for many epochs with a high learning rate overfits hard. Was the data diverse, or thousands of near-duplicate tickets? Fixes, in order of cost: fewer epochs and a lower learning rate, LoRA with a modest rank instead of full fine-tuning, mixing a slice of general instruction data into the training set, and gating releases on a regression threshold agreed in advance. Also make sure the complaints aren't about an EOS or stop-token bug that makes outputs ramble.

</details>
