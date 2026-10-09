---
title: Fine-tuning
summary: Post-training stages, SFT data rules, memory math for full fine-tuning, LoRA, and QLoRA, and the DPO and GRPO formulas.
---

# Fine-tuning

| Stage | Data | Loss | Teaches |
|---|---|---|---|
| Pretraining | trillions of raw tokens | next-token cross-entropy | language, knowledge |
| SFT | (prompt, response) pairs | cross-entropy on response tokens only | assistant format, task behavior |
| Preference tuning | (prompt, chosen, rejected) | DPO, or reward model + PPO | preferred style, helpfulness, refusals |
| RL with verifiable rewards | prompts + checker (tests, answers) | GRPO / PPO on checker reward | multi-step reasoning, correctness |

**Bytes per trained parameter (mixed-precision AdamW):** weights 2 + grads 2 + fp32 master 4 + Adam $m$ 4 + Adam $v$ 4 = **16**. Plus activations ($\approx 34 \cdot s \cdot d$ bytes per layer; gradient checkpointing keeps about $2 \cdot s \cdot d$ per layer).

| Method | Base weights | Trained | Synthetic 7B-style total |
|---|---|---|---|
| Full fine-tuning | 2 bytes/param, trained | all $N$ | ~108.6 GB |
| LoRA | 2 bytes/param, frozen | $r(d_{\text{in}} + d_{\text{out}})$ per matrix | ~14.9 GB |
| QLoRA | ~0.52 bytes/param (NF4), frozen | same adapters | ~5.3 GB |

**LoRA:** $W = W_0 + \frac{\alpha}{r}BA$. $A$ random, $B = 0$, so training starts at the base model. Target all 7 linear layers for best quality. Merge into a 16-bit base for zero-overhead inference. Rank barely affects memory; the frozen base dominates.

**DPO:** $\mathcal{L} = -\log\sigma\big(\beta[\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}]\big)$. Loss starts at $\ln 2 = 0.693$.

**GRPO:** sample $G$ answers, advantage $\hat{A}_i = (r_i - \text{mean})/\text{std}$ within the group. All-pass or all-fail groups give no signal.

> [!WARNING]
> Mask prompt tokens, include the EOS token, use the serving chat template, dedup and decontaminate, and run general-capability and safety regression evals before shipping.

**Use something else when:** facts change (RAG, tools), you have no eval set, a better prompt already works, or you need citations.

Lessons: [Fine-tuning in practice](../lessons/15-llms/04-fine-tuning-in-practice.md) · [Adapting LLMs](../lessons/15-llms/03-adapting-llms.md)
