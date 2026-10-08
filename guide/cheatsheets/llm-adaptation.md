---
title: LLM adaptation
summary: Prompting, RAG, fine-tuning, LoRA, and preference optimization compared by what they change and when to use them.
---

# LLM adaptation

| Method | What changes | Cost | Use when | Not for |
|---|---|---|---|---|
| Prompting (zero/few-shot) | context only | lowest | first attempt for any task; format and instructions | knowledge the model lacks |
| RAG | context, with retrieved documents | moderate (index, retrieval) | private or frequently changing facts; citations | changing style or format |
| Tool calling | context, with live data and actions | moderate | exact lookups, calculations, actions | open-ended knowledge |
| LoRA / PEFT | small added weights ($r(d_{\text{in}} + d_{\text{out}})$ per matrix) | low to moderate training | consistent format, tone, domain behavior; making a small model good at a narrow task | injecting changing facts |
| Full fine-tuning | all weights | high | large data, big domain shift | small data (overfits, forgets) |
| Instruction tuning (SFT) | weights, on instruction-response pairs | high | turning a base model into an assistant | |
| Preference optimization (RLHF, DPO) | weights, on ranked response pairs | high | helpfulness, harmlessness, style | adding knowledge |

> [!IMPORTANT]
> RAG changes the model's context; fine-tuning changes its weights. Use RAG for facts that change; fine-tune for behavior.

**Decision order:** prompt → add RAG/tools → LoRA → full fine-tuning. Each step must beat the previous one on an evaluation set.

**LoRA arithmetic:** a 4096 × 4096 matrix has 16.8M parameters; rank 8 trains 65,536 (0.39%).

**Hallucination controls:** grounding with citations, tools, constrained outputs, verification, abstention, human review. Lower temperature improves consistency, not truth.

Lessons: [Adapting LLMs](../lessons/15-llms/03-adapting-llms.md) · [The RAG pipeline](../lessons/16-rag/01-rag-pipeline.md)
