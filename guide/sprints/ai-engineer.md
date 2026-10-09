---
title: AI Engineer interview sprint
role: AI Engineer
order: 2
summary: Seven days on what AI and LLM engineer loops test - how LLMs work, adaptation, RAG, agents, evaluation, inference, and security.
---

# AI Engineer: 7-day interview sprint

**The job.** Build products on top of foundation models: prompts, retrieval, tools and agents, fine-tuning when it pays off, evaluation, and the serving, cost, and safety work that makes it production-grade.

**What the loop usually tests**

| Round | What they probe | Where it's covered |
|---|---|---|
| LLM fundamentals | Tokens, attention, decoding, context windows, why models hallucinate | Day 1 |
| Adaptation | Prompting vs RAG vs fine-tuning, LoRA, preference tuning | Day 2 |
| LLM system design | RAG, agents, evaluation, guardrails, cost and latency | Days 3 to 6 |
| Inference | KV cache, batching, quantization, latency metrics | Day 6 |
| Practical coding | Build a small RAG or tool-calling loop, write an eval | Each lesson's implementation challenge |

## Day 1: How an LLM produces text

- Read: [Tokenization and pretraining](../lessons/15-llms/01-tokenization-and-pretraining.md), [Self-attention](../lessons/14-transformers/01-self-attention.md), [Transformer architecture](../lessons/14-transformers/02-transformer-architecture.md), [Decoding](../lessons/15-llms/02-decoding.md)
- Labs: tokenizer, self-attention, temperature/top-k/top-p
- Cheat sheet: [transformer architecture](../cheatsheets/transformer-architecture.md)
- Be ready to: explain why token count drives cost and latency, what temperature does to the distribution, and what the KV cache stores.

## Day 2: Adapting models

- Read: [Adapting LLMs](../lessons/15-llms/03-adapting-llms.md), [Fine-tuning in practice](../lessons/15-llms/04-fine-tuning-in-practice.md)
- Lab: LoRA and fine-tuning memory
- Cheat sheets: [LLM adaptation](../cheatsheets/llm-adaptation.md), [fine-tuning](../cheatsheets/fine-tuning.md)
- Be ready to: argue prompting vs RAG vs fine-tuning for a concrete product, compute LoRA's trainable parameters, and explain DPO in two sentences.

## Day 3: Retrieval

- Read: [The RAG pipeline](../lessons/16-rag/01-rag-pipeline.md), [Vector search](../lessons/16-rag/02-vector-search.md), [Text to vectors](../lessons/09-classical-nlp/01-text-to-vectors.md)
- Labs: RAG pipeline, approximate nearest neighbors, TF-IDF
- Cheat sheets: [RAG pipeline](../cheatsheets/rag-pipeline.md), [vector search](../cheatsheets/vector-search.md)
- Be ready to: debug "the right document exists but the answer is wrong", justify hybrid search, and pick chunking settings.

## Day 4: Tools and agents

- Read: [Tool use and agents](../lessons/19-agents/01-tool-use-and-agents.md), [Reliable agents](../lessons/19-agents/02-reliable-agents.md)
- Lab: agent loop
- Cheat sheet: [agents](../cheatsheets/agents-and-tool-use.md)
- Be ready to: explain how a tool call works end to end, when a fixed workflow beats an agent, and how per-step reliability compounds.

## Day 5: Evaluation and security

- Read: [Evaluating LLM systems](../lessons/17-llm-evaluation/01-llm-evaluation.md), [Securing AI systems](../lessons/21-safety-security/01-ai-security.md)
- Lab: prompt injection
- Cheat sheets: [LLM evaluation](../cheatsheets/llm-evaluation.md), [security controls](../cheatsheets/security-controls.md)
- Be ready to: design an eval set and an LLM-as-judge rubric, prove a prompt change helped, and stop indirect prompt injection in code rather than in the prompt.

## Day 6: Inference, serving, and production

- Read: [LLM inference](../lessons/18-llm-inference/01-llm-inference.md), [Serving LLMs](../lessons/18-llm-inference/02-serving-llms.md), [Production architecture](../lessons/20-production-ai/01-production-architecture.md), [Reliability, cost, and observability](../lessons/20-production-ai/02-reliability-cost-and-observability.md)
- Labs: KV cache and inference memory, static vs continuous batching, production architecture
- Cheat sheets: [LLM inference](../cheatsheets/llm-inference.md), [production reliability](../cheatsheets/production-reliability.md)
- Be ready to: estimate KV-cache memory, explain TTFT vs time per output token, and cut serving cost without losing quality.

## Day 7: Mock interview

1. Rapid-fire: 20 mixed cards from this sprint, 60 seconds each ([open the drill](https://sreshtalluri.github.io/ml-ai-learn/drill/?sprint=ai-engineer)).
2. Design, 40 minutes each, out loud: *Design a customer-support assistant that answers from 50,000 help articles and can issue refunds.* Then: *Our LLM feature costs USD 40k a month and p95 latency is 9 seconds. Cut both in half.*
3. Coding, 25 minutes: a minimal tool-calling loop with a JSON schema, a step limit, and error handling; then a 20-question eval harness that scores it.
4. Project story: the [RAG evaluation workbench or the agent sandbox](../lessons/23-projects/02-project-ladder.md), told in five minutes with numbers.
