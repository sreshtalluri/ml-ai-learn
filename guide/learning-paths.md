---
title: Learning paths
summary: Routes through the course by goal and time available, with the labs to do along the way.
---

# Learning paths

Pick one route and follow it in order. Each step is a lesson; the lab next to it is where to spend your hands-on time. Every lab lesson ends its visualization section with **Try it** prompts: make the prediction before touching a control.

On the website, any lesson can be switched to **Quick read**, which keeps the intuition, the lab, the summary, and the interview questions, and hides the math, implementation, engineering, and knowledge-check sections. The whole course takes about 8 hours this way instead of about 30. Use it for review, or for a first pass before reading in full.

## 1. Full course (14 weeks, 5 to 7 hours a week)

The complete sequence. Weeks 1 to 12 match the [12-week plan](lessons/23-projects/01-twelve-week-plan.md); weeks 13 and 14 add LLM inference, agents, and ML system design.

| Week | Lessons | Labs |
|---|---|---|
| 1 | [Vectors](lessons/00-foundations/01-vectors-and-matrices.md), [calculus](lessons/00-foundations/02-calculus-for-ml.md), [probability](lessons/00-foundations/03-probability-and-statistics.md), [Python](lessons/00-foundations/04-python-toolkit.md), [vocabulary](lessons/01-ml-vocabulary/01-ml-vocabulary.md), [paradigms](lessons/01-ml-vocabulary/02-learning-paradigms.md), [workflow](lessons/02-ml-workflow/01-ml-workflow.md), [overfitting](lessons/02-ml-workflow/02-overfitting-and-bias-variance.md) | paradigm guide, fit explorer |
| 2 | [Linear regression](lessons/03-regression/01-linear-regression.md), [regularization](lessons/03-regression/02-regularization-and-regression-metrics.md), [logistic regression](lessons/04-classification/01-logistic-regression.md), [metrics](lessons/04-classification/02-classification-metrics.md) | linear regression, threshold |
| 3 | [KNN](lessons/05-instance-and-probabilistic/01-k-nearest-neighbors.md), [Naive Bayes](lessons/05-instance-and-probabilistic/02-naive-bayes.md), [SVMs](lessons/05-instance-and-probabilistic/03-support-vector-machines.md), [decision trees](lessons/06-trees-and-ensembles/01-decision-trees.md) | KNN, decision tree |
| 4 | [Random forests and boosting](lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md) | boosting |
| 5 | [K-means](lessons/07-unsupervised/01-k-means.md), [other clustering](lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md), [PCA](lessons/08-dimensionality-reduction/01-pca.md) | K-means, PCA |
| 6 | [Text to vectors](lessons/09-classical-nlp/01-text-to-vectors.md), [embeddings](lessons/09-classical-nlp/02-word-embeddings.md) | TF-IDF |
| 7 | [Forward pass](lessons/10-neural-networks/01-neural-network-forward-pass.md), [gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md), [backpropagation](lessons/11-gradient-descent-backprop/02-backpropagation.md) | forward pass, gradient descent, backprop |
| 8 | [Training](lessons/12-training-regularization/01-training-and-regularization.md), [CNNs](lessons/13-deep-architectures/01-convolutional-networks.md), [RNNs](lessons/13-deep-architectures/02-recurrent-networks.md), [autoencoders and diffusion](lessons/13-deep-architectures/03-autoencoders-diffusion-and-transfer.md) | convolution |
| 9 | [Self-attention](lessons/14-transformers/01-self-attention.md), [transformer architecture](lessons/14-transformers/02-transformer-architecture.md) | self-attention |
| 10 | [Tokenization](lessons/15-llms/01-tokenization-and-pretraining.md), [decoding](lessons/15-llms/02-decoding.md), [adapting LLMs](lessons/15-llms/03-adapting-llms.md), [fine-tuning in practice](lessons/15-llms/04-fine-tuning-in-practice.md) | tokenizer, decoding, LoRA |
| 11 | [RAG](lessons/16-rag/01-rag-pipeline.md), [vector search](lessons/16-rag/02-vector-search.md), [evaluation](lessons/17-llm-evaluation/01-llm-evaluation.md) | RAG, vector search |
| 12 | [Architecture](lessons/20-production-ai/01-production-architecture.md), [reliability](lessons/20-production-ai/02-reliability-cost-and-observability.md), [security](lessons/21-safety-security/01-ai-security.md), [projects](lessons/23-projects/02-project-ladder.md) | architecture, prompt injection |
| 13 | [LLM inference](lessons/18-llm-inference/01-llm-inference.md), [serving LLMs](lessons/18-llm-inference/02-serving-llms.md), [tool use and agents](lessons/19-agents/01-tool-use-and-agents.md), [reliable agents](lessons/19-agents/02-reliable-agents.md) | KV cache, batching, agent loop |
| 14 | [ML system design](lessons/22-ml-system-design/01-ml-system-design.md), [experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md) | recommendation funnel, A/B test |

## 2. AI-engineering fast track (about 5 weeks)

For experienced software engineers who want to build LLM products and understand what's underneath. It skips most classical ML and keeps the ideas every AI engineer needs.

| Order | Lesson | Why it's on this path | Lab |
|---|---|---|---|
| 1 | [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md) | loss vs metric, training vs inference | |
| 2 | [Workflow, splits, and leakage](lessons/02-ml-workflow/01-ml-workflow.md) | evaluation discipline applies to LLM systems too | |
| 3 | [Overfitting and bias-variance](lessons/02-ml-workflow/02-overfitting-and-bias-variance.md) | why held-out evaluation matters | fit explorer |
| 4 | [Classification metrics](lessons/04-classification/02-classification-metrics.md) | precision, recall, thresholds, calibration | threshold |
| 5 | [Text to vectors](lessons/09-classical-nlp/01-text-to-vectors.md) | sparse retrieval (BM25's ancestor) | TF-IDF |
| 6 | [Word embeddings](lessons/09-classical-nlp/02-word-embeddings.md) | dense retrieval | |
| 7 | [Forward pass](lessons/10-neural-networks/01-neural-network-forward-pass.md) | what a layer computes | forward pass |
| 8 | [Gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md) | how anything is trained | gradient descent |
| 9 | [Self-attention](lessons/14-transformers/01-self-attention.md) | the core of every LLM | self-attention |
| 10 | [Transformer architecture](lessons/14-transformers/02-transformer-architecture.md) | where parameters, compute, and the KV cache live | |
| 11 | [Tokenization and pretraining](lessons/15-llms/01-tokenization-and-pretraining.md) | tokens drive cost and context limits | tokenizer |
| 12 | [Decoding](lessons/15-llms/02-decoding.md) | temperature and sampling | decoding |
| 13 | [Adapting LLMs](lessons/15-llms/03-adapting-llms.md) | prompting vs RAG vs LoRA | |
| 14 | [Fine-tuning in practice](lessons/15-llms/04-fine-tuning-in-practice.md) | SFT, LoRA and QLoRA memory, preference tuning | LoRA |
| 15 | [The RAG pipeline](lessons/16-rag/01-rag-pipeline.md) | the most common LLM architecture | RAG |
| 16 | [Vector search](lessons/16-rag/02-vector-search.md) | ANN indexes, hybrid search, filtering | vector search |
| 17 | [Evaluating LLM systems](lessons/17-llm-evaluation/01-llm-evaluation.md) | proving a change helped | |
| 18 | [Tool use and agents](lessons/19-agents/01-tool-use-and-agents.md) | tool calling, the agent loop, MCP | agent loop |
| 19 | [Reliable agents](lessons/19-agents/02-reliable-agents.md) | guardrails, agent evals, cost | |
| 20 | [LLM inference](lessons/18-llm-inference/01-llm-inference.md) | KV cache, quantization, speculative decoding | KV cache |
| 21 | [Serving LLMs](lessons/18-llm-inference/02-serving-llms.md) | batching, latency SLOs, cost per token | batching |
| 22 | [Production architecture](lessons/20-production-ai/01-production-architecture.md) | the system around the model | architecture |
| 23 | [Reliability, cost, and observability](lessons/20-production-ai/02-reliability-cost-and-observability.md) | keeping it up and affordable | |
| 24 | [Securing AI systems](lessons/21-safety-security/01-ai-security.md) | prompt injection and tool safety | prompt injection |

Then build project 3 (RAG evaluation workbench) or project 5 (agent reliability sandbox) from the [project ladder](lessons/23-projects/02-project-ladder.md).

## 3. Interview prep (7 days per role)

Pick the [interview sprint](sprints/README.md) for your target role: [ML Engineer](sprints/ml-engineer.md), [AI Engineer](sprints/ai-engineer.md), [Applied Scientist](sprints/applied-scientist.md), or [Data Scientist](sprints/data-scientist.md). Each day lists the lessons, labs, cheat sheets, and interview questions that role is tested on, and day 7 is a mock interview.

Whatever the role, also:

1. Read the [model-selection](cheatsheets/model-selection.md), [classification metrics](cheatsheets/classification-metrics.md), [formulas](cheatsheets/formulas.md), and [failure modes](cheatsheets/failure-modes.md) cheat sheets.
2. Take every quiz without notes; reread only the lessons where you missed questions. The website's review queue tracks these for you.
3. Redo the hand calculations: [linear regression's gradient step](lessons/03-regression/01-linear-regression.md), [backpropagation's nine steps](lessons/11-gradient-descent-backprop/02-backpropagation.md), [attention's worked example](lessons/14-transformers/01-self-attention.md), [TF-IDF](lessons/09-classical-nlp/01-text-to-vectors.md), and [Gini](lessons/06-trees-and-ensembles/01-decision-trees.md). Check them in the Math Lab.
4. Practice the "explain it, calculate it, implement it, diagnose it" test out loud for: bias-variance, precision vs recall, bagging vs boosting, Q/K/V, RAG vs fine-tuning, and prompt-injection controls.
5. Drill the interview questions at the end of every lesson. On the website, the [rapid-fire drill](https://sreshtalluri.github.io/ml-ai-learn/drill/) times you and brings back the cards you marked shaky.
