# Backlog

The first milestone from the spec is complete: application shell, home dashboard, navigation, structured content, progress persistence, Model Explorer, Math Lab, glossary, lesson template, all 41 lessons (the seven flagship lessons in full depth), the required labs, the quiz system, cheat sheets, tests, and documentation. This list is what remains from the full spec.

## Labs not yet built

Each lesson that would host one of these currently has a static figure generated from its Python script.

| Lab | Lesson | Spec reference |
|---|---|---|
| Dataset split and leakage detective game | [Workflow, splits, and leakage](../guide/lessons/02-ml-workflow/01-ml-workflow.md) | Module 2 |
| Train vs validation curve / underfitting-overfitting explorer | [Overfitting and bias-variance](../guide/lessons/02-ml-workflow/02-overfitting-and-bias-variance.md) | Module 2 |
| Naive Bayes probability calculator, SVM margin and kernel views | [Naive Bayes](../guide/lessons/05-instance-and-probabilistic/02-naive-bayes.md), [SVMs](../guide/lessons/05-instance-and-probabilistic/03-support-vector-machines.md) | Module 5 |
| Build a decision tree one split at a time; boosting residual animation | [Decision trees](../guide/lessons/06-trees-and-ensembles/01-decision-trees.md), [Ensembles](../guide/lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md) | Module 6 |
| DBSCAN vs K-means; Gaussian mixture soft memberships | [Clustering](../guide/lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md) | Module 7 |
| Rotatable PCA projection | [PCA](../guide/lessons/08-dimensionality-reduction/01-pca.md) | Module 8 |
| Live tokenizer, n-gram explorer, semantic vector space | [Text to vectors](../guide/lessons/09-classical-nlp/01-text-to-vectors.md), [Embeddings](../guide/lessons/09-classical-nlp/02-word-embeddings.md) | Module 9 |
| Dropout animation, learning-rate schedule and search comparison | [Training and regularization](../guide/lessons/12-training-regularization/01-training-and-regularization.md) | Module 12 |
| Convolution kernel and pooling animation; RNN unrolling; LSTM gates; diffusion steps | [CNNs](../guide/lessons/13-deep-architectures/01-convolutional-networks.md), [RNNs](../guide/lessons/13-deep-architectures/02-recurrent-networks.md), [Diffusion](../guide/lessons/13-deep-architectures/03-autoencoders-diffusion-and-transfer.md) | Module 13 |
| Transformer block explorer (residual paths, normalization) | [Transformer architecture](../guide/lessons/14-transformers/02-transformer-architecture.md) | Module 14 |
| Tokenization explorer, context-window meter, autoregressive generation animation | [Tokenization](../guide/lessons/15-llms/01-tokenization-and-pretraining.md) | Module 15 |
| RAG lab (chunking, retrieval, reranking, a retrieval miss) | [The RAG pipeline](../guide/lessons/16-rag/01-rag-pipeline.md) (the offline Python version exists in `guide/code/16-rag/`) | Module 16 |
| Evaluation dashboard with side-by-side experiments | [LLM evaluation](../guide/lessons/17-llm-evaluation/01-llm-evaluation.md) | Module 17 |
| Clickable production architecture diagram | [Production architecture](../guide/lessons/18-production-ai/01-production-architecture.md) (component table exists in the lesson) | Module 18 |
| Prompt-injection security lab with mocked tools | [Securing AI systems](../guide/lessons/19-safety-security/01-ai-security.md) | Module 19 |

Pattern for adding a lab: create `web/src/components/labs/<Name>Lab.tsx` using `LabFrame` and `Plot`, register it in `registry.tsx`, add metadata to `src/lib/labs.ts`, and wrap the lesson's figure in `<!-- lab:<id> --> … <!-- /lab -->`. The registry test fails if metadata and components disagree.

## Platform improvements

- **Playwright end-to-end tests** for critical flows (complete a lesson, take a quiz, export and import progress). Unit and component tests cover the logic today; browser QA was done manually.
- **Time-spent tracking** (the spec marks it "if practical"); streaks are implemented.
- **Command palette search** across lessons, models, glossary, and cheat sheets.
- **Lesson tabs** (Intuition / Math / Code / Engineering) as an optional view; the sticky table of contents covers navigation today.
- **Mermaid or Excalidraw diagrams** rendered on both GitHub and the website.
- **Interactive Python** (Pyodide) for running the `guide/code` scripts in the browser.
- **Cloud sync and an AI tutor**: the progress store is a single module, so a synced backend can replace `localStorage` without touching callers.
