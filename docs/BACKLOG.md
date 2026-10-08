# Backlog

The first milestone from the spec is complete: application shell, home dashboard, navigation, structured content, progress persistence, Model Explorer, Math Lab, glossary, lesson template, all 41 lessons (the seven flagship lessons in full depth), the required labs, the quiz system, cheat sheets, tests, and documentation. This list is what remains from the full spec.

## Labs not yet built

The lesson-by-lesson review in [COURSE_DESIGN.md](COURSE_DESIGN.md) ranks every candidate lab. Twenty are built. The B-priority ones still to build:

| Lab | Lesson |
|---|---|
| Vector playground (drag vectors, projection, cosine) | [Vectors and matrices](../guide/lessons/00-foundations/01-vectors-and-matrices.md) |
| Bayes base-rate lab | [Probability and statistics](../guide/lessons/00-foundations/03-probability-and-statistics.md) |
| Data split and leakage detective | [Workflow, splits, and leakage](../guide/lessons/02-ml-workflow/01-ml-workflow.md) |
| DBSCAN vs K-means, Gaussian mixture memberships | [Clustering](../guide/lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md) |
| Semantic vector space and analogies | [Word embeddings](../guide/lessons/09-classical-nlp/02-word-embeddings.md) |
| Training curves: early stopping, dropout, schedules | [Training and regularization](../guide/lessons/12-training-regularization/01-training-and-regularization.md) |
| RNN unrolling and vanishing gradients | [Recurrent networks](../guide/lessons/13-deep-architectures/02-recurrent-networks.md) |
| Transformer sizing (params, compute, KV cache) | [Transformer architecture](../guide/lessons/14-transformers/02-transformer-architecture.md) |
| Evaluation comparison (fixed vs broken cases, intervals) | [LLM evaluation](../guide/lessons/17-llm-evaluation/01-llm-evaluation.md) |
| Reliability simulator (retries, breakers, fallbacks) | [Reliability, cost, and observability](../guide/lessons/18-production-ai/02-reliability-cost-and-observability.md) |

Pattern for adding a lab: create `web/src/components/labs/<Name>Lab.tsx` using `LabFrame` and `Plot`, register it in `registry.tsx`, add metadata to `src/lib/labs.ts`, and wrap the lesson's figure in `<!-- lab:<id> --> … <!-- /lab -->`. The registry test fails if metadata and components disagree.

## Platform improvements

- **Playwright end-to-end tests** for critical flows (complete a lesson, take a quiz, export and import progress). Unit and component tests cover the logic today; browser QA was done manually.
- **Time-spent tracking** (the spec marks it "if practical"); streaks are implemented.
- **Command palette search** across lessons, models, glossary, and cheat sheets.
- **Lesson tabs** (Intuition / Math / Code / Engineering) as an optional view; the sticky table of contents covers navigation today.
- **Mermaid or Excalidraw diagrams** rendered on both GitHub and the website.
- **Interactive Python** (Pyodide) for running the `guide/code` scripts in the browser.
- **Cloud sync and an AI tutor**: the progress store is a single module, so a synced backend can replace `localStorage` without touching callers.
