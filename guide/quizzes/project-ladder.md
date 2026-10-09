<!-- GENERATED from project-ladder.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: The portfolio project ladder

Covers the lesson [The portfolio project ladder](../lessons/23-projects/02-project-ladder.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/project-ladder/) grades these interactively and tracks a review queue.

## 1. Match (easy)

Match each project to what makes it stand out.

| Concept | Options |
|---|---|
| Tabular failure-analysis lab | Tool failures, permissions, and prompt injection |
| RAG evaluation workbench | Routing by quality, privacy, latency, and budget |
| Adaptive model router | Separate retrieval, grounding, latency, and cost metrics |
| Agent reliability sandbox | Calibration, slices, drift, explainability |

<details>
<summary>Answer</summary>

- Tabular failure-analysis lab → Calibration, slices, drift, explainability
- RAG evaluation workbench → Separate retrieval, grounding, latency, and cost metrics
- Adaptive model router → Routing by quality, privacy, latency, and budget
- Agent reliability sandbox → Tool failures, permissions, and prompt injection

Each rung adds a layer of production thinking.

</details>

## 2. Calculation (medium)

A router sends 60% of requests to a model costing 0.3 cents and 40% to one costing 1.8 cents. Everything previously went to the 1.8-cent model. What is the percentage cost saving?

<details>
<summary>Answer</summary>

**50** (within ±0.1)

New cost $0.6 \times 0.3 + 0.4 \times 1.8 = 0.18 + 0.72 = 0.90$ cents. Saving $(1.8 - 0.9)/1.8 = 50\%$.

</details>

## 3. Multiple choice (medium)

What is the best minimum viable version for the RAG evaluation workbench?

- **A.** A polished web UI with authentication
- **B.** Compare two chunking strategies on recall@5 and answer correctness over a labeled question set, with a paired comparison
- **C.** Fine-tune a model on the documents
- **D.** Index every document you can find

<details>
<summary>Answer</summary>

**B.** Compare two chunking strategies on recall@5 and answer correctness over a labeled question set, with a paired comparison

The MVP should produce a measured comparison end to end; polish comes later.

</details>

## 4. Multiple choice (medium)

Which resume bullet is strongest?

- **A.** Worked on AI.
- **B.** Built a RAG workbench that improved everything by 300%.
- **C.** Built a RAG evaluation workbench measuring recall@k, groundedness, latency, and cost; chose a reranking setup supported by a paired comparison on 120 questions.
- **D.** Expert in all ML models.

<details>
<summary>Answer</summary>

**C.** Built a RAG evaluation workbench measuring recall@k, groundedness, latency, and cost; chose a reranking setup supported by a paired comparison on 120 questions.

Specific, measurable, honest, and scoped.

</details>

## 5. Arrange in order (easy)

Order the ladder from first to last.

- Agent reliability sandbox
- Adaptive model router
- RAG evaluation workbench
- Semantic incident-triage system
- Tabular model failure-analysis lab

<details>
<summary>Answer</summary>

1. Tabular model failure-analysis lab
2. Semantic incident-triage system
3. RAG evaluation workbench
4. Adaptive model router
5. Agent reliability sandbox

Each project builds on skills from the previous ones.

</details>

## 6. Reflection (medium)

An interviewer asks about a metric in your project README that you can't reproduce anymore. What should you do, and how could you have prevented it?

<details>
<summary>Answer</summary>

**Model answer.** Say so honestly, explain how it was produced and what you'd expect if rerun, and offer to walk through the pipeline. Prevention: pin data and code versions, fix seeds, store configs and results per run, and include a one-command reproduction script in the repository.

Reproducibility protects your credibility as much as your results.

</details>
