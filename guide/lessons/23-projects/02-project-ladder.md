---
title: The portfolio project ladder
summary: Five progressively harder projects, each with a problem statement, architecture, milestones, evaluation, testing, deployment, security, resume bullets, and interview talking points.
skill: engineering
minutes: 30
prerequisites: [twelve-week-plan]
related: [twelve-week-plan, rag-pipeline, production-architecture]
---

# The portfolio project ladder

> **Mental model.** A strong portfolio project isn't "I trained a model." It's "here is a real problem, a baseline, a system, an evaluation that found its failures, and the engineering that keeps it reliable." Each rung adds one layer of production thinking.

**You will learn to**
- Scope five projects from a tabular model to an agent sandbox.
- Plan milestones, a minimum viable version, and stretch goals.
- Write evaluation, testing, deployment, and security plans.
- Turn projects into resume bullets and interview stories without inventing numbers.

**Why it matters.** Hiring managers for ML and AI engineering roles look for evidence of judgment: baselines, evaluation discipline, failure analysis, and production concerns. These projects are designed to produce that evidence.

## 1. Intuition

| Level | Project | What makes it stand out |
|---|---|---|
| 1 | Tabular model failure-analysis lab | not just accuracy: calibration, slices, drift, explainability |
| 2 | Semantic incident triage | cluster logs, classify severity, retrieve similar incidents |
| 3 | RAG evaluation workbench | separate retriever, reranker, grounding, latency, and cost metrics |
| 4 | Adaptive model router | route requests by quality, privacy, latency, and budget |
| 5 | Agent reliability sandbox | instrument tool failures, retries, permissions, and prompt injection |

Do them in order: each reuses skills from the previous one. Track milestones on the website's [progress dashboard](https://sreshtalluri.github.io/ml-ai-learn/progress/).

## 2. Visualization

![A matrix of the five projects against seven skills (classical ML, NLP and embeddings, retrieval, evaluation, backend and APIs, observability, security), with dots showing depth. Evaluation is exercised heavily by every project; security grows from none in project 1 to the most in project 5.](../../figures/project-ladder.png)

## 3. The math: measuring a project

Every project reports results against a baseline with uncertainty:

```math
\Delta = \text{metric}(\text{system}) - \text{metric}(\text{baseline}), \quad \text{with a paired bootstrap 95\% interval}
```

and operational numbers per request: p50 and p95 latency, cost, and error rate. Only report numbers you measured, and state the dataset and conditions.

## 4. Implementation: the five projects

### Project 1: Tabular model failure-analysis lab

- **Problem statement:** a churn (or credit, or fraud) model looks accurate overall, but nobody knows where it fails, whether its probabilities can be trusted, or how it will degrade.
- **Learning goals:** leakage audits, gradient boosting, threshold selection by cost, calibration, slice analysis, drift simulation, SHAP.
- **Architecture:** data validation → train/validation/test split by time → baseline (logistic regression) → gradient-boosted model → calibration → slice and drift report → batch scoring job.
- **Dataset options:** public telecom churn, credit default, or bank marketing datasets; or a synthetic dataset with injected drift.
- **Milestones:** (1) EDA and leakage checklist; (2) baseline and tuned boosting with early stopping; (3) cost-based threshold and calibration curve; (4) slice report and drift simulation.
- **Minimum viable version:** baseline vs boosted model, a cost-based threshold, a calibration plot, and a one-page failure report.
- **Stretch goals:** conformal prediction intervals; a monitoring dashboard comparing live and training distributions; fairness metrics across groups.
- **Evaluation plan:** time-based test set; PR-AUC, recall at a fixed review budget, Brier score, metrics per slice, performance under simulated drift.
- **Testing plan:** unit tests for feature functions; a test that fails if any feature uses data after the prediction time; reproducibility test (same seed, same metrics).
- **Deployment plan:** containerized batch scoring job with versioned model and feature code; output table with score, threshold version, and model version.
- **Security considerations:** personal data minimization, access control on scored outputs, no raw identifiers in logs.
- **Resume bullet examples:** "Built a churn model failure-analysis pipeline (gradient boosting, calibration, slice analysis) that surfaced a segment where recall was [X] points below average; added cost-based thresholding." Fill in brackets only with numbers you measured.
- **Interview talking points:** why the time split; the leakage you found; why you chose the threshold; calibrated versus uncalibrated probabilities; what drift would break first.

### Project 2: Semantic incident-triage system

- **Problem statement:** on-call engineers get floods of alerts and tickets; they need severity prediction, grouping of duplicates, and similar past incidents with their fixes.
- **Learning goals:** TF-IDF baselines, embeddings, clustering, classification, retrieval, building an API.
- **Architecture:** ingest tickets and logs → clean and deduplicate → TF-IDF and embedding representations → severity classifier → clustering of recurring themes → similar-incident retrieval → API and simple UI.
- **Dataset options:** public issue trackers (GitHub issues of a large open-source project), public incident postmortem collections, or synthetic logs.
- **Milestones:** (1) TF-IDF severity baseline; (2) embedding-based similar-incident search; (3) clustering report of recurring themes; (4) API with an evaluation report.
- **Minimum viable version:** severity classifier vs majority baseline, plus top-5 similar incidents for a new ticket.
- **Stretch goals:** hybrid search; an LLM-written summary grounded in retrieved incidents with citations; active learning for labels.
- **Evaluation plan:** macro-F1 for severity; recall@5 and MRR for retrieval on a hand-labeled set; human review of cluster coherence.
- **Testing plan:** tests for text normalization; a fixed retrieval test set in CI; latency test for the API.
- **Deployment plan:** API service with a versioned index; re-index job on new incidents; dashboards for latency and usage.
- **Security considerations:** logs contain secrets and personal data: redact before indexing; restrict access by team.
- **Resume bullet examples:** "Built a semantic incident-triage service (TF-IDF and embedding retrieval, severity classification, clustering), evaluated with recall@5 on a hand-labeled set of [N] incidents."
- **Interview talking points:** TF-IDF versus embeddings in practice; how you labeled evaluation data; clusters as hypotheses; redaction strategy.

### Project 3: RAG evaluation workbench

- **Problem statement:** teams change chunking, embedding models, rerankers, and prompts without knowing what helped; they need a workbench that measures each layer.
- **Learning goals:** RAG pipeline design, retrieval metrics, groundedness scoring, LLM-as-judge with calibration, latency and cost accounting.
- **Architecture:** configurable pipeline (chunker, embedder, retriever, reranker, prompt, model) → evaluation runner over a labeled set → per-case results store → comparison report with paired statistics.
- **Dataset options:** your own documentation set with hand-written questions; public QA datasets with supporting passages.
- **Milestones:** (1) ingestion and chunking; (2) hybrid retrieval with reranking; (3) retrieval metrics (recall@k, MRR, nDCG); (4) groundedness, latency, and cost dashboard.
- **Minimum viable version:** compare two chunking strategies on recall@5 and answer correctness over 50 questions, with a paired comparison.
- **Stretch goals:** judge-human agreement study; prompt-injection test documents; caching experiments.
- **Evaluation plan:** layer-by-layer metrics; failure categories; per-version latency and cost.
- **Testing plan:** deterministic retrieval tests; golden-set regression gate in CI; schema validation of outputs.
- **Deployment plan:** CLI and web report; versioned configs and indexes; results stored for history.
- **Security considerations:** treat documents as untrusted; injection tests in the eval set; no secrets in prompts.
- **Resume bullet examples:** "Built a RAG evaluation workbench measuring retrieval (recall@k, MRR, nDCG), groundedness, latency, and cost per configuration; used it to choose a chunking and reranking setup supported by a paired bootstrap comparison."
- **Interview talking points:** why retrieval is evaluated separately; judge biases and how you calibrated; the trade-off you chose between quality and cost.

### Project 4: Adaptive model router

- **Problem statement:** sending every request to the largest model is slow and expensive; some requests are easy, some contain private data that must stay on a local model.
- **Learning goals:** request classification, cost and latency modeling, fallbacks, circuit breakers, offline replay evaluation.
- **Architecture:** request features (length, task type, sensitivity) → routing policy (rules first, then a learned classifier) → model pool (small local model, large hosted model) → validation → fallback → logs with cost and latency.
- **Dataset options:** a mix of public instruction datasets labeled by difficulty; your own logs with permission; synthetic requests with injected personal data.
- **Milestones:** (1) request classifier and rule-based policy; (2) quality, latency, and cost logging; (3) fallbacks and circuit breakers; (4) offline replay evaluation comparing policies.
- **Minimum viable version:** rule-based router with two models, measuring quality loss versus cost saving on a replay set.
- **Stretch goals:** learned router; budget-aware routing; privacy detector that forces local routing.
- **Evaluation plan:** quality per route on an eval set; blended cost per request; p95 latency; misroute analysis.
- **Testing plan:** policy unit tests; fault injection (provider timeouts) to verify fallbacks; privacy-routing tests.
- **Deployment plan:** router as a service in front of model providers; feature flags for policies; dashboards per route.
- **Security considerations:** sensitive-data detection before routing to external providers; per-tenant quotas; audit logs.
- **Resume bullet examples:** "Designed an adaptive LLM router with fallbacks and circuit breakers; offline replay showed a [X]% cost reduction at a quality difference of [Y] points on an evaluation set of [N] requests."
- **Interview talking points:** how you measured quality per route; failure handling; the privacy constraint; why rules before learning.

### Project 5: Agent reliability sandbox

- **Problem statement:** tool-using agents fail in messy ways (tool errors, loops, injections, excessive permissions); teams need a sandbox to measure and harden them.
- **Learning goals:** tool calling with schemas, fault injection, retries and idempotency, permission models, prompt-injection testing, audit logging.
- **Architecture:** agent loop with schema-validated tool calls → mocked tools with configurable failures → permission and approval layer → injection test documents → step-level traces → reliability report.
- **Dataset options:** a set of multi-step tasks you write (support workflows, data lookups) with expected end states.
- **Milestones:** (1) tool-calling agent with schemas; (2) fault injection for tools; (3) prompt-injection suite; (4) permission model and audit log.
- **Minimum viable version:** 30 tasks, success rate with and without injected tool failures, and executed-harm rate with mocked dangerous tools.
- **Stretch goals:** compare agent designs (plan-then-act versus react-style); cost and step budgets; human-approval UX.
- **Evaluation plan:** task success rate, steps and cost per task, recovery rate after tool failures, attack success and executed-harm rates.
- **Testing plan:** deterministic mocks; regression suite of injection cases; tests that dangerous tools are unreachable for read-only tasks.
- **Deployment plan:** sandbox as a reusable harness others can point their agents at; reports per run.
- **Security considerations:** never run real dangerous actions; least-privilege mock credentials; sandboxed code execution with no network.
- **Resume bullet examples:** "Built an agent reliability sandbox with tool fault injection and a prompt-injection suite; permission checks and task-scoped tool allowlists reduced executed-harm to zero across [N] attack cases."
- **Interview talking points:** why prompts aren't a security boundary; idempotency for retried tools; how you designed the permission model; what still fails.

## 5. Engineering

**Presenting a project:** a README with the problem, a diagram, the baseline, results with intervals, failure analysis, costs, and what you'd do next. Include instructions to reproduce in one command.

**Honesty is a feature.** Report what didn't work and why. Interviewers probe numbers; never fabricate metrics or claim production scale you don't have. The brackets in the resume bullets are placeholders for your measured results.

> [!WARNING]
> **Failure modes.** Projects that are tutorials with new data; no baseline; metrics without uncertainty; no failure analysis; impressive-sounding numbers that can't be reproduced.

### Common mistakes

- Starting with the hardest project.
- Spending weeks on a UI before the evaluation exists.
- Using private or licensed data in a public repository.

## 6. Knowledge check

<!-- quiz:project-ladder -->
**[Take the project ladder quiz](../../quizzes/project-ladder.md)**
<!-- /quiz -->

**Practice exercise.** Write the minimum viable version, evaluation plan, and one security consideration for a sixth project of your own design.

<details>
<summary>What a good answer includes</summary>

A narrow MVP that runs end to end; a baseline; a labeled evaluation set with a named primary metric and slices; operational metrics; and a concrete risk (data leakage, prompt injection, personal data) with a specific control.
</details>

**Implementation challenge.** Pick project 1 and finish its minimum viable version in one week, including a README that follows the presentation checklist above.

## Summary

- Five rungs: tabular failure analysis, semantic triage, RAG evaluation, model routing, and agent reliability.
- Every project needs a baseline, an evaluation plan, testing, deployment, and security thinking.
- Report measured numbers with uncertainty; fill resume bullets only with results you can defend.
- The ladder builds the combination employers want: ML understanding, backend engineering, evaluation discipline, and observability.

**Related:** [The 12-week plan](01-twelve-week-plan.md) · [Production architecture](../20-production-ai/01-production-architecture.md) · [The RAG pipeline](../16-rag/01-rag-pipeline.md)

## Interview angle

<details>
<summary><strong>Walk me through your RAG project. What did you measure?</strong></summary>

Frame it as an evaluation project, because that is what separates it from a tutorial. Example structure, using the course's RAG evaluation workbench: "Teams changed chunking, embedders, and prompts without knowing what helped. I built a configurable pipeline (chunker, embedder, hybrid retriever, reranker, prompt, model) and an evaluation runner over [N] hand-written questions with labeled source passages." Then name the layers you measured separately: retrieval with recall@k, MRR, and nDCG; groundedness of answers; answer correctness; latency; and cost per configuration. Explain why retrieval is evaluated separately: if the evidence isn't retrieved, no prompt fixes the answer. Give one comparison with a paired bootstrap interval, for example two chunking strategies on recall@5, with your measured numbers in place of the brackets, and one surprising failure category you found. Finish with the trade-off you chose, such as a reranker's quality gain against its added latency, and why.

</details>

<details>
<summary><strong>What would you do differently if you rebuilt your model-router project?</strong></summary>

Good answers name a specific decision, what you learned, and the evidence, without trashing the whole project. For the adaptive router, credible answers include the following. "I'd build the offline replay evaluation before the router. I tuned rules on intuition for a week and only later found that misroutes on one request type erased much of the saving." "I'd log quality per route from day one, because a blended cost number hid that one route was degrading." "I started with rules, which was right, since they're debuggable and give a baseline, but I'd have defined the switch criterion to a learned router up front: switch only if it beats rules on the replay set at equal cost." "I'd add fault injection earlier; the first provider timeout exposed a fallback bug." Pick one or two, tie them to something you measured or observed, and say what the change would cost.

</details>

<details>
<summary><strong>How did you decide on the trade-off between quality and cost?</strong></summary>

Show that you treated it as a measured frontier, not a guess. For a router: compute the blended cost from route shares, such as $0.8 \times 0.1 + 0.2 \times 1.5 = 0.38$ cents versus 1.5 cents for sending everything to the large model, a 75% saving in the course's illustrative numbers. Then measure quality per route on the same evaluation set, including misroutes, the hard requests the router wrongly sent to the small model. Plot cost against quality for several router thresholds and choose the point where the quality loss is within an agreed tolerance, such as "no more than [Y] points on the evaluation set and no regression on the high-stakes slice." Mention the constraints that override cost: privacy rules that force local routing, p95 latency budgets, and fallback behavior. Use bracketed placeholders in a resume and real measured values in the interview; never claim a saving you didn't measure.

</details>

<details>
<summary><strong>Your project claims a big improvement. How do I know the number is real?</strong></summary>

Answer with the evidence chain, briefly. Baseline: the improvement is a difference $\Delta$ from a simple, honest baseline, not an absolute score. Same cases: both systems ran on the same held-out evaluation set, so I used a paired bootstrap to get a 95% interval for $\Delta$; with a small set I say the interval is wide. For example, 40 cases with a 10-point gain still had an interval crossing zero. Clean evaluation: the test set was fixed before tuning, time-split where it matters, and checked for leakage. Slices: the gain holds across the important segments, and I report where it doesn't. Operations: latency and cost were measured too, because a quality gain at 50% more cost is a different decision. Reproducibility: one command regenerates the table from versioned data and configs. Be ready to say what didn't work. Interviewers trust a candidate who volunteers limitations far more than one with a perfect story.

</details>
