// Portfolio project ladder (Module 23). Milestones are tracked on the progress dashboard;
// full write-ups live in guide/lessons/23-projects/02-project-ladder.md.
export const PROJECTS = [
  { id: "tabular-failure-lab", title: "Tabular model failure-analysis lab", milestones: ["Baseline + leakage audit", "Gradient-boosted model with tuned threshold", "Calibration and error slices", "Drift simulation and write-up"] },
  { id: "incident-triage", title: "Semantic incident-triage system", milestones: ["TF-IDF severity classifier baseline", "Embedding-based similar-incident search", "Clustering of recurring failure themes", "API with evaluation report"] },
  { id: "rag-workbench", title: "RAG evaluation workbench", milestones: ["Ingestion and chunking pipeline", "Hybrid retrieval with reranking", "Retrieval metrics (recall@k, MRR)", "Groundedness and cost dashboard"] },
  { id: "model-router", title: "Adaptive model router", milestones: ["Request classifier and routing policy", "Quality, latency, and cost logging", "Fallbacks and circuit breakers", "Offline replay evaluation"] },
  { id: "agent-sandbox", title: "Agent reliability sandbox", milestones: ["Tool-calling agent with schemas", "Fault injection for tools", "Prompt-injection test suite", "Permission model and audit log"] },
] as const;
