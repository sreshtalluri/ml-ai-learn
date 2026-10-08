import Link from "next/link";

// "How everything connects": the ML lifecycle, with each branch of the course attached
// to the stage where it mostly lives. Every node links to a lesson.
const STAGES = [
  { id: "data", label: "Data", href: "/learn/ml-workflow/", note: "collect, clean, split" },
  { id: "repr", label: "Representation", href: "/learn/vectors-and-matrices/", note: "features, vectors, tokens" },
  { id: "model", label: "Model", href: "/learn/linear-regression/", note: "a parameterized function" },
  { id: "pred", label: "Prediction", href: "/learn/ml-vocabulary/", note: "ŷ for one input" },
  { id: "loss", label: "Loss", href: "/learn/logistic-regression/", note: "how wrong, as one number" },
  { id: "opt", label: "Optimization", href: "/learn/gradient-descent/", note: "follow the gradient" },
  { id: "eval", label: "Evaluation", href: "/learn/classification-metrics/", note: "metrics on unseen data" },
  { id: "deploy", label: "Deployment", href: "/learn/production-architecture/", note: "serve predictions" },
  { id: "monitor", label: "Monitoring", href: "/learn/reliability-cost-and-observability/", note: "drift, quality, cost" },
] as const;

const BRANCHES: { label: string; stage: (typeof STAGES)[number]["id"]; href: string; color: string }[] = [
  { label: "NLP", stage: "repr", href: "/learn/text-to-vectors/", color: "var(--c-orange)" },
  { label: "Dimensionality reduction", stage: "repr", href: "/learn/pca/", color: "var(--c-teal)" },
  { label: "Regression", stage: "model", href: "/learn/linear-regression/", color: "var(--c-teal)" },
  { label: "Classification", stage: "model", href: "/learn/logistic-regression/", color: "var(--c-teal)" },
  { label: "Clustering", stage: "model", href: "/learn/k-means/", color: "var(--c-teal)" },
  { label: "Neural networks", stage: "model", href: "/learn/neural-network-forward-pass/", color: "var(--c-purple)" },
  { label: "Transformers", stage: "model", href: "/learn/self-attention/", color: "var(--c-blue)" },
  { label: "LLM applications", stage: "pred", href: "/learn/decoding/", color: "var(--c-teal)" },
  { label: "Backpropagation", stage: "opt", href: "/learn/backpropagation/", color: "var(--c-purple)" },
  { label: "LLM evaluation", stage: "eval", href: "/learn/llm-evaluation/", color: "var(--c-orange)" },
  { label: "Retrieval-augmented generation", stage: "deploy", href: "/learn/rag-pipeline/", color: "var(--c-teal)" },
  { label: "Agents and tool use", stage: "deploy", href: "/learn/production-architecture/", color: "var(--c-orange)" },
  { label: "Safety and security", stage: "monitor", href: "/learn/ai-security/", color: "var(--c-orange)" },
];

export function LifecycleMap() {
  return (
    <div className="overflow-x-auto">
      <ol className="grid gap-3 lg:grid-cols-9 lg:gap-2 min-w-0">
        {STAGES.map((s, i) => (
          <li key={s.id} className="relative flex lg:flex-col gap-3 lg:gap-2">
            <Link href={s.href} className="group block w-40 shrink-0 lg:w-auto rounded-xl border border-line bg-surface px-3 py-2.5 hover:border-accent focus-visible:border-accent">
              <span className="block font-mono text-[10px] text-faint">{i + 1}</span>
              <span className="block text-sm font-semibold group-hover:text-accent">{s.label}</span>
              <span className="block text-xs text-muted leading-snug mt-0.5">{s.note}</span>
            </Link>
            {i < STAGES.length - 1 && <span aria-hidden className="hidden lg:block absolute -right-[7px] top-6 text-faint text-xs">›</span>}
            <div className="flex flex-wrap lg:flex-col gap-1.5 items-start lg:items-stretch pt-1">
              {BRANCHES.filter((b) => b.stage === s.id).map((b) => (
                <Link key={b.label} href={b.href}
                  className="rounded-lg border-l-2 bg-surface-2/70 px-2 py-1 text-xs text-muted hover:text-ink hover:bg-surface-2"
                  style={{ borderLeftColor: b.color }}>
                  {b.label}
                </Link>
              ))}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
