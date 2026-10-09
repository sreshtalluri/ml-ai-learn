"use client";
import { useState } from "react";
import { Button, LabFrame, Segmented, type TourStep } from "./ui";

type Id = "client" | "gateway" | "orchestration" | "cache" | "retrieval" | "model" | "tools" | "logging" | "evaluation" | "governance";
interface Comp { label: string; x: number; y: number; color: string; responsibility: string; io: string; failures: string; metrics: string; security: string; scaling: string }

const C: Record<Id, Comp> = {
  client: { label: "Client", x: 20, y: 150, color: "var(--faint)", responsibility: "Collect input, stream output, gather feedback.", io: "user actions → HTTPS requests; streamed tokens → UI", failures: "double submits, dropped streams", metrics: "client latency, feedback rate", security: "never holds model keys or secrets", scaling: "CDN, streaming responses" },
  gateway: { label: "API gateway", x: 140, y: 150, color: "var(--c-blue)", responsibility: "Authenticate, enforce rate limits and quotas, reject oversized inputs.", io: "raw request → authenticated, size-checked request", failures: "auth bypass, abuse, overload", metrics: "req/s, 4xx/5xx, throttled requests", security: "authn + authz, input size caps, per-tenant quotas", scaling: "stateless, horizontal" },
  orchestration: { label: "Orchestration", x: 280, y: 150, color: "var(--c-purple)", responsibility: "Choose prompt template and model, assemble context, call tools, validate output, retry and fall back.", io: "request → validated, schema-checked response", failures: "invalid output, retry storms, runaway tool loops", metrics: "step latency, retries, validation failures", security: "separate trusted instructions from untrusted text; tool allowlists", scaling: "stateless workers, queues" },
  cache: { label: "Cache", x: 440, y: 40, color: "var(--faint)", responsibility: "Return stored answers for repeated requests.", io: "request key → cached response", failures: "stale answers, cross-user leakage", metrics: "hit rate, staleness", security: "key by tenant/user where answers differ", scaling: "TTL, eviction, sharding" },
  retrieval: { label: "Retrieval", x: 440, y: 110, color: "var(--c-teal)", responsibility: "Find relevant, permitted chunks for the query.", io: "query + user filters → ranked chunks", failures: "low recall, stale index, permission leaks", metrics: "recall@k, latency, index freshness", security: "permission filter inside the retrieval query", scaling: "sharded ANN indexes, replicas" },
  model: { label: "Model", x: 440, y: 180, color: "var(--c-orange)", responsibility: "Generate tokens from the assembled prompt.", io: "prompt → tokens / tool calls", failures: "timeouts, hallucinations, provider outage", metrics: "TTFT, tokens/s, error rate, cost", security: "data residency, prompt logging policy", scaling: "batching, replicas, fallback models" },
  tools: { label: "Tools", x: 440, y: 250, color: "var(--c-orange)", responsibility: "Fetch live data or take actions the model requested.", io: "validated tool call → result", failures: "side effects on retry, slow APIs", metrics: "success rate, latency", security: "least privilege, user's permissions, idempotency keys, approvals", scaling: "per-tool rate limits" },
  logging: { label: "Logging & tracing", x: 600, y: 60, color: "var(--c-blue)", responsibility: "Trace every step with versions, tokens, latency, cost.", io: "events → traces, dashboards, alerts", failures: "missing context, PII in logs", metrics: "trace coverage, alert latency", security: "redaction, access control, retention", scaling: "sampling, retention tiers" },
  evaluation: { label: "Evaluation", x: 600, y: 150, color: "var(--c-purple)", responsibility: "Score quality offline and online; catch regressions.", io: "traces + labels → scores, release gates", failures: "unrepresentative test sets", metrics: "quality by segment, regressions", security: "includes injection and data-leak tests", scaling: "automated, sampled" },
  governance: { label: "Governance", x: 600, y: 240, color: "var(--c-teal)", responsibility: "Privacy, retention, access, audit, policy.", io: "policies → enforced rules, audit logs", failures: "shadow data copies, missing audit trail", metrics: "audit completeness", security: "least privilege, deletion requests", scaling: "central policy service" },
};

type Scenario = "happy" | "cache" | "outage" | "tool";
const PATHS: Record<Scenario, { id: Id; ms: number; note: string }[]> = {
  happy: [
    { id: "client", ms: 0, note: "User asks: “What's your refund policy for damaged items?”" },
    { id: "gateway", ms: 15, note: "Token valid, under rate limit, input 220 chars: accepted." },
    { id: "orchestration", ms: 5, note: "Intent = policy question → template v7, small-model route." },
    { id: "cache", ms: 3, note: "Cache miss (new phrasing)." },
    { id: "retrieval", ms: 90, note: "Top 3 chunks retrieved with this user's permission filter; evidence at rank 2." },
    { id: "model", ms: 1200, note: "Grounded answer with citation [2], 180 output tokens." },
    { id: "orchestration", ms: 4, note: "Schema valid, citation points at a retrieved chunk. Done." },
    { id: "logging", ms: 0, note: "Trace stored: prompt v7, model version, index version, tokens, 1.3 s, cost." },
  ],
  cache: [
    { id: "client", ms: 0, note: "Same question asked again by the same tenant." },
    { id: "gateway", ms: 15, note: "Accepted." },
    { id: "orchestration", ms: 5, note: "Normalize request → cache key." },
    { id: "cache", ms: 3, note: "Cache hit: answer returned without retrieval or model calls." },
    { id: "logging", ms: 0, note: "Trace records cache_hit = true; cost ≈ 0." },
  ],
  outage: [
    { id: "client", ms: 0, note: "User asks a policy question." },
    { id: "gateway", ms: 15, note: "Accepted." },
    { id: "orchestration", ms: 5, note: "Route to primary model." },
    { id: "retrieval", ms: 90, note: "Chunks retrieved." },
    { id: "model", ms: 5000, note: "Primary model times out after 5 s (timeout set, not infinite)." },
    { id: "orchestration", ms: 2, note: "Circuit breaker opens after repeated failures → fall back to secondary model." },
    { id: "model", ms: 1500, note: "Fallback model answers with the same retrieved context." },
    { id: "logging", ms: 0, note: "Alert fires on primary error rate; trace marks fallback = true." },
  ],
  tool: [
    { id: "client", ms: 0, note: "User: “Refund my order 991.”" },
    { id: "gateway", ms: 15, note: "Accepted." },
    { id: "orchestration", ms: 5, note: "Intent = action. Exposes only the refund tool for this intent." },
    { id: "model", ms: 900, note: "Model proposes refund_order(order_id=991, amount=120)." },
    { id: "orchestration", ms: 3, note: "Arguments validated; user owns order 991; amount under auto-approve limit." },
    { id: "tools", ms: 400, note: "Refund executed once with an idempotency key (a retry can't double-charge)." },
    { id: "governance", ms: 0, note: "Audit log: who, what, which model output triggered it." },
  ],
};
const EDGES: [Id, Id][] = [["client", "gateway"], ["gateway", "orchestration"], ["orchestration", "cache"], ["orchestration", "retrieval"], ["orchestration", "model"], ["orchestration", "tools"], ["retrieval", "logging"], ["model", "evaluation"], ["tools", "governance"]];
const W = 120, H = 44;

export default function ArchitectureLab() {
  const [sel, setSel] = useState<Id>("orchestration");
  const [scenario, setScenario] = useState<Scenario>("happy");
  const [step, setStep] = useState(0);
  const path = PATHS[scenario];
  const cur = path[Math.min(step, path.length - 1)];
  const elapsed = path.slice(0, step + 1).reduce((s, p) => s + p.ms, 0);
  const comp = C[sel];
  const visited = new Set(path.slice(0, step + 1).map((p) => p.id));
  const go = (s: number) => { setStep(s); setSel(path[Math.min(s, path.length - 1)].id); };

  // Guided tour: each scenario's request walks hop by hop through the diagram.
  const at = (sc: Scenario, s: number, id = PATHS[sc][s].id) => { setScenario(sc); setStep(s); setSel(id); };
  const walk = (sc: Scenario) => (t: number) => { const s = Math.round(t * (PATHS[sc].length - 1)); setStep(s); setSel(PATHS[sc][s].id); };
  const tour: TourStep[] = [
    { id: "map", caption: "Ten boxes, one job each. A request enters at the client on the left, the purple orchestration box in the middle decides what to call, and logging, evaluation and governance on the right watch everything.", apply: () => at("happy", 0, "orchestration") },
    { id: "normal", caption: "Follow a normal question hop by hop. The bold outline is where the request is now. Retrieval takes 90 ms, the model 1.2 s: almost all of the 1.3 seconds is generation.", apply: () => at("happy", 0), animate: walk("happy"), animMs: 3500 },
    { id: "orchestration", caption: "Orchestration is the part you write. It picks the prompt template and model, assembles context, and checks the output against a schema before anything reaches the user.", apply: () => at("happy", 2) },
    { id: "cache", caption: "Ask the same question again and the cache answers in 23 ms, skipping retrieval and the model entirely. The cache key must include the tenant, or one customer sees another's answer.", apply: () => at("cache", 0), animate: walk("cache"), animMs: 2400 },
    { id: "outage", caption: "Now the primary model hangs. A 5 second timeout stops the wait, a circuit breaker routes to a fallback model, and an alert fires. Slow, but the user still gets an answer.", apply: () => at("outage", 0), animate: walk("outage"), animMs: 3500 },
    { id: "tool", caption: "A refund request. The model only proposes the call; orchestration checks the user owns the order, the tool runs once with an idempotency key, and governance records who did what.", apply: () => at("tool", 0), animate: walk("tool"), animMs: 3500 },
    { id: "trace", caption: "Every path ends at logging: prompt version, model version, tokens, latency and cost for each request. Without that trace, none of the failures you just saw can be debugged.", apply: () => at("happy", PATHS.happy.length - 1) },
  ];

  return (
    <LabFrame
      id="architecture"
      title="Production LLM architecture"
      subtitle="Click any component for its responsibilities and failure modes, or trace a request through the system."
      onReset={() => { setSel("orchestration"); setScenario("happy"); setStep(0); }}
      tour={tour}
      controls={
        <>
          <Segmented label="Scenario" value={scenario} onChange={(v) => { setScenario(v); setStep(0); setSel("client"); }}
            options={[{ value: "happy", label: "Normal" }, { value: "cache", label: "Cache hit" }, { value: "outage", label: "Outage" }, { value: "tool", label: "Tool call" }]} />
          <div className="flex gap-2">
            <Button onClick={() => go(Math.max(0, step - 1))} disabled={step === 0}>Back</Button>
            <Button primary onClick={() => go(Math.min(path.length - 1, step + 1))} disabled={step >= path.length - 1}>Next hop</Button>
          </div>
          <p className="text-xs text-muted">Hop {step + 1} of {path.length} · elapsed <span className="font-mono text-ink">{(elapsed / 1000).toFixed(2)} s</span></p>
          <p className="text-sm text-ink">{cur.note}</p>
        </>
      }
      interpretation={
        <dl className="grid gap-x-6 gap-y-1.5 @xl:grid-cols-2 text-sm">
          <div className="@xl:col-span-2"><dt className="inline font-medium text-ink">{comp.label}: </dt><dd className="inline">{comp.responsibility}</dd></div>
          <div><dt className="inline font-medium text-ink">In → out: </dt><dd className="inline">{comp.io}</dd></div>
          <div><dt className="inline font-medium text-ink">Common failures: </dt><dd className="inline">{comp.failures}</dd></div>
          <div><dt className="inline font-medium text-ink">Metrics: </dt><dd className="inline">{comp.metrics}</dd></div>
          <div><dt className="inline font-medium text-ink">Security: </dt><dd className="inline">{comp.security}</dd></div>
          <div><dt className="inline font-medium text-ink">Scaling: </dt><dd className="inline">{comp.scaling}</dd></div>
        </dl>
      }
    >
      <svg viewBox="0 0 740 310" className="w-full h-auto" role="img" aria-label="Architecture diagram">
        <title>Production LLM application architecture</title>
        {EDGES.map(([a, b]) => {
          const A = C[a], B = C[b];
          const active = visited.has(a) && visited.has(b);
          return <line key={a + b} x1={A.x + W} y1={A.y + H / 2} x2={B.x} y2={B.y + H / 2} stroke={active ? "var(--text)" : "var(--border)"} strokeWidth={active ? 2 : 1.2} />;
        })}
        {(Object.keys(C) as Id[]).map((id) => {
          const c = C[id];
          const isCur = cur.id === id, isSel = sel === id;
          return (
            <g key={id} transform={`translate(${c.x} ${c.y})`} onClick={() => setSel(id)} className="cursor-pointer" role="button" tabIndex={0}
              aria-label={c.label} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setSel(id)}>
              <rect width={W} height={H} rx={10} fill={`color-mix(in srgb, ${c.color} ${visited.has(id) ? 22 : 8}%, var(--surface))`}
                stroke={isCur ? "var(--text)" : c.color} strokeWidth={isCur ? 3 : isSel ? 2 : 1.2} />
              <text x={W / 2} y={H / 2 + 4} textAnchor="middle" fontSize="12" fill="var(--text)" fontWeight={isSel ? 600 : 400}>{c.label}</text>
            </g>
          );
        })}
        <text x={20} y={300} fontSize="11" fill="var(--faint)">Bold outline = where the request is now · shaded = already visited · click any box for details</text>
      </svg>
    </LabFrame>
  );
}
