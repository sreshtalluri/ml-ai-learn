"use client";
import { useState } from "react";
import { fmt } from "@/lib/ml";
import { LabFrame, Legend, Slider, Stat, type TourStep } from "./ui";

// Synthetic cost model, identical to guide/code/22-ml-system-design/recsys_funnel.py:
//   latency = fixed + itemsScored * perItemMs   (one worker, no sharding)
//   compute = itemsScored * perItemMs           (busy worker-ms per request)
//   recall  = 1 - exp(-itemsKept / tau)         (saturating curve, not a measurement)
export const CATALOG = 10_000_000;
export const OVERHEAD_MS = 20;
export const QPS = 10_000;
export const STAGES = [
  { name: "Candidate generation", fixed: 12, tau: 250, color: "var(--c-blue)" },
  { name: "Ranking", fixed: 15, tau: 50, color: "var(--c-teal)" },
  { name: "Re-ranking", fixed: 3, tau: 10, color: "var(--c-purple)" },
] as const;

export interface FunnelInput { k1: number; k2: number; k3: number; perItem: [number, number, number] }

export function funnel({ k1, k2, k3, perItem }: FunnelInput) {
  const b = Math.min(k2, k1), c = Math.min(k3, b);
  const scored = [k1, k1, b], kept = [k1, b, c];
  const stages = STAGES.map((s, i) => ({
    ...s,
    scored: scored[i],
    kept: kept[i],
    latency: s.fixed + scored[i] * perItem[i],
    compute: scored[i] * perItem[i],
    recall: 1 - Math.exp(-kept[i] / s.tau),
  }));
  return {
    stages,
    total: OVERHEAD_MS + stages.reduce((t, s) => t + s.latency, 0),
    compute: stages.reduce((t, s) => t + s.compute, 0),
    recall: stages.reduce((r, s) => r * s.recall, 1),
  };
}

const DEFAULTS = { k1: 1000, k2: 100, k3: 20, p1: 0.005, p2: 0.06, p3: 0.1, budget: 150 };
const W = 560;

export default function RecsysFunnelLab() {
  const [st, setSt] = useState(DEFAULTS);
  const set = (patch: Partial<typeof DEFAULTS>) => setSt((s) => ({ ...s, ...patch }));
  const r = funnel({ k1: st.k1, k2: st.k2, k3: st.k3, perItem: [st.p1, st.p2, st.p3] });
  const over = r.total - st.budget;
  const slowest = [...r.stages].sort((a, b) => b.latency - a.latency)[0];
  const weakest = [...r.stages].sort((a, b) => a.recall - b.recall)[0];

  // Guided tour: lesson defaults, then one knob moves per step.
  const setup = (p: Partial<typeof DEFAULTS> = {}) => setSt({ ...DEFAULTS, ...p });
  const sweep = (key: "k1" | "p2" | "budget", a: number, b: number, round: (v: number) => number) => (t: number) => set({ [key]: round(a + t * (b - a)) });
  const by100 = (v: number) => Math.round(v / 100) * 100, by10 = (v: number) => Math.round(v / 10) * 10, cents = (v: number) => Math.round(v * 100) / 100;
  // Caption numbers come from the same funnel() the readouts use.
  const at = (p: Partial<typeof DEFAULTS> = {}) => { const q = { ...DEFAULTS, ...p }; return funnel({ k1: q.k1, k2: q.k2, k3: q.k3, perItem: [q.p1, q.p2, q.p3] }); };
  const base = at(), wide = at({ k1: 5000 }), narrow = at({ k1: 200 });
  const ms = (v: number) => fmt(v, 0), rc = (v: number) => fmt(v, 3);
  const tour: TourStep[] = [
    { id: "funnel", caption: "Ten million items go in, twenty come out. Each bar is one stage: a cheap retriever keeps 1,000, a ranker keeps 100, a re-ranker picks the 20 shown.", apply: () => setup() },
    { id: "latency", caption: `The bottom bar is where the time goes: ${ms(base.total)} ms against a ${DEFAULTS.budget} ms budget. The teal ranking stage is ${ms(base.stages[1].latency)} ms of it, because it scores all ${DEFAULTS.k1.toLocaleString("en-US")} candidates with a heavy model.`, apply: () => setup() },
    { id: "rank-more", caption: `Pass more candidates to the ranker, from 1,000 to 5,000. Latency shoots past the dashed budget line to ${ms(wide.total)} ms, while recall barely moves (${rc(base.recall)} → ${rc(wide.recall)}): the retriever had already found most of the good items.`, apply: () => setup(), animate: sweep("k1", 1000, 5000, by100), animMs: 3000 },
    { id: "starved", caption: `Go the other way, down to 200 candidates. Now everything is fast (${ms(narrow.total)} ms), but recall falls to ${rc(narrow.recall)}: good items are dropped before the ranker ever sees them.`, apply: () => setup(), animate: sweep("k1", 1000, 200, by100), animMs: 2600 },
    { id: "heavy-ranker", caption: "Keep 1,000 candidates but make the ranker more expensive per item. The teal segment stretches until the request blows the budget: model cost times items scored is the whole story.", apply: () => setup(), animate: sweep("p2", 0.06, 0.2, cents), animMs: 2800 },
    { id: "budget", caption: `Squeeze the budget from 150 to 100 ms and the same ${ms(base.total)} ms funnel no longer fits. Every funnel is sized backwards from the latency budget: decide how many items each stage can afford.`, apply: () => setup(), animate: sweep("budget", 150, 100, by10), animMs: 2400 },
  ];

  // Funnel: bar width proportional to log10(items), so 10M and 20 both stay visible.
  const levels = [{ label: "Catalog", n: CATALOG, color: "var(--faint)" }, ...r.stages.map((s) => ({ label: `${s.name} keeps`, n: s.kept, color: s.color }))];
  const wOf = (n: number) => 40 + (Math.log10(Math.max(n, 1)) / 7) * (W - 80);
  const scale = Math.max(st.budget, r.total) * 1.1;
  const px = (ms: number) => (ms / scale) * (W - 20);
  let acc = 0;
  const segs = [{ name: "Overhead", latency: OVERHEAD_MS, color: "var(--faint)" }, ...r.stages].map((s) => {
    const x = acc;
    acc += s.latency;
    return { ...s, x };
  });

  return (
    <LabFrame
      id="recsys-funnel"
      title="Recommendation funnel"
      subtitle="Synthetic 10M-item catalog. Choose how many items each stage passes on and how expensive each stage is per item; watch latency, compute, and recall."
      onReset={() => setSt(DEFAULTS)}
      tour={tour}
      presets={[
        { label: "Rank 5,000", apply: () => set({ k1: 5000 }) },
        { label: "Heavy ranker", apply: () => set({ p2: 0.2 }) },
        { label: "Starved ranker", apply: () => set({ k1: 200 }) },
        { label: "Tight budget", apply: () => set({ budget: 100 }) },
      ]}
      controls={
        <>
          <Slider label="k1: candidates to ranker" value={st.k1} min={100} max={5000} step={100} onChange={(v) => set({ k1: v })} />
          <Slider label="k2: ranked items to re-ranker" value={st.k2} min={10} max={500} step={10} onChange={(v) => set({ k2: v })} />
          <Slider label="k3: items shown" value={st.k3} min={5} max={50} step={5} onChange={(v) => set({ k3: v })} />
          <Slider label="candidate gen ms / item" value={st.p1} min={0.001} max={0.02} step={0.001} format={(v) => fmt(v, 3)} onChange={(v) => set({ p1: v })} />
          <Slider label="ranker ms / item" value={st.p2} min={0.01} max={0.2} step={0.01} format={(v) => fmt(v, 2)} onChange={(v) => set({ p2: v })} />
          <Slider label="re-ranker ms / item" value={st.p3} min={0.02} max={0.5} step={0.02} format={(v) => fmt(v, 2)} onChange={(v) => set({ p3: v })} />
          <Slider label="latency budget (ms)" value={st.budget} min={50} max={300} step={10} onChange={(v) => set({ budget: v })} />
        </>
      }
      readout={
        <>
          {r.stages.map((s) => <Stat key={s.name} label={s.name} value={`${fmt(s.latency, 1)} ms`} color={s.color} />)}
          <Stat label="total latency" value={`${fmt(r.total, 1)} ms`} color={over > 0 ? "var(--c-red)" : "var(--c-green)"} />
          <Stat label="headroom" value={`${fmt(-over, 1)} ms`} />
          <Stat label="compute / request" value={`${fmt(r.compute, 1)} worker-ms`} />
          <Stat label={`workers at ${QPS.toLocaleString("en-US")} QPS`} value={Math.ceil((r.compute * QPS) / 1000).toLocaleString("en-US")} />
          <Stat label="recall (synthetic)" value={fmt(r.recall, 3)} />
        </>
      }
      interpretation={
        over > 0
          ? `Over budget by ${fmt(over, 1)} ms. ${slowest.name} is the slowest stage at ${fmt(slowest.latency, 1)} ms. Pass fewer candidates into it or make each item cheaper (distill the model, prune features, batch on a GPU) before you touch anything else.`
          : r.recall < 0.6
            ? `Fits the budget with ${fmt(-over, 1)} ms to spare, but only ${fmt(r.recall * 100, 1)}% of relevant items survive. ${weakest.name} loses the most (stage recall ${fmt(weakest.recall, 3)}). Spend the spare latency on passing more items through that stage.`
            : `Fits the budget with ${fmt(-over, 1)} ms to spare and ${fmt(r.recall * 100, 1)}% synthetic recall. Each stage scores far fewer items than the one before, so the expensive models only ever see a few hundred or thousand items, never the 10M catalog (ranking all of it at ${fmt(st.p2, 2)} ms per item would take ${fmt((CATALOG * st.p2) / 1000, 0)} s).`
      }
    >
      <svg viewBox={`0 0 ${W} 190`} role="img" aria-label="Funnel of items kept per stage" className="w-full h-auto">
        <title>Items kept per stage</title>
        {levels.map((l, i) => {
          const w = wOf(l.n);
          return (
            <g key={l.label}>
              <rect x={(W - w) / 2} y={8 + i * 45} width={w} height={34} rx={6} fill={l.color} opacity={0.25} stroke={l.color} />
              <text x={W / 2} y={30 + i * 45} textAnchor="middle" fontSize="12" fill="var(--text)">
                {l.label}: {l.n.toLocaleString("en-US")}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="mt-4 text-xs text-muted">Latency per request (bar) against the budget (dashed line)</p>
      <svg viewBox={`0 0 ${W} 70`} role="img" aria-label="Latency breakdown versus budget" className="w-full h-auto">
        <title>Latency breakdown versus budget</title>
        {segs.map((s) => (
          <g key={s.name}>
            <rect x={10 + px(s.x)} y={18} width={Math.max(px(s.latency), 1)} height={24} fill={s.color} opacity={0.8} />
            {px(s.latency) > 40 && <text x={10 + px(s.x + s.latency / 2)} y={12} textAnchor="middle" fontSize="11" fill="var(--text)">{fmt(s.latency, 0)} ms</text>}
          </g>
        ))}
        <line x1={10 + px(st.budget)} x2={10 + px(st.budget)} y1={14} y2={50} stroke={over > 0 ? "var(--c-red)" : "var(--c-orange)"} strokeWidth={2} strokeDasharray="5 3" />
        <text x={10 + px(st.budget)} y={64} textAnchor="middle" fontSize="11" fill="var(--muted)">budget {st.budget} ms</text>
      </svg>
      <Legend items={segs.map((s) => ({ label: s.name, color: s.color }))} />
    </LabFrame>
  );
}
