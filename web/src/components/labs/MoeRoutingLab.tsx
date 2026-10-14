"use client";
import { useState } from "react";
import { fmt } from "@/lib/ml";
import { LabFrame, Slider, Stat, Toggle, type TourStep } from "./ui";

// SYNTHETIC router: seeded random logits with a built-in skew toward expert 0, for illustration only.
export const TOKENS = ["the", "cat", "sat", "on", "a", "warm", "mat", "and", "then", "it", "slept", "."];
const SEED = 7;
const SKEW = 1.2; // extra logit for expert 0, the "popular" expert routers drift toward
const GAMMA = 0.25; // bias step per balancing round
const ROUNDS = 20;

function mulberry32(a: number) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Router logits [tokens][experts]. Same seed → same logits for the first E experts' worth of draws per token. */
export function routerLogits(nTokens: number, nExperts: number, seed = SEED): number[][] {
  const rng = mulberry32(seed);
  return Array.from({ length: nTokens }, () => {
    const row = Array.from({ length: 8 }, (_, e) => (rng() * 2 - 1) * 1.5 + (e === 0 ? SKEW : 0));
    return row.slice(0, nExperts);
  });
}

export const softmax = (z: number[]) => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

const topK = (z: number[], k: number) => z.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map((p) => p[1]);

export const capacityFor = (nTokens: number, k: number, nExperts: number, cf: number) => Math.ceil((cf * nTokens * k) / nExperts);

export interface Slot { expert: number; gate: number; dropped: boolean }
export function route(logits: number[][], k: number, cf: number, balance: boolean) {
  const E = logits[0].length;
  const T = logits.length;
  const target = (T * k) / E;
  const bias = Array(E).fill(0);
  const select = () => logits.map((z) => topK(z.map((v, e) => v + bias[e]), k));
  if (balance) {
    // Aux-loss-free balancing (DeepSeek-V3 style): bias only shifts which experts are selected.
    for (let r = 0; r < ROUNDS; r++) {
      const load = Array(E).fill(0);
      select().forEach((ch) => ch.forEach((e) => load[e]++));
      load.forEach((l, e) => { bias[e] += l > target ? -GAMMA : l < target ? GAMMA : 0; });
    }
  }
  const chosen = select();
  const capacity = capacityFor(T, k, E, cf);
  const demand = Array(E).fill(0);
  const load = Array(E).fill(0);
  let dropped = 0;
  const probs = logits.map(softmax);
  const slots: Slot[][] = chosen.map((ch, t) => {
    const mass = ch.reduce((a, e) => a + probs[t][e], 0);
    return ch.map((e) => {
      demand[e]++;
      const over = load[e] >= capacity; // token order: earlier tokens claim slots first
      if (over) dropped++; else load[e]++;
      return { expert: e, gate: probs[t][e] / mass, dropped: over }; // gates use raw softmax, renormalized
    });
  });
  const mean = (T * k) / E;
  return { probs, slots, demand, load, capacity, dropped, bias, imbalance: Math.max(...demand) / mean, maxDemand: Math.max(...demand) };
}

/** SYNTHETIC config: 4B shared (attention, embeddings, router) + 2B per expert. Billions. */
export const paramCounts = (E: number, k: number) => ({ total: 4 + 2 * E, active: 4 + 2 * k });

const DEFAULT = { E: 8, k: 2, cf: 1, balance: false };

export default function MoeRoutingLab() {
  const [s, setS] = useState(DEFAULT);
  const set = (p: Partial<typeof DEFAULT>) => setS((x) => {
    const y = { ...x, ...p };
    return { ...y, k: Math.min(y.k, 3, y.E) };
  });
  const logits = routerLogits(TOKENS.length, s.E);
  const r = route(logits, s.k, s.cf, s.balance);
  const p = paramCounts(s.E, s.k);
  const T = TOKENS.length;

  const at = (o: Partial<typeof DEFAULT>) => { const c = { ...DEFAULT, ...o }; return route(routerLogits(T, c.E), c.k, c.cf, c.balance); };
  const k1 = at({ k: 1, cf: 2 });
  const d = at({});
  const b = at({ balance: true });
  const roomy = at({ cf: 2 });
  const setup = (o: Partial<typeof DEFAULT>) => setS({ ...DEFAULT, ...o });
  const tour: TourStep[] = [
    { id: "gate", caption: `The router scores each of the ${T} tokens against every expert, and a softmax turns the scores into gate probabilities. Each row of the grid is one token.`, apply: () => setup({ k: 1, cf: 2 }) },
    { id: "top-1", caption: `With top-1 routing each token goes to its single best expert. Expert 0 gets ${k1.demand[0]} of ${T} tokens: an untrained-looking router piles work on one expert.`, apply: () => setup({ k: 1, cf: 2 }) },
    { id: "top-2", caption: `Top-2 sends every token to two experts and mixes their outputs by the renormalized gates. Compute per token doubles, but the busiest expert now has ${d.maxDemand} requests against a fair share of ${(T * 2) / 8}.`, apply: () => setup({ k: 2, cf: 2 }), animate: (t) => setS((x) => ({ ...x, k: t < 0.5 ? 1 : 2 })) },
    { id: "capacity", caption: `Capacity factor 1 gives each expert ${d.capacity} slots. Requests past capacity are dropped, so ${d.dropped} token-expert slots skip the layer and only pass through the residual.`, apply: () => setup({}), animate: (t) => setS((x) => ({ ...x, cf: 2 - Math.round(t * 4) * 0.25 })), animMs: 2400 },
    { id: "balance", caption: `Load balancing adds a per-expert bias that only affects which experts are picked. After ${ROUNDS} rounds the busiest expert drops from ${d.maxDemand} to ${b.maxDemand} requests and drops fall from ${d.dropped} to ${b.dropped}.`, apply: () => setup({ balance: true }) },
    { id: "headroom", caption: `Without balancing, raising the capacity factor to 2 gives each expert ${roomy.capacity} slots and cuts drops to ${roomy.dropped}, but every expert must reserve those slots, so memory and padding grow while most sit empty.`, apply: () => setup({ cf: 2 }) },
    { id: "params", caption: `Memory follows total parameters, compute follows active ones. With 8 experts and top-2 this synthetic model stores ${paramCounts(8, 2).total}B parameters but runs ${paramCounts(8, 2).active}B per token.`, apply: () => setup({}), animate: (t) => setS((x) => ({ ...x, E: 4 + Math.round(t * 4) })) },
  ];

  const W = 100 / (s.E + 1.4);
  return (
    <LabFrame
      id="moe-routing"
      title="Mixture-of-experts routing"
      subtitle="12 synthetic tokens, a seeded synthetic router. Each token picks its top-k experts; each expert has a fixed number of slots."
      tour={tour}
      onReset={() => setS(DEFAULT)}
      presets={[
        { label: "Top-1", apply: () => set({ k: 1 }) },
        { label: "Tight capacity", apply: () => set({ cf: 0.75 }) },
        { label: "Balancing on", apply: () => set({ balance: true }) },
        { label: "Roomy (cf 2)", apply: () => set({ cf: 2 }) },
      ]}
      controls={
        <>
          <Slider label="experts E" value={s.E} min={4} max={8} onChange={(E) => set({ E })} />
          <Slider label="top-k" value={s.k} min={1} max={Math.min(3, s.E)} onChange={(k) => set({ k })} />
          <Slider label="capacity factor" value={s.cf} min={0.5} max={2} step={0.25} format={(v) => v.toFixed(2)} onChange={(cf) => set({ cf })}
            hint={`capacity = ceil(${s.cf} × ${T} × ${s.k} / ${s.E}) = ${r.capacity} slots per expert`} />
          <Toggle label="load balancing (selection bias)" checked={s.balance} onChange={(balance) => set({ balance })} />
        </>
      }
      readout={
        <>
          <Stat label="capacity per expert" value={r.capacity} />
          <Stat label="busiest expert" value={`${r.maxDemand} requests`} color="var(--c-orange)" />
          <Stat label="imbalance (max / mean)" value={fmt(r.imbalance, 2)} />
          <Stat label="dropped slots" value={`${r.dropped} of ${T * s.k}`} color={r.dropped ? "var(--c-red)" : "var(--c-green)"} />
          <Stat label="total params (synthetic)" value={`${p.total}B`} color="var(--c-purple)" />
          <Stat label="active per token" value={`${p.active}B`} color="var(--c-teal)" />
        </>
      }
      interpretation={
        r.dropped === 0
          ? `No drops: every token reached its ${s.k} expert${s.k > 1 ? "s" : ""}. ${r.imbalance > 1.5 ? "Load is still uneven, so some experts sit idle while one does most of the work." : "Load is fairly even."}`
          : `${r.dropped} token-expert slots overflowed and were dropped; those tokens skip that expert and keep only the residual. ${s.balance ? "Balancing already spreads load; raise capacity to remove the rest." : "Turn on balancing or raise the capacity factor."}`
      }
    >
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-[2px] text-center font-mono text-[0.7rem]" aria-label="Token to expert routing">
          <thead>
            <tr><th className="text-left font-normal text-muted">token</th>
              {Array.from({ length: s.E }, (_, e) => <th key={e} className="font-normal text-muted">E{e}</th>)}</tr>
          </thead>
          <tbody>
            {TOKENS.map((tok, t) => (
              <tr key={t}>
                <td className="pr-1 text-left text-ink">{tok}</td>
                {r.probs[t].map((pr, e) => {
                  const slot = r.slots[t].find((x) => x.expert === e);
                  const bg = !slot ? "var(--surface-2)" : slot.dropped ? "color-mix(in srgb, var(--c-red) 30%, var(--surface))"
                    : `color-mix(in srgb, var(--c-blue) ${Math.round(25 + slot.gate * 70)}%, var(--surface))`;
                  return (
                    <td key={e} className="rounded px-0.5 py-1" style={{ background: bg, color: slot && !slot.dropped && slot.gate > 0.6 ? "white" : slot ? "var(--text)" : "var(--faint)" }}
                      title={`${tok} → expert ${e}: router prob ${fmt(pr, 2)}${slot ? `, gate ${fmt(slot.gate, 2)}${slot.dropped ? ", dropped" : ""}` : ""}`}>
                      {slot ? (slot.dropped ? "✕" : fmt(slot.gate, 2)) : fmt(pr, 2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-1 text-xs text-faint">Blue cells are chosen experts with their renormalized gate weight; grey numbers are router probabilities of unchosen experts; ✕ means the expert was full.</p>

      <p className="mb-1 mt-4 text-sm font-medium">Expert load vs capacity</p>
      <svg viewBox="0 0 100 40" className="h-auto w-full" role="img" aria-label={`Requests per expert; capacity ${r.capacity}`}>
        {r.demand.map((dem, e) => {
          const scale = 30 / Math.max(r.capacity, ...r.demand);
          const x = 4 + e * W + W * 0.15;
          const kept = r.load[e] * scale;
          const over = (dem - r.load[e]) * scale;
          return (
            <g key={e}>
              <rect x={x} y={34 - kept} width={W * 0.7} height={kept} fill="var(--c-teal)" rx={0.5} />
              <rect x={x} y={34 - kept - over} width={W * 0.7} height={over} fill="var(--c-red)" rx={0.5} />
              <text x={x + W * 0.35} y={38.5} textAnchor="middle" fontSize={2.6} fill="var(--muted)">E{e}</text>
              <text x={x + W * 0.35} y={33 - kept - over} textAnchor="middle" fontSize={2.4} fill="var(--text)">{dem}</text>
            </g>
          );
        })}
        <line x1={2} x2={98} y1={34 - r.capacity * (30 / Math.max(r.capacity, ...r.demand))} y2={34 - r.capacity * (30 / Math.max(r.capacity, ...r.demand))}
          stroke="var(--c-orange)" strokeWidth={0.4} strokeDasharray="1.5 1" />
        <line x1={2} x2={98} y1={34} y2={34} stroke="var(--grid)" strokeWidth={0.3} />
      </svg>
      <div className="mt-1 flex flex-wrap gap-4 text-xs text-muted">
        <span className="inline-flex items-center gap-1"><span className="h-2 w-3 rounded-sm" style={{ background: "var(--c-teal)" }} /> processed</span>
        <span className="inline-flex items-center gap-1"><span className="h-2 w-3 rounded-sm" style={{ background: "var(--c-red)" }} /> dropped (over capacity)</span>
        <span className="inline-flex items-center gap-1"><span className="h-0 w-3 border-t border-dashed" style={{ borderColor: "var(--c-orange)" }} /> capacity</span>
      </div>
    </LabFrame>
  );
}
