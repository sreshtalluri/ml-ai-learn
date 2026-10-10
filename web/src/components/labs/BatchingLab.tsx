"use client";
import { useMemo, useState } from "react";
import { fmt, rng } from "@/lib/ml";
import { Button, LabFrame, Segmented, Slider, Stat, type TourStep } from "./ui";

// Step-level serving simulator. One time step = one decode iteration; every busy slot emits one token.
// Mirrors guide/code/18-llm-inference/batching_sim.py. Workloads are SYNTHETIC.
export type Policy = "static" | "continuous";
export interface Req { arrival: number; tokens: number }
export interface Segment { req: number; start: number; end: number }

export function simulate(reqs: Req[], slots: number, policy: Policy) {
  const order = reqs.map((_, i) => i).sort((a, b) => reqs[a].arrival - reqs[b].arrival || a - b);
  const queue: number[] = [];
  const busyUntil = new Array(slots).fill(0);
  const finish: number[] = new Array(reqs.length).fill(-1);
  const segments: Segment[][] = Array.from({ length: slots }, () => []);
  let next = 0, done = 0;
  for (let t = 0; done < reqs.length; t++) {
    while (next < order.length && reqs[order[next]].arrival <= t) queue.push(order[next++]);
    const free = busyUntil.map((b, s) => (b <= t ? s : -1)).filter((s) => s >= 0);
    // Static: start a new batch only when every slot is free. Continuous: refill any free slot right away.
    if (policy === "continuous" || free.length === slots) {
      for (const s of free) {
        if (!queue.length) break;
        const i = queue.shift()!;
        busyUntil[s] = finish[i] = t + reqs[i].tokens;
        segments[s].push({ req: i, start: t, end: finish[i] });
        done++;
      }
    }
  }
  return { finish, segments };
}

export function metrics(reqs: Req[], slots: number, finish: number[]) {
  const tokens = reqs.reduce((a, r) => a + r.tokens, 0);
  const makespan = Math.max(...finish);
  const lat = finish.map((f, i) => f - reqs[i].arrival).sort((a, b) => a - b);
  return {
    throughput: tokens / makespan,
    mean: lat.reduce((a, b) => a + b, 0) / lat.length,
    p95: lat[Math.ceil(0.95 * lat.length) - 1],
    idle: 1 - tokens / (slots * makespan),
    makespan,
  };
}

/** Synthetic workload: exponential inter-arrival gaps (rate = requests per step), log-normal output lengths. */
export function makeWorkload(n: number, rate: number, meanLen: number, sigma: number, seed: number): Req[] {
  const r = rng(seed);
  const normal = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  let t = 0;
  return Array.from({ length: n }, () => {
    t += -Math.log(1 - r()) / rate;
    return { arrival: Math.floor(t), tokens: Math.max(1, Math.round(meanLen * Math.exp(sigma * normal() - (sigma * sigma) / 2))) };
  });
}

export const EXAMPLE: Req[] = [2, 8, 3, 3].map((tokens) => ({ arrival: 0, tokens }));
const N = 30, MEAN = 24;
const COLORS = ["var(--c-blue)", "var(--c-teal)", "var(--c-purple)", "var(--c-orange)"];
const DEFAULT = { policy: "continuous" as Policy, rate: 0.1, slots: 4, sigma: 0.8, seed: 1, example: false };

export default function BatchingLab() {
  const [s, setS] = useState(DEFAULT);
  const set = (p: Partial<typeof DEFAULT>) => setS((x) => ({ ...x, example: false, ...p }));

  const reqs = useMemo(() => (s.example ? EXAMPLE : makeWorkload(N, s.rate, MEAN, s.sigma, s.seed)), [s.example, s.rate, s.sigma, s.seed]);
  const slots = s.example ? 2 : s.slots;
  const runs = useMemo(() => {
    const out = {} as Record<Policy, ReturnType<typeof simulate> & { m: ReturnType<typeof metrics> }>;
    for (const p of ["static", "continuous"] as Policy[]) {
      const sim = simulate(reqs, slots, p);
      out[p] = { ...sim, m: metrics(reqs, slots, sim.finish) };
    }
    return out;
  }, [reqs, slots]);
  const cur = runs[s.policy], st = runs.static.m, co = runs.continuous.m;
  const capacity = slots / MEAN;

  // Guided tour: lesson example first, then the synthetic workload with one knob moving per step.
  const setup = (p: Partial<typeof DEFAULT>) => setS({ ...DEFAULT, ...p });
  const sweep = (key: "sigma" | "rate" | "slots", a: number, b: number, dp: number) => (t: number) => setS((x) => ({ ...x, [key]: Number((a + t * (b - a)).toFixed(dp)) }));
  // Caption numbers for the spread steps, from the same simulation the timeline draws.
  const BUSY = 0.3, SPREAD = 1.2;
  const spread = (() => { const w = makeWorkload(N, BUSY, MEAN, SPREAD, DEFAULT.seed); return { st: metrics(w, 4, simulate(w, 4, "static").finish), co: metrics(w, 4, simulate(w, 4, "continuous").finish) }; })();
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  const tour: TourStep[] = [
    { id: "example-static", caption: "Two slots, four requests that need 2, 8, 3 and 3 tokens. Static batching runs a pair to completion before starting the next pair, so the short request's slot sits empty: 11 steps.", apply: () => setup({ example: true, policy: "static" }) },
    { id: "example-continuous", caption: "Continuous batching refills a slot the step after its request finishes. Same work, no gap: 8 steps, 2 tokens per step instead of 1.45.", apply: () => setup({ example: true, policy: "continuous" }) },
    { id: "static-gaps", caption: `Now a busy server: 30 synthetic requests arriving faster than 4 slots can serve them, under static batching. Watch blank gaps open as output lengths spread out: each batch waits for its longest request, leaving ${pct(spread.st.idle)} of slot time idle.`, apply: () => setup({ policy: "static", rate: BUSY, sigma: 0 }), animate: sweep("sigma", 0, SPREAD, 1), animMs: 3000 },
    { id: "continuous", caption: `The same requests under continuous batching. Bars pack tightly however uneven the lengths, because a free slot is refilled at once. Idle time falls to ${pct(spread.co.idle)}: a slot sits empty only when no request is waiting.`, apply: () => setup({ policy: "continuous", rate: BUSY, sigma: 0 }), animate: sweep("sigma", 0, SPREAD, 1), animMs: 3000 },
    { id: "load", caption: `Raise the arrival rate past capacity, ${DEFAULT.slots} slots ÷ ${MEAN} tokens ≈ ${fmt(DEFAULT.slots / MEAN, 2)} requests per step. No policy can keep up: requests queue, and mean latency (readout) becomes mostly waiting.`, apply: () => setup({ rate: 0.04 }), animate: sweep("rate", 0.04, BUSY, 2), animMs: 3200 },
    { id: "slots", caption: "More slots raise capacity, so the same heavy load clears sooner. On a real GPU the slot count is set by how many KV caches fit in memory.", apply: () => setup({ rate: 0.2, slots: 2 }), animate: sweep("slots", 2, 12, 0), animMs: 3000 },
  ];

  const W = 560, rowH = Math.max(14, Math.min(30, 220 / slots)), padL = 48, padT = 8, padB = 30;
  const H = padT + rowH * slots + padB;
  const tMax = Math.max(st.makespan, co.makespan);
  const x = (t: number) => padL + (t / tMax) * (W - padL - 8);
  const tickStep = [1, 2, 5, 10, 20, 50, 100, 200, 500].find((d) => tMax / d <= 8) ?? 1000;

  return (
    <LabFrame
      id="batching"
      title="Static vs continuous batching"
      subtitle={s.example ? "Lesson example: 2 slots, 4 requests at t = 0 with 2, 8, 3, and 3 output tokens." : `${N} synthetic requests, mean output ${MEAN} tokens. One step = one decode iteration; each busy slot emits one token per step.`}
      onReset={() => setS(DEFAULT)}
      tour={tour}
      presets={[
        { label: "Lesson example", apply: () => setS({ ...DEFAULT, example: true }) },
        { label: "Light load", apply: () => setS({ ...DEFAULT, rate: 0.04 }) },
        { label: "Near capacity", apply: () => setS({ ...DEFAULT, rate: 0.16 }) },
        { label: "Same lengths", apply: () => setS({ ...DEFAULT, sigma: 0 }) },
      ]}
      controls={
        <>
          <Segmented label="Policy shown" value={s.policy} onChange={(v) => setS((x) => ({ ...x, policy: v }))} options={[{ value: "static", label: "Static" }, { value: "continuous", label: "Continuous" }]} />
          <Slider label="arrival rate (requests/step)" value={s.rate} min={0.02} max={0.3} step={0.01} onChange={(v) => set({ rate: v })} format={(v) => fmt(v, 2)}
            hint={s.example ? "lesson example: 4 fixed requests, all at t = 0" : `capacity ≈ slots ÷ mean length = ${fmt(capacity, 3)}`} />
          <Slider label="batch slots K" value={slots} min={1} max={16} onChange={(v) => set({ slots: v })} />
          <Slider label="output-length spread σ" value={s.sigma} min={0} max={1.5} step={0.1} onChange={(v) => set({ sigma: v })} format={(v) => fmt(v, 1)} hint="log-normal σ; 0 = every request the same length" />
          <Button onClick={() => set({ seed: s.seed + 1 })}>New workload</Button>
        </>
      }
      readout={
        <>
          <div className="grid grid-cols-[1fr_auto_auto] gap-x-3 text-muted"><span /><span>static</span><span>contin.</span></div>
          {([
            ["tokens/step", fmt(st.throughput, 2), fmt(co.throughput, 2)],
            ["mean latency", fmt(st.mean, 1), fmt(co.mean, 1)],
            ["p95 latency", String(st.p95), String(co.p95)],
            ["idle slots", `${Math.round(st.idle * 100)}%`, `${Math.round(co.idle * 100)}%`],
            ["finish step", String(st.makespan), String(co.makespan)],
          ] as const).map(([k, a, b]) => (
            <div key={k} className="grid grid-cols-[1fr_auto_auto] gap-x-3">
              <span className="text-muted">{k}</span><span style={{ color: "var(--c-orange)" }}>{a}</span><span style={{ color: "var(--c-blue)" }}>{b}</span>
            </div>
          ))}
          <Stat label="latency unit" value="steps" />
        </>
      }
      interpretation={
        <>
          <p className="text-ink">
            {s.policy === "static"
              ? "Static batching starts a batch only when every slot is free, so each batch lasts as long as its longest request. Short requests finish early and their slots sit idle (the gaps), while new arrivals wait in the queue."
              : "Continuous batching refills a slot on the very next step after its request finishes, so slots only go idle when nobody is waiting."}
          </p>
          <p className="mt-1">
            Here continuous batching gives {fmt(co.throughput / st.throughput, 2)}× the throughput and {fmt(st.mean / co.mean, 2)}× lower mean latency (p95 {st.p95} → {co.p95} steps).{" "}
            {s.example ? "Same numbers as the worked example: 16/11 = 1.45 vs 16/8 = 2.00 tokens per step." : s.sigma === 0 ? "With identical lengths every request in a batch ends together, so static batching wastes little; the gap comes from requests that arrive mid-batch." : s.rate > capacity ? "Arrivals exceed capacity, so the queue grows and latency is dominated by waiting under either policy." : "The wider the spread of output lengths, the more static batching wastes."}
          </p>
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Slot timeline for ${s.policy} batching`} className="w-full h-auto select-none">
        <title>{`Slot timeline for ${s.policy} batching`}</title>
        {Array.from({ length: Math.floor(tMax / tickStep) + 1 }, (_, i) => i * tickStep).map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={padT} y2={padT + rowH * slots} stroke="var(--grid)" />
            <text x={x(t)} y={padT + rowH * slots + 14} textAnchor="middle" fontSize="11" fill="var(--faint)">{t}</text>
          </g>
        ))}
        <text x={(padL + W) / 2} y={H - 2} textAnchor="middle" fontSize="12" fill="var(--muted)">time step</text>
        {cur.segments.map((segs, slot) => (
          <g key={slot}>
            <text x={padL - 6} y={padT + rowH * slot + rowH / 2 + 4} textAnchor="end" fontSize="11" fill="var(--faint)">slot {slot}</text>
            {segs.map((g) => (
              <g key={g.req}>
                <rect x={x(g.start)} y={padT + rowH * slot + 2} width={Math.max(1, x(g.end) - x(g.start) - 1)} height={rowH - 4} rx={3} fill={COLORS[g.req % 4]} opacity={0.85}>
                  <title>{`request ${g.req}: arrived ${reqs[g.req].arrival}, ran ${g.start}–${g.end} (${reqs[g.req].tokens} tokens)`}</title>
                </rect>
                {x(g.end) - x(g.start) > 24 && rowH >= 16 && (
                  <text x={(x(g.start) + x(g.end)) / 2} y={padT + rowH * slot + rowH / 2 + 4} textAnchor="middle" fontSize="10" fill="white">r{g.req}</text>
                )}
              </g>
            ))}
          </g>
        ))}
      </svg>
      <p className="text-xs text-muted mt-2">Each bar is one request occupying a slot from admission to its last token. Blank space is an idle slot. Hover a bar for its arrival time.</p>
    </LabFrame>
  );
}
