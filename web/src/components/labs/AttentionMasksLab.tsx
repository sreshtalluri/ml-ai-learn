"use client";
import { useState } from "react";
import { LabFrame, Segmented, Slider, Stat, type TourStep } from "./ui";

export type MaskKind = "full" | "causal" | "sliding" | "strided" | "global";
export interface MaskOpts { w: number; stride: number }

const LOCAL = 2; // local window of the strided pattern (self + 1 neighbour)
const RADIUS = 2; // bidirectional local radius of the global-token pattern
const GLOBAL = 0; // the global token (think [CLS])

/** May query i read key j? */
export function allowed(kind: MaskKind, i: number, j: number, o: MaskOpts): boolean {
  switch (kind) {
    case "full": return true;
    case "causal": return j <= i;
    case "sliding": return j <= i && i - j < o.w; // the last w tokens, self included
    case "strided": return j <= i && (i - j < LOCAL || j % o.stride === o.stride - 1);
    case "global": return Math.abs(i - j) <= RADIUS || i === GLOBAL || j === GLOBAL;
  }
}

export const maskMatrix = (kind: MaskKind, n: number, o: MaskOpts) =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => allowed(kind, i, j, o)));

export const attendedCount = (m: boolean[][]) => m.reduce((a, row) => a + row.filter(Boolean).length, 0);

/** Tokens that can influence query i after L stacked causal sliding-window layers of width w. */
export const receptiveField = (L: number, w: number, i: number) => Math.min(i + 1, L * (w - 1) + 1);

const KINDS: { value: MaskKind; label: string }[] = [
  { value: "full", label: "Full" }, { value: "causal", label: "Causal" }, { value: "sliding", label: "Sliding" },
  { value: "strided", label: "Strided" }, { value: "global", label: "Global" },
];
const NAME: Record<MaskKind, string> = {
  full: "full bidirectional", causal: "causal", sliding: "sliding window", strided: "local + strided", global: "local + global token",
};
const DEFAULT = { kind: "causal" as MaskKind, n: 16, w: 4, stride: 4, q: 12, layers: 3 };

export default function AttentionMasksLab() {
  const [s, setS] = useState(DEFAULT);
  const set = (p: Partial<typeof DEFAULT>) => setS((x) => {
    const y = { ...x, ...p };
    return { ...y, q: Math.min(y.q, y.n - 1) };
  });
  const o = { w: s.w, stride: s.stride };
  const m = maskMatrix(s.kind, s.n, o);
  const cells = attendedCount(m);
  const sees = m[s.q].filter(Boolean).length;
  const swa = s.kind === "sliding";
  const reach = receptiveField(s.layers, s.w, s.q);
  const inReach = (j: number) => swa && j <= s.q && s.q - j < reach;

  // Tour captions from the real masks at n = 16.
  const c = (k: MaskKind, w = 4) => attendedCount(maskMatrix(k, 16, { w, stride: 4 }));
  const seen = (k: MaskKind) => maskMatrix(k, 16, { w: 4, stride: 4 })[12].filter(Boolean).length;
  const setup = (p: Partial<typeof DEFAULT>) => setS({ ...DEFAULT, ...p });
  const tour: TourStep[] = [
    { id: "full", caption: `Full attention: every query reads every key, all ${c("full")} cells of a 16 by 16 grid. Cost grows with the square of the length.`, apply: () => setup({ kind: "full" }) },
    { id: "causal", caption: `A causal mask hides the future. ${c("causal")} cells remain, and query 12 sees ${seen("causal")} keys: itself and everything before it.`, apply: () => setup({ kind: "causal" }) },
    { id: "sliding", caption: `A sliding window of 4 keeps only the last 4 tokens per query: ${c("sliding")} cells instead of ${c("causal")}. Watch the band narrow as the window shrinks.`, apply: () => setup({ kind: "sliding" }), animate: (t) => setS((x) => ({ ...x, w: Math.round(16 - t * 12) })), animMs: 2400 },
    { id: "receptive", caption: `One layer only reaches 4 tokens back, but stacking layers relays information: with window 4 the last token is reached by ${[1, 2, 3, 4, 5].map((L) => receptiveField(L, 4, 15)).join(", ")} tokens after 1 to 5 layers.`, apply: () => setup({ kind: "sliding", q: 15, layers: 1 }), animate: (t) => setS((x) => ({ ...x, layers: 1 + Math.round(t * 4) })), animMs: 3000 },
    { id: "strided", caption: `Local plus strided: a 2-token window plus every fourth token. ${c("strided")} cells, yet distant tokens are one hop away through the stride columns.`, apply: () => setup({ kind: "strided" }) },
    { id: "global", caption: `A global token (position 0) reads and is read by everyone, on top of a small local window: ${c("global")} cells. Every pair of tokens is now at most two hops apart.`, apply: () => setup({ kind: "global" }) },
  ];

  const cost = s.kind === "full" ? s.n * s.n : s.kind === "causal" ? (s.n * (s.n + 1)) / 2 : cells;
  return (
    <LabFrame
      id="attention-masks"
      title="Attention mask patterns"
      subtitle="Rows are queries, columns are keys. A coloured cell means that query may attend to that key. Click a row to pick a query."
      tour={tour}
      onReset={() => setS(DEFAULT)}
      presets={[
        { label: "Causal", apply: () => set({ kind: "causal" }) },
        { label: "Sliding window 4", apply: () => set({ kind: "sliding", w: 4 }) },
        { label: "Local + strided", apply: () => set({ kind: "strided", stride: 4 }) },
        { label: "Global token", apply: () => set({ kind: "global" }) },
      ]}
      controls={
        <>
          <Segmented label="Mask" value={s.kind} options={KINDS} onChange={(kind) => set({ kind })} />
          <Slider label="sequence length n" value={s.n} min={8} max={32} onChange={(n) => set({ n })} />
          <Slider label="query position" value={s.q} min={0} max={s.n - 1} onChange={(q) => set({ q })} />
          {swa && <Slider label="window w (self included)" value={s.w} min={1} max={16} onChange={(w) => set({ w })} />}
          {swa && <Slider label="layers stacked" value={s.layers} min={1} max={6} onChange={(layers) => set({ layers })} />}
          {s.kind === "strided" && <Slider label="stride" value={s.stride} min={2} max={8} onChange={(stride) => set({ stride })} />}
        </>
      }
      readout={
        <>
          <Stat label={`keys query ${s.q} sees`} value={sees} color="var(--c-orange)" />
          <Stat label="attended cells" value={`${cells} / ${s.n * s.n}`} />
          <Stat label="share of full" value={`${Math.round((100 * cells) / (s.n * s.n))}%`} />
          <Stat label="score cost" value={swa ? `≈ n·w = ${s.n * s.w}` : s.kind === "full" ? `n² = ${s.n * s.n}` : `${cost} cells`} color="var(--c-teal)" />
          {swa && <Stat label={`reach after ${s.layers} layer${s.layers > 1 ? "s" : ""}`} value={`${reach} tokens`} color="var(--c-purple)" />}
        </>
      }
      interpretation={
        swa
          ? `Each layer lets query ${s.q} read ${Math.min(s.w, s.q + 1)} tokens, but stacking ${s.layers} layers relays information so ${reach} tokens can influence it (L·(w−1)+1, capped by position). Memory per layer is set by w, not n.`
          : s.kind === "full"
            ? "Full bidirectional attention, as in an encoder like BERT: every token sees both sides. Cost and memory grow as n²."
            : s.kind === "causal"
              ? `Causal attention, as in a GPT-style decoder: query ${s.q} sees ${sees} keys, never the future. Still about n²/2 work.`
              : s.kind === "strided"
                ? `A 2-token local window plus every ${s.stride}th earlier token. Long-range keys are reachable in one hop at a fraction of the cells.`
                : "Token 0 acts as a global hub: it reads every token and every token reads it, so information crosses the sequence in two hops."
      }
    >
      <div className="mx-auto max-w-[30rem]">
        <p className="mb-1 text-center text-xs text-faint">key position →</p>
        <div className="grid gap-px" role="grid" aria-label={`${NAME[s.kind]} mask, ${s.n} by ${s.n}`}
          style={{ gridTemplateColumns: `repeat(${s.n}, minmax(0, 1fr))` }}>
          {m.map((row, i) => row.map((ok, j) => {
            const mine = i === s.q;
            const bg = mine && ok ? "var(--c-orange)"
              : mine && inReach(j) ? "color-mix(in srgb, var(--c-purple) 55%, var(--surface))"
              : ok ? "color-mix(in srgb, var(--c-blue) 30%, var(--surface))" : "var(--surface-2)";
            return (
              <div key={`${i}-${j}`} onClick={() => set({ q: i })} title={`query ${i}, key ${j}: ${ok ? "attends" : "masked"}`}
                className="aspect-square cursor-pointer rounded-[2px]" style={{ background: bg, outline: mine ? "1px solid var(--c-orange)" : undefined }} />
            );
          }))}
        </div>
        <div className="mt-2 flex flex-wrap gap-1" aria-label="Pick a query row">
          {Array.from({ length: s.n }, (_, i) => (
            <button key={i} type="button" aria-pressed={i === s.q} aria-label={`query ${i}`} onClick={() => set({ q: i })}
              className={`h-5 min-w-5 rounded px-1 font-mono text-[0.65rem] ${i === s.q ? "bg-accent text-white dark:text-zinc-950" : "border border-line text-muted"}`}>{i}</button>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-4 text-xs text-muted">
          <span className="inline-flex items-center gap-1"><span className="h-2 w-3 rounded-sm" style={{ background: "var(--c-orange)" }} /> what query {s.q} reads directly</span>
          {swa && <span className="inline-flex items-center gap-1"><span className="h-2 w-3 rounded-sm" style={{ background: "color-mix(in srgb, var(--c-purple) 55%, var(--surface))" }} /> reached through {s.layers} stacked layers</span>}
          <span className="inline-flex items-center gap-1"><span className="h-2 w-3 rounded-sm" style={{ background: "color-mix(in srgb, var(--c-blue) 30%, var(--surface))" }} /> allowed for other queries</span>
        </div>
        {swa && (
          <div className="mt-4" aria-label="Receptive field by layer">
            <p className="mb-1 text-sm font-medium">Receptive field of query {s.q} by layer</p>
            {[1, 2, 3, 4, 5, 6].map((L) => {
              const r = receptiveField(L, s.w, s.q);
              return (
                <div key={L} className="flex items-center gap-2 text-xs">
                  <span className="w-14 text-muted">{L} layer{L > 1 ? "s" : ""}</span>
                  <div className="h-3 flex-1 rounded bg-surface-2">
                    <div className="h-3 rounded" style={{ width: `${(100 * r) / s.n}%`, background: L <= s.layers ? "var(--c-purple)" : "var(--faint)" }} />
                  </div>
                  <span className="w-8 text-right font-mono">{r}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </LabFrame>
  );
}
