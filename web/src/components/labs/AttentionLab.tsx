"use client";
import { useMemo, useState } from "react";
import { attention, fmt, gaussian, matmul, rng, transpose, type Mat } from "@/lib/ml";
import { Button, LabFrame, Segmented, Stat, Tex, Toggle, type TourStep } from "./ui";

const D_MODEL = 4;
const D_K = 2;

/** Deterministic toy embedding per word (hash -> seed). SYNTHETIC: not learned. */
function embed(word: string): number[] {
  let h = 7;
  for (const c of word.toLowerCase()) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  const r = rng(h);
  return Array.from({ length: D_MODEL }, () => +(gaussian(r) * 0.8).toFixed(2));
}

/** Sinusoidal positional encoding (Vaswani et al.), d_model = 4. */
const posEnc = (pos: number) => [Math.sin(pos), Math.cos(pos), Math.sin(pos / 100), Math.cos(pos / 100)].map((v) => +v.toFixed(2));

function weights(seed: number): Mat {
  const r = rng(seed);
  return Array.from({ length: D_MODEL }, () => Array.from({ length: D_K }, () => +(gaussian(r) * 0.7).toFixed(2)));
}
const WQ = weights(101), WK = weights(202), WV = weights(303);

// The course's hand-worked example: one query, two keys.
const EX = { Q: [[1, 0]], K: [[1, 0], [0, 1]], V: [[10, 0], [0, 6]] };
const EX_R = attention(EX.Q, EX.K, EX.V, false);

function MatrixView({ m, rows, cols, title, shape, heat, focusRow, digits = 2 }: {
  m: Mat; rows: string[]; cols: string[]; title: string; shape: string; heat?: boolean; focusRow?: number; digits?: number;
}) {
  return (
    <figure className="min-w-0">
      <figcaption className="flex items-baseline justify-between gap-2 text-xs mb-1">
        <span className="font-medium text-ink">{title}</span>
        <span className="font-mono text-faint">{shape}</span>
      </figcaption>
      <div className="overflow-x-auto">
        <table className="font-mono text-[0.72rem] border-separate border-spacing-0.5">
          <thead>
            <tr><th />{cols.map((c, j) => <th key={j} className="px-1 font-normal text-faint">{c}</th>)}</tr>
          </thead>
          <tbody>
            {m.map((row, i) => (
              <tr key={i} className={focusRow === i ? "outline outline-2 outline-[var(--c-orange)] rounded" : ""}>
                <th className="pr-1.5 text-right font-normal text-faint whitespace-nowrap">{rows[i]}</th>
                {row.map((v, j) => (
                  <td key={j} className="px-1.5 py-1 text-center rounded min-w-10"
                    style={heat ? { background: `color-mix(in srgb, var(--c-blue) ${Math.round((Number.isFinite(v) ? v : 0) * 85)}%, transparent)`, color: v > 0.55 ? "white" : undefined } : { background: "var(--surface-2)" }}>
                    {Number.isFinite(v) ? fmt(v, digits) : "−∞"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}

const STEPS = [
  { short: "Embed", title: "Tokens and embeddings", tex: "X = E[\\text{tokens}] + P" },
  { short: "Q, K, V", title: "Project to queries, keys, values", tex: "Q = XW_Q,\\quad K = XW_K,\\quad V = XW_V" },
  { short: "QKᵀ", title: "Compare every query with every key", tex: "S = QK^\\top" },
  { short: "Scale", title: "Scale by the square root of the key dimension", tex: "S' = S / \\sqrt{d_k}" },
  { short: "Mask", title: "Apply the causal mask", tex: "S'_{ij} = -\\infty \\text{ for } j > i" },
  { short: "Softmax", title: "Softmax each row into attention weights", tex: "A = \\text{softmax}(S') \\text{ (row-wise)}" },
  { short: "Output", title: "Weighted sum of values", tex: "\\text{output} = AV" },
];

export default function AttentionLab() {
  const [mode, setMode] = useState<"sentence" | "example">("sentence");
  const [text, setText] = useState("the cat sat on the mat");
  const [causal, setCausal] = useState(true);
  const [positions, setPositions] = useState(true);
  const [stepIdx, setStepIdx] = useState(0);
  const [focus, setFocus] = useState(1);

  const tokens = useMemo(() => (text.match(/[\p{L}\p{N}']+/gu) ?? []).slice(0, 6), [text]);
  const n = tokens.length;

  const data = useMemo(() => {
    if (mode === "example") {
      return { X: null as Mat | null, ...EX, r: EX_R, qLabels: ["q"], kLabels: ["k₁", "k₂"] };
    }
    if (!n) return null;
    const X = tokens.map((t, i) => embed(t).map((v, j) => +(v + (positions ? posEnc(i)[j] : 0)).toFixed(2)));
    const Q = matmul(X, WQ), K = matmul(X, WK), V = matmul(X, WV);
    const labels = tokens.map((t, i) => `${i + 1}:${t}`);
    return { X, Q, K, V, r: attention(Q, K, V, causal), qLabels: labels, kLabels: labels };
  }, [mode, tokens, n, causal, positions]);

  const firstStep = mode === "example" ? 2 : 0;
  const lastStep = STEPS.length - 1;
  const cur = Math.max(stepIdx, firstStep);
  const skipMask = mode === "example" || !causal;
  const go = (d: number) => {
    let s = cur + d;
    if (s === 4 && skipMask) s += d;
    setStepIdx(Math.min(lastStep, Math.max(firstStep, s)));
  };

  if (!data) {
    return (
      <LabFrame id="attention" title="Self-attention lab" subtitle="Type a short sentence.">
        <input value={text} onChange={(e) => setText(e.target.value)} className="w-full rounded-lg border border-line bg-bg px-3 py-2" aria-label="Sentence" />
        <p className="mt-3 text-sm text-muted">Enter at least one word to see attention.</p>
      </LabFrame>
    );
  }

  const { X, Q, K, V, r, qLabels, kLabels } = data;
  const nq = Q.length, nk = K.length;
  const fi = Math.min(focus, nq - 1);
  const dims = (k: number, p = "d") => Array.from({ length: k }, (_, i) => `${p}${i + 1}`);

  const detail = (() => {
    const q = Q[fi];
    if (cur <= 1) return null;
    if (cur === 2 || cur === 3) {
      const terms = K.map((k, j) => `${kLabels[j].split(":")[1] ?? kLabels[j]}: ${q.map((v, d) => `${fmt(v, 2)}\\cdot${fmt(k[d], 2)}`).join("+")} = ${fmt(r.scores[fi][j], 3)}`);
      return cur === 2
        ? <>Row {fi + 1} is the dot product of query {qLabels[fi]} with every key: <Tex>{terms.slice(0, 3).join(",\\; ")}</Tex>{nk > 3 ? " …" : ""}</>
        : <>Dividing by <Tex>{`\\sqrt{${D_K}} \\approx ${fmt(Math.sqrt(D_K), 3)}`}</Tex> keeps scores from growing with dimension, so softmax does not saturate.</>;
    }
    if (cur === 4) return <>Token {fi + 1} may only look at tokens 1 to {fi + 1}. Future positions get −∞, which softmax turns into exactly 0.</>;
    if (cur === 5) {
      const row = r.masked[fi];
      const ex = row.map((v) => (Number.isFinite(v) ? Math.exp(v - Math.max(...row.filter(Number.isFinite))) : 0));
      return <>Row {fi + 1}: exponentiate (after subtracting the max) to <Tex>{`[${ex.map((v) => fmt(v, 3)).join(", ")}]`}</Tex>, then divide by their sum. Each row of A sums to 1.</>;
    }
    const terms = r.weights[fi].map((w, j) => `${fmt(w, 2)}\\,v_{${j + 1}}`).join(" + ");
    return <>Output for {qLabels[fi]} <Tex>{`= ${terms} = [${r.output[fi].map((v) => fmt(v, 3)).join(", ")}]`}</Tex>. It is a blend of value vectors, weighted by attention.</>;
  })();

  // Guided tour (Watch mode + explainers). No causal mask, so every token sees every other.
  const setup = (step: number, f = 1, m: "sentence" | "example" = "sentence") => {
    setMode(m); setText("the cat sat on the mat"); setCausal(false); setPositions(true); setStepIdx(step); setFocus(f);
  };
  const sweep = (t: number) => setFocus(Math.round(t * (nq - 1)));
  const tour: TourStep[] = [
    { id: "embed", caption: "Each word becomes a row of four numbers: a toy embedding plus a position signal. The orange outline marks “cat”, the token we follow through every step.", apply: () => setup(0) },
    { id: "qkv", caption: "Three weight matrices turn every row into a query, a key and a value, two numbers each. The query asks, the key advertises, the value is what gets passed on.", apply: () => setup(1) },
    { id: "scores", caption: "Each cell of S is one query dotted with one key: large when the two point the same way. Watch the outline walk down the rows as every token scores every other token.", apply: () => setup(2, 0), animate: sweep, animMs: 2600 },
    { id: "scale", caption: "Every score is divided by √2, the square root of the key width. With 64-wide keys raw scores get large, and this keeps softmax from locking onto one word.", apply: () => setup(3) },
    { id: "softmax", caption: "Softmax turns each row into weights that sum to 1. Darker blue cells (columns are the words being looked at) show where each token pays most attention; follow the outline row by row.", apply: () => setup(5, 0), animate: sweep, animMs: 2600 },
    { id: "output", caption: "Each output row is a blend of the value rows, mixed in exactly those proportions. The new vector for “cat” now carries a little of every word it attended to.", apply: () => setup(6) },
    { id: "referent", caption: `The course example: the query matches key 1, so it gets weight ${fmt(EX_R.weights[0][0], 2)} and the output, [${EX_R.output[0].map((v) => fmt(v, 2)).join(", ")}], is mostly value 1. Training tunes the projections so “it” matches “animal” the same way.`, apply: () => setup(6, 0, "example") },
  ];

  return (
    <LabFrame
      id="attention"
      title="Self-attention lab"
      tour={tour}
      subtitle={mode === "example" ? "The course's worked example: q = [1, 0], keys [1, 0] and [0, 1], values [10, 0] and [0, 6], d_k = 2." : "Toy embeddings (d_model = 4) and random projections (d_k = 2), fixed seeds. Real models learn these."}
      onReset={() => { setMode("sentence"); setText("the cat sat on the mat"); setCausal(true); setPositions(true); setStepIdx(0); setFocus(1); }}
      presets={[
        { label: "Course example", apply: () => { setMode("example"); setStepIdx(2); setFocus(0); } },
        { label: "Sentence", apply: () => { setMode("sentence"); setStepIdx(0); } },
        { label: "Jump to heatmap", apply: () => setStepIdx(5) },
      ]}
      controls={
        <>
          <Segmented label="Input" value={mode} onChange={(v) => { setMode(v); setStepIdx(v === "example" ? 2 : 0); setFocus(v === "example" ? 0 : 1); }}
            options={[{ value: "sentence", label: "Sentence" }, { value: "example", label: "Course example" }]} />
          {mode === "sentence" && (
            <>
              <label className="block">
                <span className="text-[0.8rem] text-muted">Sentence (up to 6 tokens)</span>
                <input value={text} onChange={(e) => setText(e.target.value)} className="mt-1 w-full rounded-lg border border-line bg-bg px-2.5 py-1.5 text-sm" />
              </label>
              <Toggle label="Causal mask (decoder)" checked={causal} onChange={setCausal} />
              <Toggle label="Add positional encoding" checked={positions} onChange={setPositions} />
            </>
          )}
          <label className="block">
            <span className="text-[0.8rem] text-muted">Focus on query</span>
            <select value={fi} onChange={(e) => setFocus(Number(e.target.value))} className="mt-1 w-full rounded-lg border border-line bg-bg px-2 py-1.5 text-sm">
              {qLabels.map((l, i) => <option key={i} value={i}>{l}</option>)}
            </select>
          </label>
          <div className="flex gap-2">
            <Button onClick={() => go(-1)} disabled={cur === firstStep}>Back</Button>
            <Button primary onClick={() => go(1)} disabled={cur === lastStep}>Next step</Button>
          </div>
        </>
      }
      readout={
        <>
          <Stat label="step" value={`${cur + 1} / ${STEPS.length}`} />
          <Stat label="tokens n" value={nq === nk ? nq : `${nq} q, ${nk} k`} />
          <Stat label="d_k" value={D_K} />
          <Stat label="weights row sum" value={fmt(r.weights[fi].reduce((a, b) => a + b, 0), 3)} />
        </>
      }
      interpretation={<><p className="text-ink font-medium">{STEPS[cur].title}</p><p className="mt-1">{detail ?? "Each token's embedding is a vector. Q, K and V are three different linear views of it: what this token is looking for, what it offers, and what it passes along."}</p></>}
    >
      <ol className="flex flex-wrap gap-1 mb-4 text-xs" aria-label="Steps">
        {STEPS.map((s, i) => {
          const disabled = i < firstStep || (i === 4 && skipMask);
          return (
            <li key={i}>
              <button type="button" disabled={disabled} onClick={() => setStepIdx(i)} aria-current={i === cur ? "step" : undefined}
                className={`rounded-full px-2.5 py-1 border ${i === cur ? "border-accent bg-accent-soft text-accent" : i < cur ? "border-line text-ink" : "border-line text-faint"} disabled:opacity-30`}>
                {i + 1}. {s.short}
              </button>
            </li>
          );
        })}
      </ol>
      <div className="rounded-lg bg-surface-2/60 px-3 py-2 mb-4"><Tex display>{STEPS[cur].tex}</Tex></div>

      <div className="grid gap-5 @xl:grid-cols-2">
        {cur === 0 && X && <MatrixView m={X} rows={qLabels} cols={dims(D_MODEL)} title="X: token embeddings + positions" shape={`[${n}, ${D_MODEL}]`} focusRow={fi} />}
        {cur === 0 && X && <MatrixView m={WQ} rows={dims(D_MODEL)} cols={dims(D_K, "k")} title="W_Q (learned in real models)" shape={`[${D_MODEL}, ${D_K}]`} />}
        {cur === 1 && <MatrixView m={Q} rows={qLabels} cols={dims(D_K)} title="Q = X W_Q" shape={`[${nq}, ${D_K}]`} focusRow={fi} />}
        {cur === 1 && <MatrixView m={K} rows={kLabels} cols={dims(D_K)} title="K = X W_K" shape={`[${nk}, ${D_K}]`} />}
        {cur === 1 && <MatrixView m={V} rows={kLabels} cols={dims(D_K)} title="V = X W_V" shape={`[${nk}, ${D_K}]`} />}
        {cur >= 2 && cur <= 3 && <MatrixView m={Q} rows={qLabels} cols={dims(D_K)} title="Q" shape={`[${nq}, ${D_K}]`} focusRow={fi} />}
        {cur >= 2 && cur <= 3 && <MatrixView m={transpose(K)} rows={dims(D_K)} cols={kLabels} title="Kᵀ" shape={`[${D_K}, ${nk}]`} />}
        {cur === 2 && <MatrixView m={r.scores} rows={qLabels} cols={kLabels} title="S = QKᵀ" shape={`[${nq}, ${nk}]`} focusRow={fi} digits={3} />}
        {cur === 3 && <MatrixView m={r.scaled} rows={qLabels} cols={kLabels} title="S / √d_k" shape={`[${nq}, ${nk}]`} focusRow={fi} digits={3} />}
        {cur === 4 && <MatrixView m={r.masked} rows={qLabels} cols={kLabels} title="masked scores" shape={`[${nq}, ${nk}]`} focusRow={fi} digits={3} />}
        {cur >= 5 && <MatrixView m={r.weights} rows={qLabels} cols={kLabels} title="A = softmax(S′): attention heatmap" shape={`[${nq}, ${nk}]`} heat focusRow={fi} digits={2} />}
        {cur === 6 && <MatrixView m={V} rows={kLabels} cols={dims(D_K)} title="V" shape={`[${nk}, ${D_K}]`} />}
        {cur === 6 && <MatrixView m={r.output} rows={qLabels} cols={dims(D_K)} title="output = A V" shape={`[${nq}, ${D_K}]`} focusRow={fi} digits={3} />}
      </div>
    </LabFrame>
  );
}
