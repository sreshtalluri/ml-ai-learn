"use client";
import { useMemo, useState } from "react";
import { fmt, rng, softmax, topK, topP } from "@/lib/ml";
import { Button, LabFrame, Slider, Stat, Tex } from "./ui";

// SYNTHETIC next-token logits after the prompt "The capital of France is". Illustrative values, not from a real model.
const PROMPT = "The capital of France is";
const TOKENS = [" Paris", " a", " the", " located", " known", " Lyon", " not", " beautiful", " France", " Marseille"];
const LOGITS = [6.1, 3.2, 2.9, 2.4, 2.1, 1.6, 1.2, 1.0, 0.6, 0.4];

export default function DecodingLab() {
  const [T, setT] = useState(1);
  const [k, setK] = useState(10);
  const [p, setP] = useState(1);
  const [seed, setSeed] = useState(1);
  const [samples, setSamples] = useState<string[]>([]);

  const probs = useMemo(() => softmax(LOGITS, T), [T]);
  const afterK = useMemo(() => topK(probs, k), [probs, k]);
  const final = useMemo(() => topP(afterK, p), [afterK, p]);
  const kept = final.filter((v) => v > 0).length;
  const entropy = -final.reduce((s, v) => s + (v > 0 ? v * Math.log2(v) : 0), 0);
  const greedy = TOKENS[final.indexOf(Math.max(...final))];

  const sample = (n: number) => {
    const r = rng(seed * 7919 + samples.length);
    const out: string[] = [];
    for (let i = 0; i < n; i++) {
      let u = r(), idx = 0;
      while ((u -= final[idx]) > 0 && idx < final.length - 1) idx++;
      out.push(TOKENS[idx]);
    }
    setSamples((s) => [...out, ...s].slice(0, 30));
    setSeed((x) => x + 1);
  };

  const counts = TOKENS.map((t) => samples.filter((s) => s === t).length);

  return (
    <LabFrame
      id="decoding"
      title="Decoding lab"
      subtitle={`Prompt: “${PROMPT} …”. Synthetic logits for 10 candidate tokens (illustrative, not from a real model).`}
      onReset={() => { setT(1); setK(10); setP(1); setSamples([]); }}
      presets={[
        { label: "Greedy-like (T = 0.1)", apply: () => { setT(0.1); setK(10); setP(1); setSamples([]); } },
        { label: "Creative (T = 1.8)", apply: () => { setT(1.8); setK(10); setP(1); setSamples([]); } },
        { label: "Nucleus p = 0.9", apply: () => { setT(1); setK(10); setP(0.9); setSamples([]); } },
      ]}
      controls={
        <>
          <Slider label="temperature T" value={T} min={0.05} max={2.5} step={0.05} onChange={(v) => { setT(v); setSamples([]); }} format={(v) => fmt(v, 2)} />
          <Slider label="top-k" value={k} min={1} max={10} onChange={(v) => { setK(v); setSamples([]); }} />
          <Slider label="top-p (nucleus)" value={p} min={0.05} max={1} step={0.01} onChange={(v) => { setP(v); setSamples([]); }} format={(v) => fmt(v, 2)} />
          <div className="flex gap-2"><Button primary onClick={() => sample(1)}>Sample 1</Button><Button onClick={() => sample(20)}>Sample 20</Button></div>
        </>
      }
      readout={
        <>
          <Stat label="candidates kept" value={`${kept} / ${TOKENS.length}`} />
          <Stat label="greedy pick" value={`“${greedy.trim()}”`} />
          <Stat label="P(Paris)" value={fmt(final[0], 3)} color="var(--c-blue)" />
          <Stat label="entropy (bits)" value={fmt(entropy, 2)} />
        </>
      }
      interpretation={
        <>
          <p className="text-ink">Pipeline: <Tex>{"\\text{logits} \\xrightarrow{\\;/T\\;} \\text{softmax} \\xrightarrow{\\text{top-}k} \\xrightarrow{\\text{top-}p} \\text{renormalize} \\to \\text{sample}"}</Tex></p>
          <p className="mt-1">{T < 0.5 ? "Low temperature sharpens the distribution toward the top logit: output becomes nearly deterministic (and can get repetitive)." : T > 1.3 ? "High temperature flattens the distribution: unlikely tokens like “beautiful” get real probability, so output becomes more varied and more error-prone." : "At T = 1 the model's own distribution is used unchanged."}{" "}
            These probabilities describe which token tends to come next in text, not whether a statement is true. A fluent wrong answer can have high probability.</p>
        </>
      }
    >
      <div className="space-y-1.5">
        {TOKENS.map((t, i) => {
          const removed = final[i] === 0;
          return (
            <div key={t} className="grid grid-cols-[6.5rem_minmax(0,1fr)_3.5rem_3.5rem] items-center gap-2 text-sm">
              <span className={`font-mono truncate ${removed ? "text-faint line-through" : ""}`}>“{t.trim()}”</span>
              <span className="relative h-5 rounded bg-surface-2 overflow-hidden">
                <span className="absolute inset-y-0 left-0 bg-faint/40" style={{ width: `${probs[i] * 100}%` }} />
                <span className="absolute inset-y-0 left-0 bg-blue transition-[width] duration-300" style={{ width: `${final[i] * 100}%`, opacity: 0.85 }} />
              </span>
              <span className="font-mono text-xs text-right">{fmt(final[i], 3)}</span>
              <span className="font-mono text-[11px] text-faint text-right">z={LOGITS[i]}</span>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-muted mt-2">Gray: softmax with temperature. Blue: after top-k and top-p filtering and renormalization.</p>
      {samples.length > 0 && (
        <div className="mt-4">
          <p className="text-sm font-medium">Samples (newest first)</p>
          <p className="mt-1 text-sm text-muted">{samples.map((s, i) => <span key={i} className="inline-block mr-1.5 rounded bg-surface-2 px-1.5 py-0.5 font-mono text-xs">{s.trim()}</span>)}</p>
          <p className="mt-1 text-xs text-faint">Counts: {TOKENS.map((t, i) => counts[i] ? `${t.trim()} ${counts[i]}` : null).filter(Boolean).join(", ")}</p>
        </div>
      )}
    </LabFrame>
  );
}
