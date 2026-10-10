"use client";
import { useEffect, useMemo, useState } from "react";
import { fmt, gaussian, gelu, relu, rng, sigmoid, softmax, tanh } from "@/lib/ml";
import { Button, LabFrame, Segmented, Slider, Stat, Tex, type TourStep } from "./ui";

type Act = "relu" | "sigmoid" | "tanh" | "gelu";
const ACTS: Record<Act, (z: number) => number> = { relu, sigmoid, tanh, gelu };
const ACT_TEX: Record<Act, string> = { relu: "\\max(0, z)", sigmoid: "1/(1+e^{-z})", tanh: "\\tanh(z)", gelu: "z\\,\\Phi(z)" };

type Params = { W1: number[][]; b1: number[]; W2: number[][]; b2: number[] };
// The lesson's worked example (2-3-1, x = [1, 2]), so the default lab matches the hand calculation.
const COURSE: Params = { W1: [[0.5, -1], [1, 1], [-0.5, 0.25]], b1: [0, -1, 0.5], W2: [[1, -0.5, 2]], b2: [0.25] };
const X0 = [1, 2, 0.5, -0.5];

function initWeights(nIn: number, nH: number, nOut: number): Params {
  if (nIn === 2 && nH === 3 && nOut === 1) return COURSE;
  const r = rng(nIn * 100 + nH * 10 + nOut);
  const m = (rows: number, cols: number) => Array.from({ length: rows }, () => Array.from({ length: cols }, () => +(gaussian(r) * 0.9).toFixed(2)));
  return { W1: m(nH, nIn), b1: Array(nH).fill(0).map(() => +(gaussian(r) * 0.3).toFixed(2)), W2: m(nOut, nH), b2: Array(nOut).fill(0).map(() => +(gaussian(r) * 0.3).toFixed(2)) };
}

function forward(p: Params, xin: number[], act: Act, nOut: number) {
  const z1 = p.W1.map((row, i) => row.reduce((s, w, j) => s + w * xin[j], 0) + p.b1[i]);
  const a1 = z1.map(ACTS[act]);
  const z2 = p.W2.map((row, i) => row.reduce((s, w, j) => s + w * a1[j], 0) + p.b2[i]);
  const y = nOut === 1 ? [sigmoid(z2[0])] : softmax(z2);
  return { z1, a1, z2, y };
}
const paren = (v: number, d = 2) => (v < 0 ? `(${fmt(v, d)})` : fmt(v, d));
const n3 = (v: number) => String(+v.toFixed(3)).replace("-", "−"); // captions: 0.5, −1.5, 0.562

type Sel = { layer: 1 | 2; i: number; j: number } | null;
const W = 520, H = 300;

export default function NeuralNetLab() {
  const [nIn, setNIn] = useState(2);
  const [nH, setNH] = useState(3);
  const [nOut, setNOut] = useState(1);
  const [act, setAct] = useState<Act>("relu");
  const [x, setX] = useState(X0);
  const [params, setParams] = useState(() => initWeights(2, 3, 1));
  const [sel, setSel] = useState<Sel>(null);
  const [focus, setFocus] = useState(0);
  const [stage, setStage] = useState(3); // 0 inputs, 1 hidden z, 2 hidden a, 3 output
  const [playing, setPlaying] = useState(false);

  const resize = (a: number, b: number, c: number) => { setNIn(a); setNH(b); setNOut(c); setParams(initWeights(a, b, c)); setSel(null); setFocus(0); };

  const xin = x.slice(0, nIn);
  const fwd = useMemo(() => forward(params, xin, act, nOut), [params, xin, act, nOut]);

  useEffect(() => {
    if (!playing) return;
    const t = setTimeout(() => {
      setStage((s) => s + 1);
      if (stage + 1 >= 3) setPlaying(false);
    }, 700);
    return () => clearTimeout(t);
  }, [playing, stage]);

  const col = (k: number) => 70 + k * ((W - 140) / 2);
  const rowY = (i: number, n: number) => H / 2 + (i - (n - 1) / 2) * Math.min(64, (H - 60) / Math.max(1, n - 1 || 1));
  const shown = (layerStage: number) => stage >= layerStage;

  const selW = sel ? (sel.layer === 1 ? params.W1[sel.i]?.[sel.j] : params.W2[sel.i]?.[sel.j]) : undefined;
  const setSelW = (v: number) => {
    if (!sel) return;
    setParams((p) => {
      const key = sel.layer === 1 ? "W1" : "W2";
      const m = p[key].map((r) => [...r]);
      m[sel.i][sel.j] = v;
      return { ...p, [key]: m };
    });
  };

  const h = Math.min(focus, nH - 1);
  const zTerms = params.W1[h].map((w, j) => `${paren(w)} \\cdot ${paren(xin[j])}`).join(" + ");

  // Guided tour (Watch mode + explainers). setup() resets to the lesson's 2-3-1 example (or 2-3-3), then sets every control.
  // Captions are computed from the same forward pass the diagram uses.
  const setup = (o: { act?: Act; out?: number; stage?: number; focus?: number } = {}) => {
    resize(2, 3, o.out ?? 1); setAct(o.act ?? "relu"); setX(X0);
    setPlaying(false); setStage(o.stage ?? 3); setFocus(o.focus ?? 0);
  };
  const x0 = X0.slice(0, 2), c = forward(COURSE, x0, "relu", 1), cs = forward(COURSE, x0, "sigmoid", 1);
  const soft = forward(initWeights(2, 3, 3), x0, "relu", 3).y;
  const zSum = (i: number) => COURSE.W1[i].map((w, j) => `${n3(w)} × ${n3(x0[j])}`).concat(n3(COURSE.b1[i])).join(" + ").replace(/\+ −/g, "− ");
  const W2off = COURSE.W2[0][0];
  const tour: TourStep[] = [
    { id: "layout", caption: `The lesson's worked example: two inputs, x = [${x0.map(n3).join(", ")}], three hidden neurons, one output. Blue lines are positive weights, orange are negative, and thicker means bigger.`, apply: () => setup() },
    { id: "flow", caption: `Numbers flow left to right. Each hidden neuron computes z, a weighted sum plus bias, then a = ReLU(z); the output squashes its own sum (z = ${n3(c.z2[0])}) through a sigmoid to ${n3(c.y[0])}.`, apply: () => setup({ stage: 0 }), animate: (t) => setStage(Math.round(t * 3)), animMs: 2400 },
    { id: "one-neuron", caption: `Zoom in on h2, outlined in green: z = ${zSum(1)} = ${n3(c.z1[1])}. It is positive, so ReLU passes it through unchanged: a = ${n3(c.a1[1])}.`, apply: () => setup({ focus: 1 }) },
    { id: "relu-off", caption: `Now h1: z = ${zSum(0)} = ${n3(c.z1[0])}. It is negative, so ReLU outputs exactly 0 and the circle stays unshaded. Watch its outgoing weight swing from ${n3(W2off)} to −2: the output stays at ${n3(c.y[0])}, because 0 times anything is 0.`, apply: () => { setup({ focus: 0 }); setSel({ layer: 2, i: 0, j: 0 }); }, animate: (t) => setParams((p) => ({ ...p, W2: [[+(W2off + (-2 - W2off) * t).toFixed(2), ...p.W2[0].slice(1)]] })), animMs: 3000 },
    { id: "sigmoid", caption: `Switch the hidden activation to sigmoid. h1's z is still ${n3(c.z1[0])}, but σ(${n3(c.z1[0])}) = ${n3(cs.a1[0])}, not 0: sigmoid values stay between 0 and 1 and are never exactly zero, so every neuron contributes. The output moves to ${n3(cs.y[0])}.`, apply: () => setup({ act: "sigmoid", focus: 0 }) },
    { id: "softmax", caption: `With three outputs the last layer uses softmax instead: three teal circles, one probability per class, that always add up to 1. Here they are ${soft.map((v) => fmt(v, 3)).join(", ")}.`, apply: () => setup({ out: 3 }) },
  ];

  return (
    <LabFrame
      id="nn-forward"
      title="Forward pass lab"
      tour={tour}
      subtitle="A fully connected network. Click any weight (line) to edit it, or any hidden neuron to see its arithmetic."
      onReset={() => { resize(2, 3, 1); setAct("relu"); setX(X0); setStage(3); }}
      presets={[
        { label: "Animate forward pass", apply: () => { setStage(0); setPlaying(true); } },
        { label: "3 classes (softmax)", apply: () => resize(nIn, nH, 3) },
        { label: "Wider hidden layer", apply: () => resize(nIn, 5, nOut) },
      ]}
      controls={
        <>
          <div className="grid grid-cols-3 gap-2">
            <Slider label="inputs" value={nIn} min={1} max={4} onChange={(v) => resize(v, nH, nOut)} />
            <Slider label="hidden" value={nH} min={1} max={5} onChange={(v) => resize(nIn, v, nOut)} />
            <Slider label="outputs" value={nOut} min={1} max={3} onChange={(v) => resize(nIn, nH, v)} />
          </div>
          <Segmented label="Hidden activation" value={act} onChange={setAct} options={[{ value: "relu", label: "ReLU" }, { value: "sigmoid", label: "Sigmoid" }, { value: "tanh", label: "Tanh" }, { value: "gelu", label: "GELU" }]} />
          {xin.map((v, j) => (
            <Slider key={j} label={`input x${j + 1}`} value={v} min={-3} max={3} step={0.1} onChange={(nv) => setX((cur) => cur.map((c, k) => (k === j ? nv : c)))} format={(v) => fmt(v, 1)} />
          ))}
          {sel && selW !== undefined && (
            <Slider label={sel.layer === 1 ? `weight W1[h${sel.i + 1}, x${sel.j + 1}]` : `weight W2[y${sel.i + 1}, h${sel.j + 1}]`} value={selW} min={-3} max={3} step={0.05} onChange={setSelW} format={(v) => fmt(v, 2)} hint="Watch the output change as you drag." />
          )}
          <Button onClick={() => { setStage(0); setPlaying(true); }}>Animate</Button>
        </>
      }
      readout={
        <>
          <Stat label="x" value={`[${nIn}]`} />
          <Stat label="W1 · b1" value={`[${nH}, ${nIn}] · [${nH}]`} />
          <Stat label="h = f(W1x + b1)" value={`[${nH}]`} />
          <Stat label="W2 · b2" value={`[${nOut}, ${nH}] · [${nOut}]`} />
          <Stat label="ŷ" value={`[${nOut}]`} />
          <Stat label="output" value={fwd.y.map((v) => fmt(v, 3)).join(", ")} color="var(--c-teal)" />
        </>
      }
      interpretation={
        <>
          <p className="text-ink">Hidden neuron h{h + 1}: <Tex>{`z = ${zTerms} + ${paren(params.b1[h])} = ${fmt(fwd.z1[h], 3)}`}</Tex>, then <Tex>{`a = ${ACT_TEX[act]} = ${fmt(fwd.a1[h], 3)}`}</Tex>.</p>
          <p className="mt-1">{act === "relu" && fwd.z1.some((z) => z <= 0)
            ? `ReLU outputs exactly 0 for ${fwd.z1.filter((z) => z <= 0).length} hidden neuron(s) right now, so their outgoing weights have no effect on this input.`
            : "Each layer is a linear map (weights and bias) followed by a nonlinearity. Without the nonlinearity, stacked layers collapse into a single linear map."}
            {nOut > 1 ? ` The output uses softmax, so the ${nOut} values sum to 1.` : " The output uses a sigmoid, giving a probability for the positive class."}</p>
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Network diagram">
        <title>Neural network diagram</title>
        {params.W1.map((row, i) => row.map((w, j) => {
          const on = sel?.layer === 1 && sel.i === i && sel.j === j;
          return (
            <line key={`a${i}${j}`} x1={col(0)} y1={rowY(j, nIn)} x2={col(1)} y2={rowY(i, nH)}
              stroke={w >= 0 ? "var(--c-blue)" : "var(--c-orange)"} strokeWidth={on ? 5 : 1 + Math.abs(w) * 1.6}
              opacity={shown(1) ? (on ? 1 : 0.55) : 0.12} className="cursor-pointer" onClick={() => setSel({ layer: 1, i, j })}
              style={{ transition: "opacity 300ms" }}>
              <title>{`W1[h${i + 1}, x${j + 1}] = ${w}`}</title>
            </line>
          );
        }))}
        {params.W2.map((row, i) => row.map((w, j) => {
          const on = sel?.layer === 2 && sel.i === i && sel.j === j;
          return (
            <line key={`b${i}${j}`} x1={col(1)} y1={rowY(j, nH)} x2={col(2)} y2={rowY(i, nOut)}
              stroke={w >= 0 ? "var(--c-blue)" : "var(--c-orange)"} strokeWidth={on ? 5 : 1 + Math.abs(w) * 1.6}
              opacity={shown(3) ? (on ? 1 : 0.55) : 0.12} className="cursor-pointer" onClick={() => setSel({ layer: 2, i, j })}
              style={{ transition: "opacity 300ms" }}>
              <title>{`W2[y${i + 1}, h${j + 1}] = ${w}`}</title>
            </line>
          );
        }))}
        {xin.map((v, j) => (
          <g key={`x${j}`} transform={`translate(${col(0)} ${rowY(j, nIn)})`}>
            <circle r={20} fill="var(--surface)" stroke="var(--muted)" strokeWidth={1.5} />
            <text textAnchor="middle" dy="4" fontSize="11" fill="var(--text)" fontFamily="var(--font-geist-mono)">{fmt(v, 1)}</text>
            <text textAnchor="end" x={-26} dy="4" fontSize="11" fill="var(--faint)">x{j + 1}</text>
          </g>
        ))}
        {fwd.a1.map((a, i) => (
          <g key={`h${i}`} transform={`translate(${col(1)} ${rowY(i, nH)})`} className="cursor-pointer" onClick={() => setFocus(i)}>
            <circle r={22} fill={`color-mix(in srgb, var(--c-purple) ${shown(2) ? Math.round(Math.min(1, Math.abs(a)) * 60) : 0}%, var(--surface))`} stroke={i === h ? "var(--c-green)" : "var(--c-purple)"} strokeWidth={i === h ? 3 : 1.5} />
            <text textAnchor="middle" dy="-2" fontSize="9" fill="var(--text)" fontFamily="var(--font-geist-mono)">{shown(1) ? `z ${fmt(fwd.z1[i], 2)}` : ""}</text>
            <text textAnchor="middle" dy="10" fontSize="9" fill="var(--text)" fontFamily="var(--font-geist-mono)">{shown(2) ? `a ${fmt(a, 2)}` : ""}</text>
          </g>
        ))}
        {fwd.y.map((y, i) => (
          <g key={`y${i}`} transform={`translate(${col(2)} ${rowY(i, nOut)})`}>
            <circle r={22} fill={`color-mix(in srgb, var(--c-teal) ${shown(3) ? Math.round(y * 70) : 0}%, var(--surface))`} stroke="var(--c-teal)" strokeWidth={1.5} />
            <text textAnchor="middle" dy="4" fontSize="11" fill="var(--text)" fontFamily="var(--font-geist-mono)">{shown(3) ? fmt(y, 3) : ""}</text>
            <text x={30} dy="4" fontSize="11" fill="var(--faint)">ŷ{nOut > 1 ? i + 1 : ""}</text>
          </g>
        ))}
        {["input [" + nIn + "]", "hidden [" + nH + "]", "output [" + nOut + "]"].map((t, k) => (
          <text key={t} x={col(k)} y={H - 6} textAnchor="middle" fontSize="11" fill="var(--muted)">{t}</text>
        ))}
      </svg>
      <p className="text-xs text-muted mt-1">Blue lines are positive weights, orange are negative; thickness is magnitude. Hidden-neuron shading shows activation size; the green outline marks the neuron whose arithmetic is shown below.</p>
    </LabFrame>
  );
}
