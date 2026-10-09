"use client";
import { useEffect, useMemo, useState } from "react";
import { fmt, gaussian, gelu, relu, rng, sigmoid, softmax, tanh } from "@/lib/ml";
import { Button, LabFrame, Segmented, Slider, Stat, Tex, type TourStep } from "./ui";

type Act = "relu" | "sigmoid" | "tanh" | "gelu";
const ACTS: Record<Act, (z: number) => number> = { relu, sigmoid, tanh, gelu };
const ACT_TEX: Record<Act, string> = { relu: "\\max(0, z)", sigmoid: "1/(1+e^{-z})", tanh: "\\tanh(z)", gelu: "z\\,\\Phi(z)" };

function initWeights(nIn: number, nH: number, nOut: number) {
  const r = rng(nIn * 100 + nH * 10 + nOut);
  const m = (rows: number, cols: number) => Array.from({ length: rows }, () => Array.from({ length: cols }, () => +(gaussian(r) * 0.9).toFixed(2)));
  return { W1: m(nH, nIn), b1: Array(nH).fill(0).map(() => +(gaussian(r) * 0.3).toFixed(2)), W2: m(nOut, nH), b2: Array(nOut).fill(0).map(() => +(gaussian(r) * 0.3).toFixed(2)) };
}

type Sel = { layer: 1 | 2; i: number; j: number } | null;
const W = 520, H = 300;

export default function NeuralNetLab() {
  const [nIn, setNIn] = useState(2);
  const [nH, setNH] = useState(3);
  const [nOut, setNOut] = useState(1);
  const [act, setAct] = useState<Act>("relu");
  const [x, setX] = useState([1, -0.5, 0.5, 2]);
  const [params, setParams] = useState(() => initWeights(2, 3, 1));
  const [sel, setSel] = useState<Sel>({ layer: 1, i: 0, j: 0 });
  const [focus, setFocus] = useState(0);
  const [stage, setStage] = useState(3); // 0 inputs, 1 hidden z, 2 hidden a, 3 output
  const [playing, setPlaying] = useState(false);

  const resize = (a: number, b: number, c: number) => { setNIn(a); setNH(b); setNOut(c); setParams(initWeights(a, b, c)); setSel({ layer: 1, i: 0, j: 0 }); setFocus(0); };

  const xin = x.slice(0, nIn);
  const fwd = useMemo(() => {
    const z1 = params.W1.map((row, i) => row.reduce((s, w, j) => s + w * xin[j], 0) + params.b1[i]);
    const a1 = z1.map(ACTS[act]);
    const z2 = params.W2.map((row, i) => row.reduce((s, w, j) => s + w * a1[j], 0) + params.b2[i]);
    const y = nOut === 1 ? [sigmoid(z2[0])] : softmax(z2);
    return { z1, a1, z2, y };
  }, [params, xin, act, nOut]);

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
  const zTerms = params.W1[h].map((w, j) => `${fmt(w, 2)} \\cdot ${fmt(xin[j], 2)}`).join(" + ");

  // Guided tour (Watch mode + explainers). setup() resets the 2-3-1 net (or 2-3-3), then sets every control.
  const setup = (o: { act?: Act; x1?: number; out?: number; stage?: number; focus?: number } = {}) => {
    resize(2, 3, o.out ?? 1); setAct(o.act ?? "relu"); setX([o.x1 ?? 1, -0.5, 0.5, 2]);
    setPlaying(false); setStage(o.stage ?? 3); setFocus(o.focus ?? 0);
  };
  const tour: TourStep[] = [
    { id: "layout", caption: "Two inputs on the left, three hidden neurons, one output. Blue lines are positive weights, orange are negative, and thicker means bigger.", apply: () => setup() },
    { id: "flow", caption: "Numbers flow left to right. Each hidden neuron computes z, a weighted sum plus bias, then a = ReLU(z); the output squashes its own sum through a sigmoid to 0.254.", apply: () => setup({ stage: 0 }), animate: (t) => setStage(Math.round(t * 3)), animMs: 2400 },
    { id: "one-neuron", caption: "Zoom in on h3, outlined in orange: z = 0.53 × 1 + 0.32 × (−0.5) + 0.22 = 0.59. It is positive, so ReLU passes it through unchanged.", apply: () => setup({ focus: 2 }) },
    { id: "relu-off", caption: "Slide x1 from 1 down to −2. Once z for h3 turns negative, ReLU outputs exactly 0, its circle goes pale, and its outgoing weight stops affecting the output.", apply: () => setup({ focus: 2 }), animate: (t) => setX((c) => [+(1 - 3 * t).toFixed(1), ...c.slice(1)]), animMs: 3000 },
    { id: "sigmoid", caption: "Switch the hidden activation to sigmoid. Hidden values now stay between 0 and 1 and are never exactly zero, so every neuron always contributes something.", apply: () => setup({ act: "sigmoid", x1: -2, focus: 2 }) },
    { id: "softmax", caption: "With three outputs the last layer uses softmax instead: three teal circles whose values always add up to 1, one probability per class.", apply: () => setup({ out: 3 }) },
  ];

  return (
    <LabFrame
      id="nn-forward"
      title="Forward pass lab"
      tour={tour}
      subtitle="A fully connected network. Click any weight (line) to edit it, or any hidden neuron to see its arithmetic."
      onReset={() => { resize(2, 3, 1); setAct("relu"); setX([1, -0.5, 0.5, 2]); setStage(3); }}
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
          <p className="text-ink">Hidden neuron h{h + 1}: <Tex>{`z = ${zTerms} + ${fmt(params.b1[h], 2)} = ${fmt(fwd.z1[h], 3)}`}</Tex>, then <Tex>{`a = ${ACT_TEX[act]} = ${fmt(fwd.a1[h], 3)}`}</Tex>.</p>
          <p className="mt-1">{act === "relu" && fwd.z1.some((z) => z <= 0)
            ? `ReLU outputs exactly 0 for ${fwd.z1.filter((z) => z <= 0).length} hidden neuron(s) right now, so their outgoing weights have no effect on this input.`
            : "Each layer is a linear map (weights and bias) followed by a nonlinearity. Without the nonlinearity, stacked layers collapse into a single linear map."}
            {nOut > 1 ? " The output uses softmax, so the three values sum to 1." : " The output uses a sigmoid, giving a probability for the positive class."}</p>
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
            <circle r={22} fill={`color-mix(in srgb, var(--c-purple) ${shown(2) ? Math.round(Math.min(1, Math.abs(a)) * 60) : 0}%, var(--surface))`} stroke={i === h ? "var(--c-orange)" : "var(--c-purple)"} strokeWidth={i === h ? 3 : 1.5} />
            <text textAnchor="middle" dy="-2" fontSize="10" fill="var(--text)" fontFamily="var(--font-geist-mono)">{shown(1) ? `z ${fmt(fwd.z1[i], 2)}` : ""}</text>
            <text textAnchor="middle" dy="11" fontSize="10" fill="var(--text)" fontFamily="var(--font-geist-mono)">{shown(2) ? `a ${fmt(a, 2)}` : ""}</text>
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
      <p className="text-xs text-muted mt-1">Blue lines are positive weights, orange are negative; thickness is magnitude. Hidden-neuron shading shows activation size.</p>
    </LabFrame>
  );
}
