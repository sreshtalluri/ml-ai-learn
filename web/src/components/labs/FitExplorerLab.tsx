"use client";
import { useMemo, useState } from "react";
import { fmt, gaussian, mean, polyEval, polyFit, rng } from "@/lib/ml";
import { LabFrame, Legend, Plot, Slider, Stat } from "./ui";

const truth = (x: number) => Math.sin(Math.PI * x);
const MAX_DEG = 15;

// SYNTHETIC: y = sin(πx) + noise on x ∈ [-1, 1]; fixed seeds so the picture is reproducible.
function makeData(n: number, noise: number, seed: number) {
  const r = rng(seed);
  return Array.from({ length: n }, () => {
    const x = r() * 2 - 1;
    return { x, y: truth(x) + gaussian(r) * noise };
  });
}

export default function FitExplorerLab() {
  const [degree, setDegree] = useState(3);
  const [n, setN] = useState(20);
  const [noise, setNoise] = useState(0.3);
  const train = useMemo(() => makeData(n, noise, 7), [n, noise]);
  const val = useMemo(() => makeData(200, noise, 99), [noise]);

  const mse = (c: number[], pts: { x: number; y: number }[]) => mean(pts.map((p) => (p.y - polyEval(c, p.x)) ** 2));
  const curve = useMemo(() => {
    return Array.from({ length: MAX_DEG }, (_, i) => {
      const d = i + 1;
      if (d >= n) return null; // more coefficients than points: not identifiable
      const c = polyFit(train.map((p) => p.x), train.map((p) => p.y), d);
      return { d, train: mse(c, train), val: mse(c, val) };
    }).filter((r) => r !== null);
  }, [train, val, n]);
  const coefs = useMemo(() => polyFit(train.map((p) => p.x), train.map((p) => p.y), Math.min(degree, n - 1)), [train, degree, n]);
  const cur = curve.find((r) => r.d === degree) ?? curve[curve.length - 1];
  const best = curve.reduce((a, b) => (b.val < a.val ? b : a));
  const xs = Array.from({ length: 160 }, (_, i) => -1 + (2 * i) / 159);
  const clampY = (v: number) => Math.max(-2.4, Math.min(2.4, v));
  const yMax = Math.max(...curve.map((r) => Math.max(r.train, r.val)));
  const logY = (v: number) => Math.log10(Math.max(v, 1e-4));

  const verdict = cur.val > best.val * 1.5 && degree > best.d
    ? `Overfitting: training error ${fmt(cur.train, 3)} keeps falling, but validation error ${fmt(cur.val, 3)} is ${fmt(cur.val / best.val, 1)}× the best. The curve is bending to fit noise.`
    : cur.val > best.val * 1.5 && degree < best.d
      ? `Underfitting: both errors are high (train ${fmt(cur.train, 3)}, validation ${fmt(cur.val, 3)}). A degree-${degree} polynomial can't bend enough to follow the sine.`
      : `Near the sweet spot: validation error ${fmt(cur.val, 3)} is close to the best (${fmt(best.val, 3)} at degree ${best.d}).`;

  return (
    <LabFrame
      id="fit-explorer"
      title="Underfitting vs overfitting"
      subtitle="Synthetic data from y = sin(πx) + noise. Blue dots train the model; 200 hidden validation points score it."
      onReset={() => { setDegree(3); setN(20); setNoise(0.3); }}
      presets={[
        { label: "Underfit (degree 1)", apply: () => setDegree(1) },
        { label: "Overfit (degree 12)", apply: () => setDegree(12) },
        { label: "More data", apply: () => setN(60) },
      ]}
      controls={
        <>
          <Slider label="polynomial degree" value={degree} min={1} max={MAX_DEG} onChange={setDegree} hint="Model complexity: number of coefficients minus one." />
          <Slider label="training points" value={n} min={8} max={60} onChange={setN} />
          <Slider label="noise σ" value={noise} min={0} max={0.8} step={0.05} onChange={setNoise} format={(v) => fmt(v, 2)} />
        </>
      }
      readout={
        <>
          <Stat label="train MSE" value={fmt(cur.train, 4)} color="var(--c-blue)" />
          <Stat label="validation MSE" value={fmt(cur.val, 4)} color="var(--c-orange)" />
          <Stat label="gap (val − train)" value={fmt(cur.val - cur.train, 4)} />
          <Stat label="best degree (val)" value={best.d} />
          <Stat label="noise floor σ²" value={fmt(noise * noise, 4)} />
        </>
      }
      interpretation={verdict}
    >
      <div className="grid gap-4 @3xl:grid-cols-2">
        <div>
          <Plot title="Training data and fitted polynomial" x={[-1, 1]} y={[-2.4, 2.4]} xLabel="x" yLabel="y" height={300}>
            {({ sx, sy }) => (
              <>
                <polyline fill="none" stroke="var(--faint)" strokeDasharray="5 4" strokeWidth={1.5} points={xs.map((x) => `${sx(x)},${sy(truth(x))}`).join(" ")} />
                <polyline fill="none" stroke="var(--c-teal)" strokeWidth={2.5} points={xs.map((x) => `${sx(x)},${sy(clampY(polyEval(coefs, x)))}`).join(" ")} />
                {train.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(clampY(p.y))} r={4} fill="var(--c-blue)" stroke="var(--surface)" strokeWidth={1.5} />)}
              </>
            )}
          </Plot>
          <Legend items={[{ label: "true function", color: "var(--faint)", dashed: true }, { label: `degree-${degree} fit`, color: "var(--c-teal)" }, { label: "training points", color: "var(--c-blue)" }]} />
        </div>
        <div>
          <Plot title="Error versus model complexity" x={[1, Math.max(2, curve[curve.length - 1].d)]} y={[logY(Math.min(...curve.map((r) => r.train))) - 0.2, logY(yMax) + 0.2]} xLabel="polynomial degree" yLabel="log₁₀ MSE" height={300}>
            {({ sx, sy }) => (
              <>
                <polyline fill="none" stroke="var(--c-blue)" strokeWidth={2} points={curve.map((r) => `${sx(r.d)},${sy(logY(r.train))}`).join(" ")} />
                <polyline fill="none" stroke="var(--c-orange)" strokeWidth={2} points={curve.map((r) => `${sx(r.d)},${sy(logY(r.val))}`).join(" ")} />
                <line x1={sx(cur.d)} x2={sx(cur.d)} y1={12} y2={260} stroke="var(--text)" strokeDasharray="3 3" />
                <circle cx={sx(best.d)} cy={sy(logY(best.val))} r={6} fill="none" stroke="var(--c-orange)" strokeWidth={2} />
              </>
            )}
          </Plot>
          <Legend items={[{ label: "training error", color: "var(--c-blue)" }, { label: "validation error", color: "var(--c-orange)" }, { label: "current degree", color: "var(--text)", dashed: true }]} />
        </div>
      </div>
    </LabFrame>
  );
}
