"use client";
import { useMemo, useRef, useState } from "react";
import { fmt, gaussian, linearFit, linearGrad, mae, mean, regularizedFit, rng, sum } from "@/lib/ml";
import { Button, LabFrame, Legend, linear, Plot, Segmented, Slider, Stat, svgPoint, Tex, type TourStep } from "./ui";

type Pt = { x: number; y: number };
type Penalty = "none" | "l2" | "l1";

// SYNTHETIC data: y = 2x + 3 + noise, fixed seed so every visitor sees the same points.
function makeData(): Pt[] {
  const r = rng(11);
  return Array.from({ length: 10 }, (_, i) => {
    const x = 0.6 + i * 0.95 + r() * 0.4;
    return { x: +x.toFixed(2), y: +(2 * x + 3 + gaussian(r) * 1.8).toFixed(2) };
  });
}

// One colour per penalty, used for its fit line, its readout, and its weight path.
const PEN_COLOR = { l2: "var(--c-purple)", l1: "var(--c-green)" } as const;

const X_DOM: [number, number] = [0, 11];
const Y_DOM: [number, number] = [-2, 30];

export default function LinearRegressionLab() {
  const initial = useMemo(() => makeData(), []);
  const [pts, setPts] = useState<Pt[]>(initial);
  const [w, setW] = useState(1);
  const [b, setB] = useState(4);
  const [penalty, setPenalty] = useState<Penalty>("none");
  const [lambda, setLambda] = useState(2);
  const [lr, setLr] = useState(0.01);
  const [drag, setDrag] = useState<number | null>(null);
  const [changed, setChanged] = useState("Drag a point, move the sliders, or press “Gradient step”.");
  const svgRef = useRef<SVGSVGElement>(null);

  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const g = linearGrad(xs, ys, w, b);
  const maeV = mae(ys, g.yHat);
  const pen = penalty === "l2" ? lambda * w * w : penalty === "l1" ? lambda * Math.abs(w) : 0;
  const ols = linearFit(xs, ys);
  const reg = regularizedFit(xs, ys, lambda, penalty);

  // Weight path: how the optimal slope changes as λ grows, for both penalties.
  const n = xs.length, mx = mean(xs), my = mean(ys);
  const sxy = sum(xs.map((v, i) => (v - mx) * (ys[i] - my))) / n;
  const lamMax = Math.max(4, Math.abs(sxy) * 2.4);
  const path = Array.from({ length: 61 }, (_, i) => {
    const l = (i / 60) * lamMax;
    return { l, l2: regularizedFit(xs, ys, l, "l2").w, l1: regularizedFit(xs, ys, l, "l1").w };
  });

  const step = () => {
    // gradient of MSE + penalty (bias is not penalized)
    const dPen = penalty === "l2" ? 2 * lambda * w : penalty === "l1" ? lambda * Math.sign(w) : 0;
    const nw = w - lr * (g.dw + dPen), nb = b - lr * g.db;
    setW(+nw.toFixed(4));
    setB(+nb.toFixed(4));
    setChanged(`One step: ∂/∂w = ${fmt(g.dw + dPen)}, ∂/∂b = ${fmt(g.db)}. w moved ${fmt(nw - w)}, b moved ${fmt(nb - b)}.`);
  };

  const onMove = (e: React.PointerEvent) => {
    if (drag === null || !svgRef.current) return;
    const p = svgPoint(e, svgRef.current);
    // Same scales Plot builds with its default size and margins.
    const nx = Math.min(X_DOM[1], Math.max(X_DOM[0], linear(X_DOM, [48, 548]).invert(p.x)));
    const ny = Math.min(Y_DOM[1], Math.max(Y_DOM[0], linear(Y_DOM, [320, 12]).invert(p.y)));
    setPts((cur) => cur.map((q, i) => (i === drag ? { x: +nx.toFixed(2), y: +ny.toFixed(2) } : q)));
    setChanged(`Point ${drag + 1} moved. Its residual is now ${fmt(ny - (w * nx + b), 2)}; squared, that contributes ${fmt((ny - (w * nx + b)) ** 2 / n, 2)} to MSE.`);
  };

  const reset = () => {
    setPts(initial); setW(1); setB(4); setPenalty("none"); setLambda(2); setLr(0.01);
    setChanged("Reset to the starting line and data.");
  };

  const bestFit = () => {
    const f = penalty === "none" ? ols : reg;
    setW(+f.w.toFixed(4)); setB(+f.b.toFixed(4));
    setChanged(penalty === "none"
      ? `Least squares: w = ${fmt(f.w)}, b = ${fmt(f.b)}. No other line has lower MSE on these points.`
      : `Best fit with ${penalty.toUpperCase()} penalty λ = ${lambda}: w = ${fmt(f.w)} (unpenalized: ${fmt(ols.w)}). The penalty traded a little MSE for a smaller weight.`);
  };

  // Guided tour (Watch mode + explainers). Each step resets the data and line, then moves one thing.
  const setup = (nw: number, nb: number, pen: Penalty = "none", lam = 2, data = initial) => {
    setPts(data); setW(nw); setB(nb); setPenalty(pen); setLambda(lam); setLr(0.01); setChanged("");
  };
  const fitOf = (p: Pt[]) => linearFit(p.map((q) => q.x), p.map((q) => q.y));
  const best0 = fitOf(initial);
  const ix = initial.map((p) => p.x), iy = initial.map((p) => p.y);
  // n gradient steps from (1, 4) with η = 0.01, recomputed from scratch so the step is absolute
  const gdAt = (k: number) => { let cw = 1, cb = 4; for (let i = 0; i < k; i++) { const gg = linearGrad(ix, iy, cw, cb); cw -= 0.01 * gg.dw; cb -= 0.01 * gg.db; } return { w: +cw.toFixed(4), b: +cb.toFixed(4) }; };
  const gdTo = (k: number) => { const p = gdAt(k); setW(p.w); setB(p.b); };
  const GD_STEPS = 80;
  const gdEnd = gdAt(GD_STEPS);
  const mseAt = (p: { w: number; b: number }) => linearGrad(ix, iy, p.w, p.b).mse;
  const last = initial[initial.length - 1];
  const withOutlier = (y: number) => [...initial.slice(0, -1), { x: last.x, y }];
  const outFit = fitOf(withOutlier(2));
  const l1Zero = 2 * Math.abs(sum(ix.map((v, i) => (v - mean(ix)) * (iy[i] - mean(iy)))) / ix.length);
  const lamSweep = (t: number) => setLambda(+(t * lamMax).toFixed(1));
  const tour: TourStep[] = [
    { id: "residuals", caption: "Ten noisy points and a guessed teal line. Each orange segment is a residual: how far the line misses that point. MSE averages their squares.", apply: () => setup(1, 4) },
    { id: "slope", caption: "Tilt the line by raising the slope. The residuals shrink, then grow again on the other side; MSE has a single lowest point.", apply: () => setup(0.5, 4), animate: (t) => setW(+(0.5 + t * 3).toFixed(2)), animMs: 3000 },
    { id: "descend", caption: `Gradient descent does the tilting for us: each step nudges w and b against the gradient. In ${GD_STEPS} steps MSE falls from ${fmt(mseAt({ w: 1, b: 4 }), 1)} to ${fmt(mseAt(gdEnd), 2)}.`, apply: () => setup(1, 4), animate: (t) => gdTo(Math.round(t * GD_STEPS)), animMs: 3000 },
    { id: "best-fit", caption: `Gradient descent was still creeping (the bias moves slowly). Least squares solves for the best line in one calculation: w = ${fmt(best0.w, 2)}, b = ${fmt(best0.b, 2)}, MSE ${fmt(mseAt(best0), 2)}, both gradients zero.`, apply: () => setup(+best0.w.toFixed(4), +best0.b.toFixed(4)) },
    { id: "outlier", caption: `Drag the last point down to y = 2. Because errors are squared, that one point drags the whole best-fit line toward it: the slope falls from ${fmt(best0.w, 2)} to ${fmt(outFit.w, 2)}.`, apply: () => setup(+best0.w.toFixed(4), +best0.b.toFixed(4)), animate: (t) => { const d = withOutlier(last.y - t * (last.y - 2)), f = fitOf(d); setPts(d); setW(+f.w.toFixed(4)); setB(+f.b.toFixed(4)); }, animMs: 2800 },
    { id: "ridge", caption: "L2 ridge adds λw² to the loss. As λ grows, the dashed purple fit flattens, and in the chart below the purple path of its slope shrinks toward zero but never reaches it.", apply: () => setup(+best0.w.toFixed(4), +best0.b.toFixed(4), "l2", 0), animate: lamSweep, animMs: 3000 },
    { id: "lasso", caption: `L1 lasso adds λ|w| instead. Its green path below drops in a straight line and hits exactly zero once λ passes ${fmt(l1Zero, 1)}: the dashed green fit goes flat and the feature is switched off.`, apply: () => setup(+best0.w.toFixed(4), +best0.b.toFixed(4), "l1", 0), animate: lamSweep, animMs: 3000 },
  ];

  const interpretation = (() => {
    if (penalty !== "none") {
      const name = penalty === "l2" ? "Ridge" : "Lasso";
      if (penalty === "l1" && reg.w === 0) return `${name} at λ = ${fmt(lambda, 1)}: the penalized best slope is exactly 0, so this feature is dropped. The dashed fit is a flat line at the mean of y.`;
      return `${name} at λ = ${fmt(lambda, 1)}: the penalized best slope is ${fmt(reg.w, 2)}, versus ${fmt(ols.w, 2)} with no penalty. The penalty trades a little MSE for a smaller weight.`;
    }
    const gap = g.mse - linearGrad(xs, ys, ols.w, ols.b).mse;
    if (gap < 1e-3) return "You are at the least-squares optimum: both gradients are zero, so any change to w or b raises MSE.";
    const dir = g.dw < 0 ? "increase" : "decrease";
    return `MSE is ${fmt(gap, 2)} above the best possible line. The slope gradient is ${fmt(g.dw, 2)}, so gradient descent will ${dir} w. MAE (${fmt(maeV, 2)}) grows linearly with error, so it reacts less than MSE when you drag a point far away.`;
  })();

  return (
    <LabFrame
      id="linear-regression"
      tour={tour}
      title="Linear regression lab"
      subtitle="Synthetic data: y = 2x + 3 + noise. Drag any point."
      onReset={reset}
      presets={[
        { label: "Add outlier", apply: () => { setPts((p) => [...p.slice(0, 9), { x: p[p.length - 1].x, y: 2 }]); setChanged("Point 10 is now an outlier. Press Best fit and watch the line tilt toward it."); } },
        { label: "Best fit", apply: bestFit },
        { label: "Flat line", apply: () => { setW(0); setB(+mean(ys).toFixed(2)); setChanged("A flat line at the mean of y is the baseline that R² compares against."); } },
      ]}
      controls={
        <>
          <Slider label="slope w" value={w} min={-2} max={5} step={0.01} onChange={(v) => { setW(v); setChanged(`Slope set to ${fmt(v, 2)}.`); }} format={(v) => fmt(v, 2)} />
          <Slider label="bias b" value={b} min={-5} max={15} step={0.01} onChange={(v) => { setB(v); setChanged(`Bias set to ${fmt(v, 2)}.`); }} format={(v) => fmt(v, 2)} />
          <Segmented label="Penalty on w" value={penalty} onChange={(v) => { setPenalty(v); setChanged(v === "none" ? "No penalty: plain MSE." : `${v === "l2" ? "Ridge (L2)" : "Lasso (L1)"} adds ${v === "l2" ? "λw²" : "λ|w|"} to the loss.`); }}
            options={[{ value: "none", label: "None" }, { value: "l2", label: "L2 ridge" }, { value: "l1", label: "L1 lasso" }]} />
          {penalty !== "none" && <Slider label="λ (strength)" value={lambda} min={0} max={+lamMax.toFixed(1)} step={0.1} onChange={setLambda} format={(v) => fmt(v, 1)} />}
          <Slider label="learning rate η" value={lr} min={0.001} max={0.05} step={0.001} onChange={setLr} format={(v) => fmt(v, 3)} hint="Above about 0.04 the steps overshoot and diverge." />
          <div className="flex gap-2"><Button primary onClick={step}>Gradient step</Button><Button onClick={bestFit}>Best fit</Button></div>
        </>
      }
      readout={
        <>
          <Stat label="ŷ =" value={`${fmt(w, 2)}x + ${fmt(b, 2)}`} />
          <Stat label="MSE" value={fmt(g.mse, 3)} color="var(--c-orange)" />
          <Stat label="MAE" value={fmt(maeV, 3)} />
          {penalty !== "none" && <Stat label={penalty === "l2" ? "λw²" : "λ|w|"} value={fmt(pen, 3)} color={PEN_COLOR[penalty]} />}
          {penalty !== "none" && <Stat label="loss" value={fmt(g.mse + pen, 3)} />}
          <Stat label="∂MSE/∂w" value={fmt(g.dw, 3)} />
          <Stat label="∂MSE/∂b" value={fmt(g.db, 3)} />
        </>
      }
      interpretation={<>{changed && <p className="text-ink mb-1">{changed}</p>}<p>{interpretation}</p></>}
    >
      <Plot
        title="Scatterplot with prediction line and residuals"
        x={X_DOM} y={Y_DOM} xLabel="x (feature)" yLabel="y (target)"
        svgProps={{ ref: svgRef, onPointerMove: onMove, onPointerUp: () => setDrag(null), onPointerLeave: () => setDrag(null) }}
      >
        {({ sx, sy }) => (
          <>
            {penalty !== "none" && (
              <line x1={sx(X_DOM[0])} x2={sx(X_DOM[1])} y1={sy(reg.w * X_DOM[0] + reg.b)} y2={sy(reg.w * X_DOM[1] + reg.b)} stroke={PEN_COLOR[penalty]} strokeWidth={2} strokeDasharray="6 4" />
            )}
            <line x1={sx(X_DOM[0])} x2={sx(X_DOM[1])} y1={sy(ols.w * X_DOM[0] + ols.b)} y2={sy(ols.w * X_DOM[1] + ols.b)} stroke="var(--faint)" strokeWidth={1.5} strokeDasharray="3 4" />
            {pts.map((p, i) => (
              <line key={`r${i}`} x1={sx(p.x)} x2={sx(p.x)} y1={sy(p.y)} y2={sy(w * p.x + b)} stroke="var(--c-orange)" strokeWidth={1.5} opacity={0.8} />
            ))}
            <line x1={sx(X_DOM[0])} x2={sx(X_DOM[1])} y1={sy(w * X_DOM[0] + b)} y2={sy(w * X_DOM[1] + b)} stroke="var(--c-teal)" strokeWidth={2.5} />
            {pts.map((p, i) => (
              <circle
                key={i}
                cx={sx(p.x)} cy={sy(p.y)} r={drag === i ? 8 : 6.5}
                fill="var(--c-blue)" stroke="var(--surface)" strokeWidth={2}
                className="cursor-grab"
                tabIndex={0}
                aria-label={`Point ${i + 1}: x ${p.x}, y ${p.y}. Arrow keys move it.`}
                onPointerDown={(e) => { (e.currentTarget.ownerSVGElement as SVGSVGElement).setPointerCapture(e.pointerId); setDrag(i); }}
                onKeyDown={(e) => {
                  const d = { ArrowUp: [0, 0.5], ArrowDown: [0, -0.5], ArrowLeft: [-0.2, 0], ArrowRight: [0.2, 0] }[e.key];
                  if (!d) return;
                  e.preventDefault();
                  setPts((cur) => cur.map((q, j) => (j === i ? { x: +(q.x + d[0]).toFixed(2), y: +(q.y + d[1]).toFixed(2) } : q)));
                }}
              />
            ))}
          </>
        )}
      </Plot>
      <Legend items={[
        { label: "your line", color: "var(--c-teal)" },
        { label: "least-squares fit", color: "var(--faint)", dashed: true },
        ...(penalty !== "none" ? [{ label: `${penalty.toUpperCase()} fit`, color: PEN_COLOR[penalty], dashed: true }] : []),
        { label: "residuals", color: "var(--c-orange)" },
      ]} />

      {penalty !== "none" && (
        <div className="mt-6">
          <p className="text-sm font-medium mb-1">How the penalty shrinks the optimal weight</p>
          <p className="text-xs text-muted mb-2">
            L2 shrinks <Tex>{"w"}</Tex> smoothly toward zero but never reaches it. L1 shrinks it linearly and sets it to exactly zero once <Tex>{"\\lambda \\ge 2|s_{xy}|"}</Tex> (here {fmt(2 * Math.abs(sxy), 2)}). That is why lasso performs feature selection.
          </p>
          <Plot title="Optimal slope as a function of regularization strength" width={560} height={200} x={[0, lamMax]} y={[Math.min(0, ols.w) - 0.2, Math.max(0, ols.w) + 0.3]} xLabel="λ" yLabel="optimal w" margin={{ t: 10, r: 12, b: 36, l: 48 }}>
            {({ sx, sy }) => (
              <>
                <polyline fill="none" stroke={PEN_COLOR.l2} strokeWidth={2} points={path.map((p) => `${sx(p.l)},${sy(p.l2)}`).join(" ")} />
                <polyline fill="none" stroke={PEN_COLOR.l1} strokeWidth={2} points={path.map((p) => `${sx(p.l)},${sy(p.l1)}`).join(" ")} />
                <line x1={sx(lambda)} x2={sx(lambda)} y1={10} y2={164} stroke="var(--c-orange)" strokeDasharray="4 3" />
              </>
            )}
          </Plot>
          <Legend items={[{ label: "L2 (ridge)", color: PEN_COLOR.l2 }, { label: "L1 (lasso)", color: PEN_COLOR.l1 }, { label: "current λ", color: "var(--c-orange)", dashed: true }]} />
        </div>
      )}
    </LabFrame>
  );
}
