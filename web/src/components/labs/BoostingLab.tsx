"use client";
import { useMemo, useState } from "react";
import { fmt, gaussian, mean, rng } from "@/lib/ml";
import { LabFrame, Legend, Plot, Slider, Stat, type TourStep } from "./ui";

type Stump = { thr: number; left: number; right: number };
const f = (x: number) => Math.sin(x) + 0.3 * x;
const MAX_ROUNDS = 200;

// SYNTHETIC 1D regression: y = sin(x) + 0.3x + noise.
export function makeData(n: number, seed: number, noise = 0.35) {
  const r = rng(seed);
  return Array.from({ length: n }, () => { const x = r() * 10; return { x, y: f(x) + gaussian(r) * noise }; }).sort((a, b) => a.x - b.x);
}

/** Best regression stump on (x, residual) by squared error. */
export function fitStump(xs: number[], rs: number[]): Stump {
  let best = { thr: xs[0], left: 0, right: mean(rs), sse: Infinity };
  for (let k = 1; k < xs.length; k++) {
    const L = rs.slice(0, k), R = rs.slice(k);
    const ml = mean(L), mr = mean(R);
    const sse = L.reduce((s, v) => s + (v - ml) ** 2, 0) + R.reduce((s, v) => s + (v - mr) ** 2, 0);
    if (sse < best.sse) best = { thr: (xs[k - 1] + xs[k]) / 2, left: ml, right: mr, sse };
  }
  return { thr: best.thr, left: best.left, right: best.right };
}
const stumpAt = (s: Stump, x: number) => (x < s.thr ? s.left : s.right);

/** Gradient boosting with stumps for MAX_ROUNDS rounds; train and test MSE after every round. */
export function boost(train: { x: number; y: number }[], test: { x: number; y: number }[], lr: number) {
  const xs = train.map((p) => p.x), ys = train.map((p) => p.y);
  const f0 = mean(ys);
  let F = xs.map(() => f0);
  const stumps: Stump[] = [], trainMse = [mean(ys.map((y) => (y - f0) ** 2))];
  const testF = test.map(() => f0), testMse = [mean(test.map((p) => (p.y - f0) ** 2))];
  for (let k = 0; k < MAX_ROUNDS; k++) {
    const res = ys.map((y, i) => y - F[i]);
    const s = fitStump(xs, res);
    stumps.push(s);
    F = F.map((v, i) => v + lr * stumpAt(s, xs[i]));
    test.forEach((p, i) => (testF[i] += lr * stumpAt(s, p.x)));
    trainMse.push(mean(ys.map((y, i) => (y - F[i]) ** 2)));
    testMse.push(mean(test.map((p, i) => (p.y - testF[i]) ** 2)));
  }
  return { f0, stumps, trainMse, testMse };
}

export default function BoostingLab() {
  const [noise, setNoise] = useState(0.35);
  const train = useMemo(() => makeData(40, 3, noise), [noise]);
  const test = useMemo(() => makeData(200, 77, noise), [noise]);
  const [lr, setLr] = useState(0.3);
  const [m, setM] = useState(5);

  const run = useMemo(() => boost(train, test, lr), [train, test, lr]);

  const predict = (x: number, upto: number) => run.f0 + run.stumps.slice(0, upto).reduce((s, st) => s + lr * stumpAt(st, x), 0);
  const residuals = train.map((p) => p.y - predict(p.x, m));
  const next = run.stumps[m];
  const bestRound = run.testMse.indexOf(Math.min(...run.testMse));
  const grid = Array.from({ length: 200 }, (_, i) => (i / 199) * 10);
  const msMax = Math.max(...run.testMse.slice(0, 1), ...run.trainMse.slice(0, 1));

  // Guided tour (Watch mode + explainers). Each step sets η and the round, then advances rounds.
  const setup = (eta: number, rounds: number, sd = 0.35) => { setLr(eta); setM(rounds); setNoise(sd); };
  const roundsTo = (a: number, b: number) => (t: number) => setM(Math.round(a + t * (b - a)));
  const tour: TourStep[] = [
    { id: "start", caption: "Round 0: the teal model is just the mean of y, a flat line at 1.92. The gray bars are residuals, what the model still gets wrong.", apply: () => setup(0.3, 0) },
    { id: "stump", caption: "Tree 1 is a one-split stump fitted to those residuals: below x = 6.04 push down, above push up. The purple dashed line adds 0.3 of it.", apply: () => setup(0.3, 0) },
    { id: "rounds", caption: "Repeat: fit a stump to what is left, add a fraction. Over 30 rounds the teal curve bends into the wave and training error falls from 1.46 to 0.10.", apply: () => setup(0.3, 0), animate: roundsTo(0, 30), animMs: 3200 },
    { id: "small-steps", caption: "With η = 0.05 each tree corrects only a little. After 100 rounds the fit is still catching up, but it moves smoothly.", apply: () => setup(0.05, 0), animate: roundsTo(0, 100), animMs: 3200 },
    { id: "big-steps", caption: "With η = 1 each tree takes the full correction. Training error drops to 0.28 after a single round, and the fit turns jumpy.", apply: () => setup(1, 0), animate: roundsTo(0, 20), animMs: 2600 },
    { id: "too-many", caption: "Now noisier data (σ = 0.6) and η = 1. Test error bottoms out by round 8, then climbs about 18% while training error keeps falling: later trees fit noise. Early stopping keeps the ringed round.", apply: () => setup(1, 0, 0.6), animate: roundsTo(0, MAX_ROUNDS), animMs: 3200 },
  ];

  return (
    <LabFrame
      id="boosting"
      tour={tour}
      title="Gradient boosting, round by round"
      subtitle="Synthetic 1D regression (y = sin x + 0.3x + noise). Each round fits a one-split tree (a stump) to the current residuals and adds a fraction of it."
      onReset={() => { setLr(0.3); setM(5); setNoise(0.35); }}
      presets={[
        { label: "Round 0", apply: () => setM(0) },
        { label: "Big steps (lr 1.0)", apply: () => setLr(1) },
        { label: "Small steps (lr 0.05)", apply: () => { setLr(0.05); setM(60); } },
        { label: "Too many rounds (noisy data)", apply: () => { setNoise(0.6); setLr(1); setM(MAX_ROUNDS); } },
      ]}
      controls={
        <>
          <Slider label="rounds m (trees added)" value={m} min={0} max={MAX_ROUNDS} onChange={setM} />
          <Slider label="learning rate η" value={lr} min={0.02} max={1} step={0.01} onChange={setLr} format={(v) => fmt(v, 2)} />
          <Slider label="label noise σ" value={noise} min={0.1} max={0.8} step={0.05} onChange={setNoise} format={(v) => fmt(v, 2)} />
        </>
      }
      readout={
        <>
          <Stat label="F₀ (mean of y)" value={fmt(run.f0, 3)} />
          <Stat label="train MSE" value={fmt(run.trainMse[m], 4)} color="var(--c-blue)" />
          <Stat label="test MSE" value={fmt(run.testMse[m], 4)} color="var(--c-orange)" />
          <Stat label="best test round" value={`${bestRound} (${fmt(run.testMse[bestRound], 4)})`} />
          {next && <Stat label={`tree ${m + 1}`} value={`x<${fmt(next.thr, 2)}: ${fmt(next.left, 2)} / ${fmt(next.right, 2)}`} />}
        </>
      }
      interpretation={
        <>
          {next && <p className="text-ink">Next update: <span className="font-mono">F<sub>{m + 1}</sub>(x) = F<sub>{m}</sub>(x) + {fmt(lr, 2)} × ({fmt(next.left, 2)} if x &lt; {fmt(next.thr, 2)} else {fmt(next.right, 2)})</span>. The stump (purple) is fitted to the residuals (gray bars), not to y.</p>}
          <p className="mt-1">{m > bestRound + 20
            ? `Training error keeps falling but test error has risen since round ${bestRound}: extra trees now fit noise. Early stopping on a validation set would stop near round ${bestRound}.`
            : lr >= 0.8 ? "With a large learning rate each tree overcorrects; error drops fast and the fit gets jumpy. Lower η needs more rounds but generalizes more smoothly."
              : "Each round removes part of what's left. With a small learning rate, many rounds make small, careful corrections."}</p>
        </>
      }
    >
      <div className="grid gap-4 @3xl:grid-cols-2">
        <div>
          <Plot title="Data, ensemble prediction, and the next tree" x={[0, 10]} y={[-2, 5]} xLabel="x" yLabel="y" height={300}>
            {({ sx, sy }) => (
              <>
                {train.map((p, i) => <line key={`r${i}`} x1={sx(p.x)} x2={sx(p.x)} y1={sy(p.y)} y2={sy(p.y - residuals[i])} stroke="var(--faint)" strokeWidth={1.5} opacity={0.6} />)}
                <polyline fill="none" stroke="var(--c-teal)" strokeWidth={2.5} points={grid.map((x) => `${sx(x)},${sy(predict(x, m))}`).join(" ")} />
                {next && <polyline fill="none" stroke="var(--c-purple)" strokeWidth={2} strokeDasharray="5 3"
                  points={grid.map((x) => `${sx(x)},${sy(predict(x, m) + lr * stumpAt(next, x))}`).join(" ")} />}
                {train.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3.5} fill="var(--c-blue)" stroke="var(--surface)" />)}
              </>
            )}
          </Plot>
          <Legend items={[{ label: `F${m}(x)`, color: "var(--c-teal)" }, { label: "after adding the next tree", color: "var(--c-purple)", dashed: true }, { label: "residuals", color: "var(--faint)" }]} />
        </div>
        <div>
          <Plot title="Error versus number of trees" x={[0, MAX_ROUNDS]} y={[0, msMax * 1.05]} xLabel="rounds" yLabel="MSE" height={300}>
            {({ sx, sy }) => (
              <>
                <polyline fill="none" stroke="var(--c-blue)" strokeWidth={2} points={run.trainMse.map((v, i) => `${sx(i)},${sy(v)}`).join(" ")} />
                <polyline fill="none" stroke="var(--c-orange)" strokeWidth={2} points={run.testMse.map((v, i) => `${sx(i)},${sy(v)}`).join(" ")} />
                <line x1={sx(m)} x2={sx(m)} y1={12} y2={260} stroke="var(--text)" strokeDasharray="3 3" />
                <circle cx={sx(bestRound)} cy={sy(run.testMse[bestRound])} r={5} fill="none" stroke="var(--c-orange)" strokeWidth={2} />
              </>
            )}
          </Plot>
          <Legend items={[{ label: "train", color: "var(--c-blue)" }, { label: "test", color: "var(--c-orange)" }, { label: "current round", color: "var(--text)", dashed: true }]} />
        </div>
      </div>
    </LabFrame>
  );
}
