"use client";
import { useMemo, useState } from "react";
import { accuracy, confusionAt, f1, fmt, gaussian, precision, recall, rng, sigmoid, specificity } from "@/lib/ml";
import { LabFrame, Legend, Plot, Slider, Stat, Toggle, type TourStep } from "./ui";

// SYNTHETIC fraud-style data: about 30% positives (124 of 400 with this seed). Scores are calibrated by construction:
// each label is drawn with probability equal to its score.
function makeData(prevalence: number) {
  const r = rng(23);
  const scores: number[] = [], labels: number[] = [];
  for (let i = 0; i < 400; i++) {
    const z = gaussian(r) * 1.6 + Math.log(prevalence / (1 - prevalence));
    const s = sigmoid(z);
    scores.push(s);
    labels.push(r() < s ? 1 : 0);
  }
  return { scores, labels };
}

const BINS = 20;

export default function ThresholdLab() {
  const [t, setT] = useState(0.5);
  const [costFP, setCostFP] = useState(1);
  const [costFN, setCostFN] = useState(10);
  const [overconfident, setOverconfident] = useState(false);
  const base = useMemo(() => makeData(0.2), []);
  // An overconfident model pushes scores toward 0 and 1 without changing their ranking.
  const scores = useMemo(() => (overconfident ? base.scores.map((s) => sigmoid(3 * Math.log(s / (1 - s)))) : base.scores), [base, overconfident]);
  const labels = base.labels;

  const c = confusionAt(scores, labels, t);
  const cost = c.fp * costFP + c.fn * costFN;
  const sweep = useMemo(() => Array.from({ length: 101 }, (_, i) => {
    const th = i / 100, cc = confusionAt(scores, labels, th);
    return { th, tpr: recall(cc), fpr: 1 - specificity(cc), cost: cc.fp * costFP + cc.fn * costFN };
  }), [scores, labels, costFP, costFN]);
  const best = sweep.reduce((a, b) => (b.cost < a.cost ? b : a));

  const hist = useMemo(() => {
    const h = Array.from({ length: BINS }, () => [0, 0]);
    scores.forEach((s, i) => h[Math.min(BINS - 1, Math.floor(s * BINS))][labels[i]]++);
    return h;
  }, [scores, labels]);
  const hMax = Math.max(...hist.map(([a, b]) => Math.max(a, b)));

  const calib = useMemo(() => Array.from({ length: 10 }, (_, b) => {
    const idx = scores.map((s, i) => [s, i] as const).filter(([s]) => Math.min(9, Math.floor(s * 10)) === b);
    if (idx.length < 5) return null;
    return { pred: idx.reduce((a, [s]) => a + s, 0) / idx.length, obs: idx.reduce((a, [, i]) => a + labels[i], 0) / idx.length };
  }).filter((x) => x !== null), [scores, labels]);

  const cell = (label: string, v: number, tone: string, hint: string) => (
    <div className={`rounded-lg p-2 text-center ${tone}`}>
      <div className="text-[11px] text-muted">{label}</div>
      <div className="text-xl font-semibold font-mono">{v}</div>
      <div className="text-[10px] text-faint">{hint}</div>
    </div>
  );

  // Guided tour (Watch mode + explainers). Each step sets threshold, costs, and model, then moves one.
  const setup = (th: number, fp = 1, fn = 10, over = false) => { setT(th); setCostFP(fp); setCostFN(fn); setOverconfident(over); };
  const toBest = (t0: number) => (u: number) => setT(+(t0 + u * (best.th - t0)).toFixed(2)); // best is re-read each frame
  const tour: TourStep[] = [
    { id: "scores", caption: "Each bar counts transactions by model score: blue are legitimate, orange are fraud. The vertical line is the threshold; everything to its right gets flagged.", apply: () => setup(0.5) },
    { id: "sweep", caption: "Slide the threshold from high to low. More fraud moves into TP, but false alarms pile up in FP, and the orange dot climbs the ROC curve.", apply: () => setup(0.9), animate: (u) => setT(+(0.9 - u * 0.85).toFixed(2)), animMs: 3200 },
    { id: "accuracy", caption: "At 0.5, accuracy is 77.5% yet the model misses 66 of 124 frauds. Always saying legitimate would already score 69%, so accuracy hides the misses.", apply: () => setup(0.5) },
    { id: "cost", caption: "Now price the mistakes: a missed fraud costs 10, a false alarm 1. The cheapest threshold drops to 0.09, catching 118 of 124 frauds.", apply: () => setup(0.5), animate: toBest(0.5), animMs: 2600 },
    { id: "flip-costs", caption: "Flip the prices, false alarms now cost 10. The cheapest threshold jumps to 0.9 and the model flags only cases it is sure of.", apply: () => setup(0.5, 10, 1), animate: toBest(0.5), animMs: 2600 },
    { id: "calibration", caption: "An overconfident model pushes scores toward 0 and 1 without changing their order. The ROC curve stays put, but the teal calibration curve leaves the diagonal.", apply: () => setup(0.5, 1, 10, true) },
  ];

  return (
    <LabFrame
      id="threshold"
      tour={tour}
      title="Classification threshold lab"
      subtitle="400 synthetic transactions, about 30% fraud. The model outputs a score; the threshold turns it into an action."
      onReset={() => { setT(0.5); setCostFP(1); setCostFN(10); setOverconfident(false); }}
      presets={[
        { label: "Min-cost threshold", apply: () => setT(best.th) },
        { label: "Catch all fraud", apply: () => setT(0.05) },
        { label: "Only sure cases", apply: () => setT(0.85) },
      ]}
      controls={
        <>
          <Slider label="threshold" value={t} min={0} max={1} step={0.01} onChange={setT} format={(v) => fmt(v, 2)} />
          <Slider label="cost of a false positive" value={costFP} min={0} max={20} onChange={setCostFP} hint="e.g. a blocked legitimate purchase" />
          <Slider label="cost of a false negative" value={costFN} min={0} max={50} onChange={setCostFN} hint="e.g. a missed fraud" />
          <Toggle label="Overconfident model (same ranking)" checked={overconfident} onChange={setOverconfident} />
        </>
      }
      readout={
        <>
          <Stat label="accuracy" value={fmt(accuracy(c), 3)} />
          <Stat label="precision" value={fmt(precision(c), 3)} color="var(--c-blue)" />
          <Stat label="recall" value={fmt(recall(c), 3)} color="var(--c-orange)" />
          <Stat label="specificity" value={fmt(specificity(c), 3)} />
          <Stat label="F1" value={fmt(f1(c), 3)} />
          <Stat label="total cost" value={cost} color="var(--c-red)" />
          <Stat label="min-cost t" value={fmt(best.th, 2)} />
        </>
      }
      interpretation={`At threshold ${fmt(t, 2)} the model flags ${c.tp + c.fp} transactions: ${c.tp} real fraud and ${c.fp} false alarms, and misses ${c.fn} fraud cases. With a false negative costing ${costFN} and a false positive ${costFP}, the cheapest threshold is ${fmt(best.th, 2)}. ${costFN > costFP ? "Missed fraud is expensive, so the best threshold sits below 0.5." : "False alarms are expensive here, so the best threshold moves up."} Note that accuracy (${fmt(accuracy(c) * 100, 1)}%) barely reflects any of this: always predicting "not fraud" would score ${Math.round((100 * (c.tn + c.fp)) / (c.tp + c.fp + c.tn + c.fn))}%.`}
    >
      <div className="grid gap-5 @2xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div>
          <Plot title="Score histogram by true class" x={[0, 1]} y={[0, hMax]} height={240} xLabel="model score" yLabel="count" margin={{ t: 10, r: 10, b: 36, l: 40 }}>
            {({ sx, sy }) => (
              <>
                {hist.map(([neg, pos], b) => (
                  <g key={b}>
                    <rect x={sx(b / BINS) + 1} y={sy(neg)} width={sx(1 / BINS) - sx(0) - 2} height={sy(0) - sy(neg)} fill="var(--c-blue)" opacity={0.45} />
                    <rect x={sx(b / BINS) + 3} y={sy(pos)} width={sx(1 / BINS) - sx(0) - 6} height={sy(0) - sy(pos)} fill="var(--c-orange)" opacity={0.85} />
                  </g>
                ))}
                <line x1={sx(t)} x2={sx(t)} y1={10} y2={204} stroke="var(--text)" strokeWidth={2} />
                <text x={sx(t) + 4} y={20} fontSize="11" fill="var(--text)">flag →</text>
              </>
            )}
          </Plot>
          <Legend items={[{ label: "legitimate", color: "var(--c-blue)" }, { label: "fraud", color: "var(--c-orange)" }, { label: "threshold", color: "var(--text)" }]} />
          <div className="mt-4 grid grid-cols-[auto_1fr_1fr] gap-1.5 text-sm items-center">
            <span />
            <span className="text-center text-xs text-muted">predicted fraud</span>
            <span className="text-center text-xs text-muted">predicted legit</span>
            <span className="text-xs text-muted pr-1">actually fraud</span>
            {cell("TP", c.tp, "bg-good/15", "caught")}
            {cell("FN", c.fn, "bg-bad/15", `missed · cost ${c.fn * costFN}`)}
            <span className="text-xs text-muted pr-1">actually legit</span>
            {cell("FP", c.fp, "bg-bad/10", `false alarm · cost ${c.fp * costFP}`)}
            {cell("TN", c.tn, "bg-good/10", "correctly ignored")}
          </div>
        </div>
        <div className="space-y-4">
          <div>
            <Plot title="ROC curve" x={[0, 1]} y={[0, 1]} width={300} height={240} xLabel="false positive rate" yLabel="true positive rate (recall)" margin={{ t: 10, r: 10, b: 36, l: 40 }}>
              {({ sx, sy }) => (
                <>
                  <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="var(--faint)" strokeDasharray="4 4" />
                  <polyline fill="none" stroke="var(--c-purple)" strokeWidth={2} points={sweep.map((p) => `${sx(p.fpr)},${sy(p.tpr)}`).join(" ")} />
                  <circle cx={sx(1 - specificity(c))} cy={sy(recall(c))} r={6} fill="var(--c-orange)" stroke="var(--surface)" strokeWidth={2} />
                </>
              )}
            </Plot>
            <p className="text-xs text-muted">Every threshold is one point on this curve. The dashed diagonal is random guessing.</p>
          </div>
          <div>
            <Plot title="Calibration (reliability) curve" x={[0, 1]} y={[0, 1]} width={300} height={220} xLabel="mean predicted score" yLabel="observed fraud rate" margin={{ t: 10, r: 10, b: 36, l: 40 }}>
              {({ sx, sy }) => (
                <>
                  <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="var(--faint)" strokeDasharray="4 4" />
                  <polyline fill="none" stroke="var(--c-teal)" strokeWidth={2} points={calib.map((p) => `${sx(p!.pred)},${sy(p!.obs)}`).join(" ")} />
                  {calib.map((p, i) => <circle key={i} cx={sx(p!.pred)} cy={sy(p!.obs)} r={3.5} fill="var(--c-teal)" />)}
                </>
              )}
            </Plot>
            <p className="text-xs text-muted">{overconfident ? "Overconfident: high scores happen less often than claimed, and low scores more often. The ROC curve is unchanged because ranking is unchanged." : "Calibrated: a score of 0.3 means about 30% of such cases are fraud. Toggle the overconfident model to see miscalibration."}</p>
          </div>
        </div>
      </div>
    </LabFrame>
  );
}
