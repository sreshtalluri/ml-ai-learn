"use client";
import { useMemo, useRef, useState } from "react";
import { fmt, gaussian, knnNeighbors, knnPredict, mean, rng, variance, type LabeledPoint } from "@/lib/ml";
import { LabFrame, Legend, linear, Plot, Segmented, Slider, Stat, svgPoint, Toggle, type TourStep } from "./ui";

// SYNTHETIC customers: x = age (years), y = income (thousand $/yr). Label 1 = bought the product.
// Purchase depends mostly on age, so income (the bigger-range feature) is mostly noise.
function makeData(): LabeledPoint[] {
  const r = rng(5);
  return Array.from({ length: 60 }, () => {
    const age = 20 + r() * 50;
    const income = 30 + r() * 150;
    const p = 1 / (1 + Math.exp(-(age - 45) / 4 + gaussian(r) * 0.6));
    return { x: +age.toFixed(1), y: +income.toFixed(1), label: r() < p ? 1 : 0 };
  });
}

const X: [number, number] = [18, 72];
const Y: [number, number] = [20, 190];
const COLORS = ["var(--c-blue)", "var(--c-orange)"];
const NAMES = ["did not buy", "bought"];
type Metric = "euclidean" | "manhattan";

export default function KnnLab() {
  const data = useMemo(() => makeData(), []);
  const [q, setQ] = useState({ x: 47, y: 100 });
  const [k, setK] = useState(5);
  const [metric, setMetric] = useState<Metric>("euclidean");
  const [scaled, setScaled] = useState(false);
  const [regions, setRegions] = useState(true);
  const [dragging, setDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  // Standardization statistics come from the (training) data only.
  const stats = useMemo(() => {
    const xs = data.map((p) => p.x), ys = data.map((p) => p.y);
    return { mx: mean(xs), sx: Math.sqrt(variance(xs)), my: mean(ys), sy: Math.sqrt(variance(ys)) };
  }, [data]);
  const tf = (p: { x: number; y: number }) => (scaled ? { x: (p.x - stats.mx) / stats.sx, y: (p.y - stats.my) / stats.sy } : p);
  const space = useMemo(() => data.map((p) => ({ ...p, ...tf(p) })), [data, scaled]); // eslint-disable-line react-hooks/exhaustive-deps

  const neighbors = knnNeighbors(space, tf(q), k, metric);
  const votes = [0, 1].map((c) => neighbors.filter((n) => n.point.label === c).length);
  const pred = knnPredict(space, tf(q), k, metric);
  const radius = neighbors[neighbors.length - 1].distance;

  // Leave-one-out accuracy: how well does this K / metric / scaling setting generalize?
  const loo = useMemo(() => {
    let correct = 0;
    space.forEach((p, i) => {
      const rest = space.filter((_, j) => j !== i);
      if (knnPredict(rest, p, k, metric) === p.label) correct++;
    });
    return correct / space.length;
  }, [space, k, metric]);

  // Background decision regions on a coarse grid.
  const grid = useMemo(() => {
    if (!regions) return [];
    const cells: { x: number; y: number; c: number }[] = [];
    for (let i = 0; i < 36; i++) for (let j = 0; j < 24; j++) {
      const x = X[0] + ((i + 0.5) / 36) * (X[1] - X[0]), y = Y[0] + ((j + 0.5) / 24) * (Y[1] - Y[0]);
      cells.push({ x, y, c: knnPredict(space, tf({ x, y }), k, metric) });
    }
    return cells;
  }, [regions, space, k, metric]); // eslint-disable-line react-hooks/exhaustive-deps

  const sx = linear(X, [48, 548]), sy = linear(Y, [320, 12]);
  const move = (e: React.PointerEvent) => {
    if (!svgRef.current) return;
    const p = svgPoint(e, svgRef.current);
    setQ({ x: +Math.min(X[1], Math.max(X[0], sx.invert(p.x))).toFixed(1), y: +Math.min(Y[1], Math.max(Y[0], sy.invert(p.y))).toFixed(1) });
  };

  // Neighborhood outline: radius r in the distance space, drawn back in raw-unit pixels.
  const pxPerX = (548 - 48) / (X[1] - X[0]), pxPerY = (320 - 12) / (Y[1] - Y[0]);
  const rx = radius * (scaled ? stats.sx : 1) * pxPerX;
  const ry = radius * (scaled ? stats.sy : 1) * pxPerY;
  const cx = sx(q.x), cy = sy(q.y);

  const interpretation = scaled
    ? `With standardized features, one standard deviation of age counts as much as one standard deviation of income, so the neighborhood stretches to cover similar ages. Leave-one-out accuracy is ${fmt(loo * 100, 1)}%.`
    : `In raw units, income spans about 150 while age spans about 50, so distance is dominated by income: the neighborhood is a thin horizontal band of similar incomes, regardless of age. Turn on scaling and compare the leave-one-out accuracy (now ${fmt(loo * 100, 1)}%).`;

  // Guided tour (Watch mode + explainers). Each step sets query, K, and scaling, then moves one.
  const setup = (kk: number, sc: boolean, query = { x: 47, y: 100 }) => { setQ(query); setK(kk); setMetric("euclidean"); setScaled(sc); setRegions(true); };
  const tour: TourStep[] = [
    { id: "query", caption: "The ringed dot is a new customer. KNN finds the 5 closest training points, joined by lines inside the purple outline, and predicts by majority vote.", apply: () => setup(5, false) },
    { id: "move", caption: "Slide the customer from age 25 to 65 at the same income. The neighbors change as it moves, and the prediction flips to bought around age 50.", apply: () => setup(5, false, { x: 25, y: 100 }), animate: (t) => setQ({ x: +(25 + t * 40).toFixed(1), y: 100 }), animMs: 3200 },
    { id: "raw-units", caption: "Look at the purple outline: a flat band. Income spans 150 units and age only 50, so distance is mostly income, the feature that doesn't matter here.", apply: () => setup(5, false) },
    { id: "scaled", caption: "Standardize both features and the neighborhood stretches to cover similar ages. This customer flips to bought, and leave-one-out accuracy rises from 76.7% to 80%.", apply: () => setup(5, true) },
    { id: "k1", caption: "With K = 1 each prediction copies a single neighbor, noise included. The shaded regions turn patchy and accuracy falls to 70%.", apply: () => setup(1, true) },
    { id: "k-sweep", caption: "Raise K and the regions smooth out. Accuracy peaks at 86.7% near K = 15, then slips to 78% at K = 29 as the vote averages over too wide an area.", apply: () => setup(1, true), animate: (t) => setK(1 + 2 * Math.round(t * 14)), animMs: 3400 },
  ];

  return (
    <LabFrame
      id="knn"
      tour={tour}
      title="K-nearest neighbors lab"
      subtitle="Synthetic customers: age vs income. Click or drag anywhere to move the query point."
      onReset={() => { setQ({ x: 47, y: 100 }); setK(5); setMetric("euclidean"); setScaled(false); setRegions(true); }}
      presets={[
        { label: "K = 1 (noisy)", apply: () => setK(1) },
        { label: "K = 25 (smooth)", apply: () => setK(25) },
        { label: "Fix scaling", apply: () => setScaled(true) },
      ]}
      controls={
        <>
          <Slider label="K (neighbors)" value={k} min={1} max={29} step={2} onChange={setK} hint="Odd values avoid ties between two classes." />
          <Segmented label="Distance" value={metric} onChange={setMetric} options={[{ value: "euclidean", label: "Euclidean" }, { value: "manhattan", label: "Manhattan" }]} />
          <Toggle label="Standardize features (z-scores)" checked={scaled} onChange={setScaled} />
          <Toggle label="Shade decision regions" checked={regions} onChange={setRegions} />
        </>
      }
      readout={
        <>
          <Stat label="query" value={`age ${q.x}, income ${q.y}`} />
          <Stat label="votes" value={`${votes[0]} : ${votes[1]}`} />
          <Stat label="prediction" value={NAMES[pred]} color={COLORS[pred]} />
          <Stat label="k-th distance" value={fmt(radius, 2)} />
          <Stat label="LOO accuracy" value={`${fmt(loo * 100, 1)}%`} />
        </>
      }
      interpretation={interpretation}
    >
      <Plot
        title="KNN neighborhood around a query point"
        x={X} y={Y} xLabel="age (years)" yLabel="income (thousand $)"
        svgProps={{
          ref: svgRef,
          onPointerDown: (e) => { (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId); setDragging(true); move(e); },
          onPointerMove: (e) => dragging && move(e),
          onPointerUp: () => setDragging(false),
          tabIndex: 0,
          "aria-label": "Plot. Arrow keys move the query point.",
          onKeyDown: (e) => {
            const d = { ArrowUp: [0, 4], ArrowDown: [0, -4], ArrowLeft: [-1.5, 0], ArrowRight: [1.5, 0] }[e.key];
            if (d) { e.preventDefault(); setQ((p) => ({ x: +Math.min(X[1], Math.max(X[0], p.x + d[0])).toFixed(1), y: +Math.min(Y[1], Math.max(Y[0], p.y + d[1])).toFixed(1) })); }
          },
        }}
      >
        {() => (
          <>
            {grid.map((g, i) => (
              <rect key={i} x={sx(g.x) - pxPerX * 0.75 + 0.5} y={sy(g.y) - pxPerY * 3.55} width={pxPerX * 1.5} height={pxPerY * 7.1} fill={COLORS[g.c]} opacity={0.09} />
            ))}
            {metric === "euclidean" ? (
              <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="none" stroke="var(--c-purple)" strokeDasharray="5 4" strokeWidth={1.5} />
            ) : (
              <polygon points={`${cx},${cy - ry} ${cx + rx},${cy} ${cx},${cy + ry} ${cx - rx},${cy}`} fill="none" stroke="var(--c-purple)" strokeDasharray="5 4" strokeWidth={1.5} />
            )}
            {neighbors.map((n) => (
              <line key={`l${n.index}`} x1={cx} y1={cy} x2={sx(data[n.index].x)} y2={sy(data[n.index].y)} stroke={COLORS[n.point.label]} strokeWidth={1.2} opacity={0.7} />
            ))}
            {data.map((p, i) => {
              const isN = neighbors.some((n) => n.index === i);
              return p.label === 1
                ? <rect key={i} x={sx(p.x) - 4.5} y={sy(p.y) - 4.5} width={9} height={9} fill={COLORS[1]} stroke={isN ? "var(--text)" : "var(--surface)"} strokeWidth={isN ? 2 : 1} />
                : <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={5} fill={COLORS[0]} stroke={isN ? "var(--text)" : "var(--surface)"} strokeWidth={isN ? 2 : 1} />;
            })}
            <g transform={`translate(${cx} ${cy})`}>
              <circle r={9} fill="var(--surface)" stroke={COLORS[pred]} strokeWidth={3} />
              <circle r={3} fill={COLORS[pred]} />
            </g>
          </>
        )}
      </Plot>
      <Legend items={[
        { label: "did not buy (circles)", color: COLORS[0] },
        { label: "bought (squares)", color: COLORS[1] },
        { label: "neighborhood (k-th distance)", color: "var(--c-purple)", dashed: true },
      ]} />
    </LabFrame>
  );
}
