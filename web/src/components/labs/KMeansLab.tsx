"use client";
import { useRef, useState } from "react";
import { fmt, gaussian, kmeansAssign, kmeansInertia, kmeansUpdate, rng, type Point2 } from "@/lib/ml";
import { Button, LabFrame, Legend, linear, Plot, Segmented, Slider, Stat, svgPoint, type TourStep } from "./ui";

type Dataset = "blobs" | "stretched" | "uneven" | "rings";
type Init = "random" | "plusplus" | "manual";

// All datasets SYNTHETIC with fixed seeds.
function makeData(kind: Dataset): Point2[] {
  const r = rng(kind.length * 97 + 3);
  const blob = (cx: number, cy: number, sx: number, sy: number, n: number) =>
    Array.from({ length: n }, () => ({ x: cx + gaussian(r) * sx, y: cy + gaussian(r) * sy }));
  if (kind === "blobs") return [...blob(2.5, 7, 0.7, 0.7, 30), ...blob(7.5, 7.5, 0.8, 0.8, 30), ...blob(5, 2.5, 0.8, 0.7, 30)];
  if (kind === "stretched") return [...blob(5, 7.3, 3.2, 0.35, 45), ...blob(5, 4.2, 3.2, 0.35, 45)];
  if (kind === "uneven") return [...blob(3.2, 5, 1.6, 1.6, 90), ...blob(8.2, 8.2, 0.35, 0.35, 12)];
  return [
    ...Array.from({ length: 70 }, () => { const a = r() * 2 * Math.PI, d = 3.6 + gaussian(r) * 0.2; return { x: 5 + d * Math.cos(a), y: 5 + d * Math.sin(a) }; }),
    ...blob(5, 5, 0.6, 0.6, 30),
  ];
}

function initCentroids(points: Point2[], k: number, how: Init, seed: number): Point2[] {
  const r = rng(seed);
  if (how === "random") return Array.from({ length: k }, () => ({ ...points[Math.floor(r() * points.length)] }));
  // k-means++: first centroid uniform, then sample proportional to squared distance to the nearest chosen centroid.
  const cs = [{ ...points[Math.floor(r() * points.length)] }];
  while (cs.length < k) {
    const d2 = points.map((p) => Math.min(...cs.map((c) => (p.x - c.x) ** 2 + (p.y - c.y) ** 2)));
    let t = r() * d2.reduce((a, b) => a + b, 0);
    const i = d2.findIndex((d) => (t -= d) <= 0);
    cs.push({ ...points[i < 0 ? points.length - 1 : i] });
  }
  return cs;
}

/** Plays n half-steps (assign, update, assign, ...) from a fresh start, like the tour does. */
function simulate(kind: Dataset, k: number, how: Init, seed: number, n: number) {
  const pts = makeData(kind);
  let c = initCentroids(pts, k, how, seed), a: number[] = [], updates = 0;
  const inertia: number[] = [];
  for (let i = 0; i < n; i++) {
    if (i % 2 === 0) a = kmeansAssign(pts, c); else { c = kmeansUpdate(pts, a, c); updates++; }
    inertia.push(kmeansInertia(pts, a, c));
  }
  return { n: pts.length, inertia, last: inertia[inertia.length - 1], updates };
}
const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;

const COLORS = ["var(--c-blue)", "var(--c-orange)", "var(--c-teal)", "var(--c-purple)", "var(--c-red)", "var(--c-green)"];
const D: [number, number] = [0, 10];

export default function KMeansLab() {
  const [dataset, setDataset] = useState<Dataset>("blobs");
  const [points, setPoints] = useState<Point2[]>(() => makeData("blobs"));
  const [k, setK] = useState(3);
  const [init, setInit] = useState<Init>("random");
  const [seed, setSeed] = useState(4);
  const [centroids, setCentroids] = useState<Point2[]>(() => initCentroids(makeData("blobs"), 3, "random", 4));
  const [assign, setAssign] = useState<number[] | null>(null);
  const [phase, setPhase] = useState<"assign" | "update">("assign");
  const [iter, setIter] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const [note, setNote] = useState("Centroids are placed. Press “Assign” to give each point to its nearest centroid.");
  const svgRef = useRef<SVGSVGElement>(null);

  const restart = (pts = points, kk = k, how = init, s = seed) => {
    setCentroids(how === "manual" ? [] : initCentroids(pts, kk, how, s));
    setAssign(null); setPhase("assign"); setIter(0); setHistory([]);
    setNote(how === "manual" ? `Click the plot to place ${kk} centroids.` : "New centroids placed. Press “Assign”.");
  };

  const ready = centroids.length === k;
  const inertia = assign ? kmeansInertia(points, assign, centroids) : null;

  const doAssign = () => {
    const a = kmeansAssign(points, centroids);
    const changed = assign ? a.filter((v, i) => v !== assign[i]).length : a.length;
    setAssign(a);
    setHistory((h) => [...h, kmeansInertia(points, a, centroids)]);
    setPhase("update");
    setNote(!assign
      ? "Assignment step: every point now belongs to its nearest centroid."
      : changed === 0
        ? `No point changed cluster: K-means has converged after ${plural(iter, "update")}. Inertia ${fmt(kmeansInertia(points, a, centroids), 2)}.`
        : `Assignment step: ${plural(changed, "point")} switched to a different centroid. Each point now belongs to the nearest one.`);
    return { a, changed };
  };

  const doUpdate = () => {
    if (!assign) return;
    const c = kmeansUpdate(points, assign, centroids);
    const moved = Math.max(...c.map((p, j) => Math.hypot(p.x - centroids[j].x, p.y - centroids[j].y)));
    setCentroids(c);
    setHistory((h) => [...h, kmeansInertia(points, assign, c)]);
    setIter((i) => i + 1);
    setPhase("assign");
    setNote(`Update step: each centroid moved to the mean of its points (largest move ${fmt(moved, 2)}). Inertia can only go down or stay the same.`);
  };

  const runAll = () => {
    let c = centroids, a = assign ?? kmeansAssign(points, c), it = iter;
    const h = [...history, kmeansInertia(points, a, c)];
    for (let s = 0; s < 100; s++) {
      c = kmeansUpdate(points, a, c);
      h.push(kmeansInertia(points, a, c));
      it++;
      const a2 = kmeansAssign(points, c);
      h.push(kmeansInertia(points, a2, c));
      const same = a2.every((v, i) => v === a[i]);
      a = a2;
      if (same) break;
    }
    setCentroids(c); setAssign(a); setIter(it); setHistory(h); setPhase("update");
    setNote(`Converged after ${plural(it, "update step")} with inertia ${fmt(kmeansInertia(points, a, c), 2)}. Try another seed: different starts can converge to different (worse) solutions.`);
  };

  const sx = linear(D, [48, 548]), sy = linear(D, [320, 12]);
  const onClick = (e: React.PointerEvent) => {
    if (!svgRef.current) return;
    const p = svgPoint(e, svgRef.current);
    const pt = { x: sx.invert(p.x), y: sy.invert(p.y) };
    if (pt.x < D[0] || pt.x > D[1] || pt.y < D[0] || pt.y > D[1]) return;
    if (!ready) {
      const next = [...centroids, pt];
      setCentroids(next);
      setNote(next.length === k ? "All centroids placed. Press “Assign”." : `Placed ${next.length} of ${k}. Click to place the next.`);
    } else {
      setPoints((ps) => [...ps, pt]);
      setAssign(null); setPhase("assign");
      setNote("Added a point. Assignments are stale, so press “Assign” again.");
    }
  };

  const failNote: Record<Dataset, string> = {
    blobs: "Round, similar-size, well-separated groups: the case K-means is built for.",
    stretched: "Two long horizontal bands. K-means prefers round clusters, so with K = 2 it often cuts the bands vertically instead of separating them.",
    uneven: "One big diffuse group and one tiny dense group. K-means tends to split the big group and swallow the small one.",
    rings: "A ring around a core. No centroid placement can separate them, because cluster boundaries in K-means are straight lines (a Voronoi diagram). DBSCAN handles this.",
  };

  const hMax = Math.max(1, ...history);

  // Guided tour: each step rebuilds dataset, K and initialization, then plays half-steps
  // (one assign or one update per frame; history.length counts the half-steps taken).
  const setup = (d: Dataset, kk: number, how: Init, s: number) => {
    const pts = makeData(d); setDataset(d); setPoints(pts); setK(kk); setInit(how); setSeed(s); restart(pts, kk, how, s);
  };
  const halfSteps = (n: number) => (t: number) => {
    if (history.length >= Math.round(t * n)) return;
    if (phase === "assign") doAssign(); else doUpdate();
  };
  // Caption numbers come from replaying the same half-steps each tour step plays.
  const good = simulate("blobs", 3, "random", 4, 5), bad = simulate("blobs", 3, "random", 11, 7), pp = simulate("blobs", 3, "plusplus", 4, 3);
  const i0 = (v: number) => fmt(v, 0);
  const tour: TourStep[] = [
    { id: "start", caption: `${good.n} grey points in three blobs, and three × centroids dropped on random points. Nothing belongs to any cluster yet.`, apply: () => setup("blobs", 3, "random", 4) },
    { id: "assign", caption: "Assignment step: every point takes the colour of its nearest centroid. The thin lines show who belongs to whom.", apply: () => setup("blobs", 3, "random", 4), animate: halfSteps(1), animMs: 1800 },
    { id: "update", caption: `Update step: each centroid jumps to the mean of the points it owns. Inertia, the total squared distance (orange, in the readout), drops from ${i0(good.inertia[0])} to ${i0(good.inertia[1])}.`, apply: () => setup("blobs", 3, "random", 4), animate: halfSteps(2), animMs: 2000 },
    { id: "converge", caption: `Repeat assign and update until no point switches cluster. Here that takes ${plural(good.updates, "update")}, and inertia settles at ${i0(good.last)} with one centroid per blob.`, apply: () => setup("blobs", 3, "random", 4), animate: halfSteps(5), animMs: 3000 },
    { id: "bad-start", caption: `Same data, seed 11. Two centroids end up splitting the top-right blob while one stretches across the other two. It still converges, but stuck at inertia ${i0(bad.last)} instead of ${i0(good.last)}.`, apply: () => setup("blobs", 3, "random", 11), animate: halfSteps(7), animMs: 3400 },
    { id: "plusplus", caption: `k-means++ picks starting points far apart, so each blob tends to get its own centroid. From this start ${plural(pp.updates, "update")} reaches inertia ${i0(pp.last)}.`, apply: () => setup("blobs", 3, "plusplus", 4), animate: halfSteps(3), animMs: 2400 },
    { id: "rings", caption: "A ring around a core. K-means boundaries are straight lines, so it slices the ring in half instead of separating ring from core, however it starts.", apply: () => setup("rings", 2, "plusplus", 4), animate: halfSteps(9), animMs: 3400 },
  ];
  return (
    <LabFrame
      id="kmeans"
      title="K-means lab"
      subtitle={`Synthetic data. ${failNote[dataset]}`}
      onReset={() => { setDataset("blobs"); const pts = makeData("blobs"); setPoints(pts); setK(3); setInit("random"); setSeed(4); restart(pts, 3, "random", 4); }}
      presets={(["blobs", "stretched", "uneven", "rings"] as Dataset[]).map((d) => ({
        label: { blobs: "Blobs", stretched: "Stretched", uneven: "Uneven sizes", rings: "Rings" }[d],
        apply: () => { const pts = makeData(d); const kk = d === "blobs" ? 3 : 2; setDataset(d); setPoints(pts); setK(kk); restart(pts, kk, init, seed); },
      }))}
      controls={
        <>
          <Slider label="K (clusters)" value={k} min={1} max={6} onChange={(v) => { setK(v); restart(points, v); }} />
          <Segmented label="Initialization" value={init} onChange={(v) => { setInit(v); restart(points, k, v); }}
            options={[{ value: "random", label: "Random" }, { value: "plusplus", label: "k-means++" }, { value: "manual", label: "Click" }]} />
          {init !== "manual" && <Slider label="init seed" value={seed} min={1} max={20} onChange={(v) => { setSeed(v); restart(points, k, init, v); }} />}
          <div className="flex flex-wrap gap-2">
            <Button primary={phase === "assign"} disabled={!ready} onClick={doAssign}>Assign</Button>
            <Button primary={phase === "update"} disabled={!ready || !assign || phase !== "update"} onClick={doUpdate}>Update</Button>
            <Button disabled={!ready} onClick={runAll}>Run</Button>
          </div>
          <p className="text-xs text-faint">{ready ? "Click the plot to add points." : `Click to place centroid ${centroids.length + 1}.`}</p>
        </>
      }
      readout={
        <>
          <Stat label="update steps" value={iter} />
          <Stat label="next step" value={phase} />
          <Stat label="inertia Σ‖x−μ‖²" value={inertia === null ? "–" : fmt(inertia, 2)} color="var(--c-orange)" />
          {history.length > 1 && (
            <svg viewBox="0 0 200 50" className="w-full mt-2" role="img" aria-label="Inertia over steps">
              <polyline fill="none" stroke="var(--c-orange)" strokeWidth={2} points={history.map((h, i) => `${(i / Math.max(1, history.length - 1)) * 196 + 2},${48 - (h / hMax) * 44}`).join(" ")} />
            </svg>
          )}
        </>
      }
      interpretation={note}
      tour={tour}
    >
      <Plot title="K-means clustering" x={D} y={D} xLabel="feature 1" yLabel="feature 2" svgProps={{ ref: svgRef, onPointerDown: onClick }}>
        {() => (
          <>
            {assign && points.map((p, i) => (
              <line key={`l${i}`} x1={sx(p.x)} y1={sy(p.y)} x2={sx(centroids[assign[i]].x)} y2={sy(centroids[assign[i]].y)} stroke={COLORS[assign[i] % 6]} strokeWidth={0.6} opacity={0.35} />
            ))}
            {points.map((p, i) => (
              <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={4.5} fill={assign ? COLORS[assign[i] % 6] : "var(--faint)"} opacity={0.85} style={{ transition: "fill 250ms" }} />
            ))}
            {centroids.map((c, j) => (
              <g key={`c${j}`} transform={`translate(${sx(c.x)} ${sy(c.y)})`} style={{ transition: "transform 400ms cubic-bezier(0.16,1,0.3,1)" }}>
                <path d="M-9,-9 L9,9 M9,-9 L-9,9" stroke="var(--surface)" strokeWidth={6} />
                <path d="M-9,-9 L9,9 M9,-9 L-9,9" stroke={COLORS[j % 6]} strokeWidth={3} />
              </g>
            ))}
          </>
        )}
      </Plot>
      <Legend items={[{ label: "× centroids (μ)", color: "var(--text)" }, { label: "points colored by assigned cluster", color: "var(--c-blue)" }]} />
    </LabFrame>
  );
}
