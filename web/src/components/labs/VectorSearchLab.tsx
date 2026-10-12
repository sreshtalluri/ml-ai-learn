"use client";
import { useMemo, useRef, useState } from "react";
import { fmt, gaussian, kmeansAssign, kmeansUpdate, rng, type Point2 } from "@/lib/ml";
import { LabFrame, Legend, Plot, Segmented, Slider, Stat, svgPoint, type TourStep } from "./ui";

// SYNTHETIC 2D "embeddings": Gaussian blobs, fixed seed. Mirrors guide/code/16-rag/vector_search.py.
export function makePoints(n = 500, seed = 11): Point2[] {
  const r = rng(seed);
  const centers = Array.from({ length: 9 }, () => ({ x: 1 + r() * 8, y: 1 + r() * 8 }));
  return Array.from({ length: n }, () => {
    const c = centers[Math.floor(r() * centers.length)];
    return { x: c.x + gaussian(r) * 0.75, y: c.y + gaussian(r) * 0.75 };
  });
}

const d2 = (a: Point2, b: Point2) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;

export interface IvfIndex { centroids: Point2[]; lists: number[][] }

/** k-means (seeded init, fixed iterations) then inverted lists of point ids per cell. */
export function buildIvf(points: Point2[], nlist: number, seed = 3, iters = 15): IvfIndex {
  const r = rng(seed);
  let centroids = Array.from({ length: nlist }, () => ({ ...points[Math.floor(r() * points.length)] }));
  let assign = kmeansAssign(points, centroids);
  for (let i = 0; i < iters; i++) { centroids = kmeansUpdate(points, assign, centroids); assign = kmeansAssign(points, centroids); }
  const lists: number[][] = centroids.map(() => []);
  assign.forEach((c, i) => lists[c].push(i));
  return { centroids, lists };
}

export function bruteForce(points: Point2[], q: Point2, k: number): number[] {
  return points.map((p, i) => [d2(p, q), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, k).map((x) => x[1]);
}

export function ivfSearch(points: Point2[], index: IvfIndex, q: Point2, k: number, nprobe: number) {
  const cells = index.centroids.map((c, j) => [d2(c, q), j]).sort((a, b) => a[0] - b[0]).slice(0, nprobe).map((x) => x[1]);
  const cand = cells.flatMap((c) => index.lists[c]);
  const top = cand.map((i) => [d2(points[i], q), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, k).map((x) => x[1]);
  return { top, cells, scanned: cand.length, distances: index.centroids.length + cand.length };
}

export const recallAtK = (found: number[], exact: number[]) => found.filter((i) => exact.includes(i)).length / exact.length;

/** Expected distance computations for balanced cells: nlist centroids + nprobe/nlist of N. */
export const ivfCost = (N: number, nlist: number, nprobe: number) => nlist + (nprobe / nlist) * N;

const POINTS = makePoints();
const EVAL_QUERIES = (() => { const r = rng(99); return Array.from({ length: 100 }, () => ({ x: 0.5 + r() * 9, y: 0.5 + r() * 9 })); })();
const D: [number, number] = [0, 10];

/** Average recall@k and distance computations over the fixed evaluation queries. */
function evalAvg(index: IvfIndex, k: number, nprobe: number) {
  let rs = 0, cs = 0;
  for (const e of EVAL_QUERIES) { const s = ivfSearch(POINTS, index, e, k, nprobe); rs += recallAtK(s.top, bruteForce(POINTS, e, k)); cs += s.distances; }
  return { recall: rs / EVAL_QUERIES.length, cost: cs / EVAL_QUERIES.length };
}

/** Colour cells so that cells sharing a border (some point's two nearest centroids) get different colours. */
function cellColours(points: Point2[], index: IvfIndex): number[] {
  const n = index.centroids.length;
  const adj = Array.from({ length: n }, () => new Set<number>());
  for (const p of points) {
    const [a, b] = index.centroids.map((c, j) => [d2(c, p), j]).sort((x, y) => x[0] - y[0]);
    if (b) { adj[a[1]].add(b[1]); adj[b[1]].add(a[1]); }
  }
  const col = new Array<number>(n).fill(-1);
  for (const j of [...Array(n).keys()].sort((x, y) => adj[y].size - adj[x].size)) {
    let c = 0;
    while ([...adj[j]].some((o) => col[o] === c)) c++;
    col[j] = c; // ponytail: greedy colouring, may need more than 6 colours (then wraps) on unusual layouts
  }
  return col;
}

// Tour numbers, computed from the real index so captions always match the screen.
const CENTER = { x: 5, y: 5 };
const IVF16 = buildIvf(POINTS, 16);
const EDGE = boundaryQuery(IVF16, POINTS); // a query with low recall at nprobe 1
const probeAt = (q: Point2, nprobe: number) => { const s = ivfSearch(POINTS, IVF16, q, 10, nprobe); return { hit: Math.round(recallAtK(s.top, bruteForce(POINTS, q, 10)) * 10), dist: s.distances }; };
const T = { center: probeAt(CENTER, 1), edge1: probeAt(EDGE, 1), edge2: probeAt(EDGE, 2), avg1: evalAvg(IVF16, 10, 1).recall, avg2: evalAvg(IVF16, 10, 2).recall };

export default function VectorSearchLab() {
  const [nlist, setNlist] = useState(16);
  const [nprobe, setNprobe] = useState(2);
  const [k, setK] = useState(10);
  const [q, setQ] = useState<Point2>({ x: 5, y: 5 });
  const [view, setView] = useState<"scan" | "cells">("scan");
  const [dragging, setDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  const index = useMemo(() => buildIvf(POINTS, nlist), [nlist]);
  const probe = Math.min(nprobe, nlist);
  const res = ivfSearch(POINTS, index, q, k, probe);
  const exact = bruteForce(POINTS, q, k);
  const recall = recallAtK(res.top, exact);
  const avg = useMemo(() => evalAvg(index, k, probe), [index, k, probe]);
  const colours = useMemo(() => cellColours(POINTS, index), [index]);

  const cellOf = useMemo(() => { const m = new Array<number>(POINTS.length); index.lists.forEach((l, c) => l.forEach((i) => (m[i] = c))); return m; }, [index]);
  const probed = new Set(res.cells);
  const found = new Set(res.top);
  const missed = exact.filter((i) => !found.has(i));
  const N = POINTS.length;
  const PAL = ["var(--c-blue)", "var(--c-teal)", "var(--c-purple)", "var(--c-orange)", "var(--c-green)", "var(--c-red)"];

  const move = (e: React.PointerEvent) => {
    if (!svgRef.current) return;
    const p = svgPoint(e, svgRef.current);
    const sx = (x: number) => ((x - 48) / (560 - 48 - 12)) * 10, sy = (y: number) => ((360 - 40 - y) / (360 - 40 - 12)) * 10;
    setQ({ x: Math.min(10, Math.max(0, sx(p.x))), y: Math.min(10, Math.max(0, sy(p.y))) });
  };

  const reset = () => { setNlist(16); setNprobe(2); setK(10); setQ({ x: 5, y: 5 }); setView("scan"); };

  // Guided tour (Watch mode + explainers). Each step sets the full state it needs, then animates.
  const setup = (nl: number, np: number, query: Point2, v: "scan" | "cells" = "scan") => { setNlist(nl); setNprobe(np); setK(10); setQ(query); setView(v); };
  const tour: TourStep[] = [
    { id: "start", caption: "Each dot is a stored embedding and the purple star is the query. The rings mark its 10 true nearest neighbours; brute force finds them by measuring all 500 distances.", apply: () => setup(16, 16, CENTER) },
    { id: "cells", caption: "IVF first clusters the vectors with k-means. Each colour is a cell and each × its centroid; watch the space split into more, smaller cells.", apply: () => setup(2, 40, CENTER, "cells"), animate: (t) => { const v = Math.round(2 + 14 * t); if (v !== nlist) setNlist(v); }, animMs: 2800 },
    { id: "probe", caption: `At query time, compare the star with the 16 centroids and scan only the nearest cell. Blue dots were scanned, grey skipped: ${T.center.dist} distances instead of ${POINTS.length}, and ${T.center.hit} of the 10 neighbours found (teal rings; orange were missed).`, apply: () => setup(16, 1, CENTER) },
    { id: "sweep", caption: "Slide the query across the map with one cell probed. Rings turn orange whenever the star nears a cell border: true neighbours sit just across it, in a cell nobody scanned.", apply: () => setup(16, 1, { x: 1, y: EDGE.y }), animate: (t) => setQ({ x: 1 + (EDGE.x - 1) * t, y: EDGE.y }), animMs: 3400 },
    { id: "boundary", caption: `Here, near a border, IVF finds only ${T.edge1.hit} of the 10 true neighbours. The other ${10 - T.edge1.hit} (orange) are close to the star but live in a cell nobody scanned.`, apply: () => setup(16, 1, EDGE) },
    { id: "nprobe", caption: `Raise nprobe to 2 and the neighbouring cell gets scanned too: ${T.edge2.hit} of 10 found, for ${T.edge2.dist} distances instead of ${T.edge1.dist}. Averaged over 100 queries, recall climbs from ${fmt(T.avg1, 2)} to ${fmt(T.avg2, 2)}.`, apply: () => setup(16, 1, EDGE), animate: (t) => { const v = t < 0.5 ? 1 : 2; if (v !== nprobe) setNprobe(v); }, animMs: 1600 },
    { id: "exact", caption: "Probe all 16 cells and the search is exact again, but it now costs slightly more than brute force. The useful settings live in between: high recall for a fraction of the work.", apply: () => setup(16, 16, EDGE) },
  ];

  return (
    <LabFrame
      id="vector-search"
      tour={tour}
      title="Approximate nearest neighbors (IVF)"
      subtitle={`${N} synthetic 2D vectors. Click or drag to move the query. IVF clusters them into cells and scans only the nprobe cells nearest the query.`}
      onReset={reset}
      presets={[
        { label: "Boundary miss", apply: () => { setNlist(16); setNprobe(1); setK(10); setQ(EDGE); } },
        { label: "Probe everything", apply: () => setNprobe(nlist) },
        { label: "Many small cells", apply: () => { setNlist(40); setNprobe(2); } },
      ]}
      controls={
        <>
          <Slider label="clusters (nlist)" value={nlist} min={2} max={40} onChange={(v) => { setNlist(v); setNprobe(Math.min(nprobe, v)); }} />
          <Slider label="cells probed (nprobe)" value={probe} min={1} max={nlist} onChange={setNprobe} />
          <Slider label="neighbors returned (k)" value={k} min={1} max={30} onChange={setK} />
          <Segmented label="Color points by" value={view} onChange={setView} options={[{ value: "scan", label: "scanned or not" }, { value: "cells", label: "IVF cell" }]} />
        </>
      }
      readout={
        <>
          <Stat label="recall@k, this query" value={fmt(recall, 2)} color={recall === 1 ? "var(--c-green)" : "var(--c-orange)"} />
          <Stat label="distances: IVF" value={`${res.distances} (${nlist} + ${res.scanned})`} />
          <Stat label="distances: brute force" value={N} />
          <Stat label="share of brute force" value={`${fmt((100 * res.distances) / N, 1)}%`} />
          <Stat label="avg recall@k, 100 queries" value={fmt(avg.recall, 3)} color="var(--c-blue)" />
          <Stat label="avg share, 100 queries" value={`${fmt((100 * avg.cost) / N, 1)}%`} />
        </>
      }
      interpretation={
        <>
          <p className="text-ink">
            IVF compared the query with {nlist} centroids, then scanned {res.scanned} of {N} vectors in {probe} cell{probe === 1 ? "" : "s"}: {res.distances} distance computations instead of {N}.
            {" "}It found {k - missed.length} of the true {k} nearest neighbors (recall@{k} = {fmt(recall, 2)}).
          </p>
          <p className="mt-1">
            {missed.length
              ? `The ${missed.length} missed neighbor${missed.length === 1 ? " sits" : "s sit"} in a cell that was not probed, even though ${missed.length === 1 ? "it is" : "they are"} close to the query: queries near cell boundaries are where IVF loses recall. Raise nprobe to recover them at extra cost.`
              : probe >= nlist
                ? "Probing every cell is exact, but costs slightly more than brute force because the centroid comparisons are extra."
                : "Every true neighbor was inside a probed cell. Average recall over 100 queries is the number to tune, not one lucky query."}
          </p>
        </>
      }
    >
      <Plot title="IVF search" x={D} y={D} xLabel="dimension 1" yLabel="dimension 2"
        svgProps={{
          ref: svgRef,
          onPointerDown: (e) => { (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId); setDragging(true); move(e); },
          onPointerMove: (e) => dragging && move(e),
          onPointerUp: () => setDragging(false),
        }}>
        {({ sx, sy }) => (
          <>
            {POINTS.map((p, i) => (
              <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2.6}
                fill={view === "cells" ? PAL[colours[cellOf[i]] % PAL.length] : probed.has(cellOf[i]) ? "var(--c-blue)" : "var(--faint)"}
                opacity={view === "cells" ? (probed.has(cellOf[i]) ? 0.9 : 0.35) : probed.has(cellOf[i]) ? 0.85 : 0.35} />
            ))}
            {index.centroids.map((c, j) => (
              <path key={`c${j}`} d={`M${sx(c.x) - 5},${sy(c.y) - 5} L${sx(c.x) + 5},${sy(c.y) + 5} M${sx(c.x) + 5},${sy(c.y) - 5} L${sx(c.x) - 5},${sy(c.y) + 5}`}
                stroke="var(--text)" strokeWidth={probed.has(j) ? 2.4 : 1.2} opacity={probed.has(j) ? 1 : 0.5} />
            ))}
            {view === "scan" && exact.map((i) => (
              <circle key={`e${i}`} cx={sx(POINTS[i].x)} cy={sy(POINTS[i].y)} r={6} fill="none" strokeWidth={2} stroke={found.has(i) ? "var(--c-teal)" : "var(--c-orange)"} />
            ))}
            <path d={star(sx(q.x), sy(q.y), 10)} fill="var(--c-purple)" stroke="var(--surface)" strokeWidth={1.5} />
          </>
        )}
      </Plot>
      <Legend items={view === "cells" ? [
        { label: "query", color: "var(--c-purple)" },
        { label: "dot colour = IVF cell (neighbouring cells differ; faded = not probed)", color: "var(--faint)" },
        { label: "× centroids (bold = probed)", color: "var(--text)" },
      ] : [
        { label: "query", color: "var(--c-purple)" },
        { label: "true top-k, found", color: "var(--c-teal)" },
        { label: "true top-k, missed", color: "var(--c-orange)" },
        { label: "× centroids (bold = probed)", color: "var(--text)" },
      ]} />
    </LabFrame>
  );
}

function star(cx: number, cy: number, r: number) {
  return Array.from({ length: 10 }, (_, i) => {
    const a = (Math.PI / 5) * i - Math.PI / 2, rr = i % 2 ? r * 0.45 : r;
    return `${i ? "L" : "M"}${cx + rr * Math.cos(a)},${cy + rr * Math.sin(a)}`;
  }).join(" ") + "Z";
}

/** A grid query with low recall at nprobe = 1, so the preset reliably shows a boundary miss. */
function boundaryQuery(index: IvfIndex, points: Point2[]): Point2 {
  let best = { x: 5, y: 5 }, worst = 2;
  for (let x = 1; x <= 9; x += 0.5) for (let y = 1; y <= 9; y += 0.5) {
    const q = { x, y };
    const r = recallAtK(ivfSearch(points, index, q, 10, 1).top, bruteForce(points, q, 10));
    if (r < worst - 1e-9 && r > 0) { worst = r; best = q; }
  }
  return best;
}
