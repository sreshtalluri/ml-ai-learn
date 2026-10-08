"use client";
import { useMemo, useState } from "react";
import { fmt, gaussian, gini, rng } from "@/lib/ml";
import { Button, LabFrame, Legend, Plot, Segmented, Slider, Stat } from "./ui";

type Pt = { x: number; y: number; label: number };
type Feat = "x" | "y";
interface Node { id: number; idx: number[]; box: [number, number, number, number]; depth: number; split?: { feat: Feat; thr: number; left: number; right: number } }

// SYNTHETIC: buy (1) mostly when age > 40 AND income > 60, plus label noise.
function makeData(): Pt[] {
  const r = rng(21);
  return Array.from({ length: 60 }, () => {
    const x = 20 + r() * 50, y = 20 + r() * 100;
    const p = x > 40 && y > 60 ? 0.9 : 0.12;
    return { x: +x.toFixed(1), y: +y.toFixed(1), label: r() < p + gaussian(r) * 0.02 ? 1 : 0 };
  });
}
const NAMES: Record<Feat, string> = { x: "age", y: "income ($k)" };
const DOM = { x: [20, 70] as [number, number], y: [20, 120] as [number, number] };
const counts = (pts: Pt[], idx: number[]) => [idx.filter((i) => pts[i].label === 0).length, idx.filter((i) => pts[i].label === 1).length];

export function bestSplit(pts: Pt[], idx: number[]) {
  let best = { feat: "x" as Feat, thr: 0, dec: -1 };
  const parent = gini(counts(pts, idx));
  for (const feat of ["x", "y"] as Feat[]) {
    const vals = [...new Set(idx.map((i) => pts[i][feat]))].sort((a, b) => a - b);
    for (let k = 0; k < vals.length - 1; k++) {
      const thr = (vals[k] + vals[k + 1]) / 2;
      const L = idx.filter((i) => pts[i][feat] < thr), R = idx.filter((i) => pts[i][feat] >= thr);
      const w = (L.length * gini(counts(pts, L)) + R.length * gini(counts(pts, R))) / idx.length;
      if (parent - w > best.dec) best = { feat, thr: +thr.toFixed(1), dec: parent - w };
    }
  }
  return best;
}

export default function DecisionTreeLab() {
  const pts = useMemo(() => makeData(), []);
  const root: Node = { id: 0, idx: pts.map((_, i) => i), box: [DOM.x[0], DOM.x[1], DOM.y[0], DOM.y[1]], depth: 0 };
  const [nodes, setNodes] = useState<Node[]>([root]);
  const [sel, setSel] = useState(0);
  const [feat, setFeat] = useState<Feat>("x");
  const [thr, setThr] = useState(45);

  const node = nodes.find((n) => n.id === sel)!;
  const L = node.idx.filter((i) => pts[i][feat] < thr), R = node.idx.filter((i) => pts[i][feat] >= thr);
  const gP = gini(counts(pts, node.idx)), gL = gini(counts(pts, L)), gR = gini(counts(pts, R));
  const weighted = node.idx.length ? (L.length * gL + R.length * gR) / node.idx.length : 0;
  const leaves = nodes.filter((n) => !n.split);
  const leafOf = (p: Pt) => {
    let n = nodes[0];
    while (n.split) n = nodes.find((m) => m.id === (p[n.split!.feat] < n.split!.thr ? n.split!.left : n.split!.right))!;
    return n;
  };
  const majority = (n: Node) => { const c = counts(pts, n.idx); return c[1] > c[0] ? 1 : 0; };
  const acc = pts.filter((p) => majority(leafOf(p)) === p.label).length / pts.length;
  const [lo, hi] = feat === "x" ? [node.box[0], node.box[1]] : [node.box[2], node.box[3]];
  const canSplit = !node.split && L.length > 0 && R.length > 0 && node.depth < 4;

  const commit = () => {
    if (!canSplit) return;
    const next = Math.max(...nodes.map((n) => n.id)) + 1;
    const [x0, x1, y0, y1] = node.box;
    const lb: Node["box"] = feat === "x" ? [x0, thr, y0, y1] : [x0, x1, y0, thr];
    const rb: Node["box"] = feat === "x" ? [thr, x1, y0, y1] : [x0, x1, thr, y1];
    setNodes((ns) => [...ns.map((n) => (n.id === sel ? { ...n, split: { feat, thr, left: next, right: next + 1 } } : n)),
      { id: next, idx: L, box: lb, depth: node.depth + 1 }, { id: next + 1, idx: R, box: rb, depth: node.depth + 1 }]);
    setSel(next);
  };
  const suggest = () => { const b = bestSplit(pts, node.idx); setFeat(b.feat); setThr(b.thr); };
  const reset = () => { setNodes([root]); setSel(0); setFeat("x"); setThr(45); };
  const pathText = (n: Node): string => {
    const parent = nodes.find((m) => m.split && (m.split.left === n.id || m.split.right === n.id));
    if (!parent) return "root";
    const s = parent.split!;
    return `${pathText(parent) === "root" ? "" : pathText(parent) + " and "}${NAMES[s.feat]} ${s.left === n.id ? "<" : "≥"} ${s.thr}`;
  };

  return (
    <LabFrame
      id="decision-tree"
      title="Build a decision tree"
      subtitle="Synthetic customers. Pick a leaf (click a region), choose a feature and threshold, then split."
      onReset={reset}
      presets={[{ label: "Suggest best split", apply: suggest }]}
      controls={
        <>
          <p className="text-[0.8rem] text-muted">Working on: <span className="text-ink">{pathText(node)}</span> ({node.idx.length} points)</p>
          <Segmented label="Split on" value={feat} onChange={(f) => { setFeat(f); const [a, b] = f === "x" ? [node.box[0], node.box[1]] : [node.box[2], node.box[3]]; setThr(+((a + b) / 2).toFixed(1)); }} options={[{ value: "x", label: "age" }, { value: "y", label: "income" }]} />
          <Slider label={`threshold (${NAMES[feat]})`} value={Math.min(hi, Math.max(lo, thr))} min={lo} max={hi} step={0.5} onChange={setThr} format={(v) => fmt(v, 1)} />
          <div className="flex flex-wrap gap-2">
            <Button primary onClick={commit} disabled={!canSplit}>Split</Button>
            <Button onClick={suggest} disabled={!!node.split}>Best split</Button>
          </div>
          {node.split && <p className="text-xs text-faint">This node is already split. Click a shaded leaf region.</p>}
        </>
      }
      readout={
        <>
          <Stat label="parent Gini" value={fmt(gP, 3)} />
          <Stat label={`left (${counts(pts, L).join("/")})`} value={fmt(gL, 3)} />
          <Stat label={`right (${counts(pts, R).join("/")})`} value={fmt(gR, 3)} />
          <Stat label="weighted children" value={fmt(weighted, 3)} />
          <Stat label="decrease" value={fmt(gP - weighted, 3)} color="var(--c-teal)" />
          <Stat label="leaves · accuracy" value={`${leaves.length} · ${fmt(acc * 100, 1)}%`} />
        </>
      }
      interpretation={
        <>
          <p className="text-ink">Weighted child Gini <span className="font-mono">({L.length}×{fmt(gL, 3)} + {R.length}×{fmt(gR, 3)}) / {node.idx.length} = {fmt(weighted, 3)}</span>, a decrease of {fmt(gP - weighted, 3)} from the parent.</p>
          <p className="mt-1">{leaves.length >= 6 ? "Many small leaves fit the training points ever more tightly. On new data, past a few splits, that is memorization: this is why max_depth and min_samples_leaf exist." : "A split is good when its children are purer than the parent. Try the threshold slider and watch the decrease peak; that peak is what the tree-growing algorithm searches for."}</p>
        </>
      }
    >
      <Plot title="Feature space partitioned by the tree" x={DOM.x} y={DOM.y} xLabel="age" yLabel="income ($k)">
        {({ sx, sy }) => (
          <>
            {leaves.map((n) => {
              const [x0, x1, y0, y1] = n.box;
              const m = majority(n);
              return (
                <rect key={n.id} x={sx(x0)} y={sy(y1)} width={sx(x1) - sx(x0)} height={sy(y0) - sy(y1)} onClick={() => setSel(n.id)} className="cursor-pointer"
                  fill={m ? "var(--c-orange)" : "var(--c-blue)"} opacity={n.idx.length ? 0.1 : 0.03} stroke={n.id === sel ? "var(--text)" : "none"} strokeWidth={2} />
              );
            })}
            {!node.split && (feat === "x"
              ? <line x1={sx(thr)} x2={sx(thr)} y1={sy(node.box[3])} y2={sy(node.box[2])} stroke="var(--c-purple)" strokeWidth={2.5} strokeDasharray="6 4" />
              : <line x1={sx(node.box[0])} x2={sx(node.box[1])} y1={sy(thr)} y2={sy(thr)} stroke="var(--c-purple)" strokeWidth={2.5} strokeDasharray="6 4" />)}
            {pts.map((p, i) => p.label
              ? <rect key={i} x={sx(p.x) - 4} y={sy(p.y) - 4} width={8} height={8} fill="var(--c-orange)" stroke="var(--surface)" pointerEvents="none" />
              : <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={4.5} fill="var(--c-blue)" stroke="var(--surface)" pointerEvents="none" />)}
          </>
        )}
      </Plot>
      <Legend items={[{ label: "did not buy (circles)", color: "var(--c-blue)" }, { label: "bought (squares)", color: "var(--c-orange)" }, { label: "candidate split", color: "var(--c-purple)", dashed: true }]} />
    </LabFrame>
  );
}
