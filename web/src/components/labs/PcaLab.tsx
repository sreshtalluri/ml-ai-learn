"use client";
import { useMemo, useState } from "react";
import { fmt, gaussian, mean, rng } from "@/lib/ml";
import { Button, LabFrame, Legend, Plot, Slider, Stat, Toggle, type TourStep } from "./ui";

// SYNTHETIC: 80 points from a 2D Gaussian with adjustable correlation.
function makeData(rho: number) {
  const r = rng(13);
  const pts = Array.from({ length: 80 }, () => {
    const a = gaussian(r), b = gaussian(r);
    return { x: 1.4 * a, y: 1.4 * (rho * a + Math.sqrt(1 - rho * rho) * b) * 0.8 };
  });
  const mx = mean(pts.map((p) => p.x)), my = mean(pts.map((p) => p.y));
  return pts.map((p) => ({ x: p.x - mx, y: p.y - my })); // centered
}

/** Covariance entries and the angle of PC1 for centered 2D points. */
export function pca2(pts: { x: number; y: number }[]) {
  const n = pts.length - 1;
  const sxx = pts.reduce((s, p) => s + p.x * p.x, 0) / n, syy = pts.reduce((s, p) => s + p.y * p.y, 0) / n;
  const sxy = pts.reduce((s, p) => s + p.x * p.y, 0) / n;
  const tr = sxx + syy, det = sxx * syy - sxy * sxy, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  return { sxx, syy, sxy, l1: tr / 2 + disc, l2: tr / 2 - disc, angle: 0.5 * Math.atan2(2 * sxy, sxx - syy) };
}

export default function PcaLab() {
  const [rho, setRho] = useState(0.8);
  const [deg, setDeg] = useState(0);
  const [showRes, setShowRes] = useState(true);
  const pts = useMemo(() => makeData(rho), [rho]);
  const P = pca2(pts);
  const th = (deg * Math.PI) / 180;
  const u = { x: Math.cos(th), y: Math.sin(th) };
  const proj = pts.map((p) => p.x * u.x + p.y * u.y);
  const varAlong = proj.reduce((s, v) => s + v * v, 0) / (pts.length - 1);
  const total = P.sxx + P.syy;
  const recon = total - varAlong; // average squared distance to the line (per point, sample-variance units)
  const pc1Deg = ((P.angle * 180) / Math.PI + 180) % 180;
  const D: [number, number] = [-4.5, 4.5];

  // Guided tour: each step sets correlation, axis angle and projection lines.
  const pc1Of = (r: number) => ((pca2(makeData(r)).angle * 180) / Math.PI + 180) % 180;
  const setup = (r: number, d: number) => { setRho(r); setDeg(+d.toFixed(1)); setShowRes(true); };
  const tour: TourStep[] = [
    { id: "cloud", caption: "Eighty correlated points, centred at the origin. The teal line is a candidate axis: each point drops onto it along an orange line, keeping one number instead of two.", apply: () => setup(0.8, 0) },
    { id: "rotate", caption: "Rotate the axis and watch the dot on the right trace how much variance the projected points keep. It peaks where the axis runs along the cloud.", apply: () => setup(0.8, 0), animate: (t) => setDeg(Math.round(t * 360) / 2), animMs: 3500 },
    { id: "pc1", caption: "That peak is PC1, at about 41.5° here. Projecting onto it keeps 91% of the variance, and the orange lines are as short as they can be.", apply: () => setup(0.8, pc1Of(0.8)) },
    { id: "trade-off", caption: "Turn 90° to PC2 and the orange lines grow as the teal share falls to 9%. Kept plus lost always adds up to the same total.", apply: () => setup(0.8, pc1Of(0.8)), animate: (t) => setDeg(+((pc1Of(0.8) + 90 * t) % 180).toFixed(1)), animMs: 2800 },
    { id: "correlation", caption: "Now loosen the correlation while the axis follows PC1. As the cloud turns round, PC1's share drops from 97% toward about 60%: there is less to compress.", apply: () => setup(0.95, pc1Of(0.95)), animate: (t) => { const r = +(0.95 - 0.95 * t).toFixed(2); setRho(r); setDeg(+pc1Of(r).toFixed(1)); }, animMs: 3200 },
    { id: "takeaway", caption: "PCA picks the direction of most variance, which is the same as the direction of least squared projection error. With strong correlation, one number per point is almost enough.", apply: () => setup(0.8, pc1Of(0.8)) },
  ];

  return (
    <LabFrame
      id="pca"
      title="Find the first principal component"
      subtitle="Synthetic, centered 2D data. Rotate the axis: PCA picks the direction that keeps the most variance when you project onto it."
      tour={tour}
      onReset={() => { setRho(0.8); setDeg(0); setShowRes(true); }}
      presets={[
        { label: "Snap to PC1", apply: () => setDeg(+pc1Deg.toFixed(1)) },
        { label: "Perpendicular (PC2)", apply: () => setDeg(+((pc1Deg + 90) % 180).toFixed(1)) },
        { label: "No correlation", apply: () => setRho(0) },
      ]}
      controls={
        <>
          <Slider label="axis angle" value={deg} min={0} max={180} step={0.5} onChange={setDeg} format={(v) => `${fmt(v, 1)}°`} />
          <Slider label="correlation of the data" value={rho} min={-0.95} max={0.95} step={0.05} onChange={setRho} format={(v) => fmt(v, 2)} />
          <Toggle label="Show projection lines" checked={showRes} onChange={setShowRes} />
          <Button primary onClick={() => setDeg(+pc1Deg.toFixed(1))}>Snap to PC1</Button>
        </>
      }
      readout={
        <>
          <Stat label="variance on this axis" value={fmt(varAlong, 3)} color="var(--c-teal)" />
          <Stat label="total variance" value={fmt(total, 3)} />
          <Stat label="share kept" value={`${fmt((varAlong / total) * 100, 1)}%`} />
          <Stat label="lost (residual)" value={fmt(recon, 3)} color="var(--c-orange)" />
          <Stat label="PC1 angle · λ₁" value={`${fmt(pc1Deg, 1)}° · ${fmt(P.l1, 3)}`} />
          <Stat label="λ₂" value={fmt(P.l2, 3)} />
        </>
      }
      interpretation={
        <>
          <p className="text-ink">Kept + lost always equals the total ({fmt(varAlong, 3)} + {fmt(recon, 3)} = {fmt(total, 3)}). Maximizing the variance you keep is the same as minimizing the squared distances you throw away.</p>
          <p className="mt-1">{Math.abs(rho) < 0.15
            ? "With uncorrelated data of similar spread every direction keeps about the same variance: PCA has nothing to compress, and PC1 is nearly arbitrary."
            : Math.abs(((deg - pc1Deg + 540) % 180) - 90) > 85
              ? `You're on PC1: one number per point keeps ${fmt((P.l1 / total) * 100, 1)}% of the variance (λ₁ / (λ₁ + λ₂)).`
              : "Rotate toward the long axis of the cloud. The teal share rises until the axis lines up with PC1, the eigenvector with the largest eigenvalue."}</p>
        </>
      }
    >
      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div>
          <Plot title="Projecting points onto an axis" x={D} y={D} xLabel="feature 1 (centered)" yLabel="feature 2 (centered)" height={420} width={460}>
            {({ sx, sy }) => (
              <>
                <line x1={sx(-5 * Math.cos(P.angle))} y1={sy(-5 * Math.sin(P.angle))} x2={sx(5 * Math.cos(P.angle))} y2={sy(5 * Math.sin(P.angle))} stroke="var(--faint)" strokeDasharray="3 4" />
                <line x1={sx(-5 * u.x)} y1={sy(-5 * u.y)} x2={sx(5 * u.x)} y2={sy(5 * u.y)} stroke="var(--c-teal)" strokeWidth={2.5} />
                {showRes && pts.map((p, i) => <line key={`l${i}`} x1={sx(p.x)} y1={sy(p.y)} x2={sx(proj[i] * u.x)} y2={sy(proj[i] * u.y)} stroke="var(--c-orange)" strokeWidth={1} opacity={0.6} />)}
                {pts.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3.5} fill="var(--c-blue)" stroke="var(--surface)" />)}
                {pts.map((_, i) => <circle key={`p${i}`} cx={sx(proj[i] * u.x)} cy={sy(proj[i] * u.y)} r={2.5} fill="var(--c-teal)" />)}
              </>
            )}
          </Plot>
          <Legend items={[{ label: "your axis", color: "var(--c-teal)" }, { label: "PC1", color: "var(--faint)", dashed: true }, { label: "lost distance", color: "var(--c-orange)" }]} />
        </div>
        <div>
          <Plot title="Variance kept as the axis rotates" x={[0, 180]} y={[0, P.l1 * 1.1]} xLabel="angle (°)" yLabel="variance" height={300} width={360}>
            {({ sx, sy }) => {
              const curve = Array.from({ length: 181 }, (_, d) => {
                const t = (d * Math.PI) / 180;
                return { d, v: P.sxx * Math.cos(t) ** 2 + 2 * P.sxy * Math.sin(t) * Math.cos(t) + P.syy * Math.sin(t) ** 2 };
              });
              return (
                <>
                  <polyline fill="none" stroke="var(--c-teal)" strokeWidth={2} points={curve.map((c) => `${sx(c.d)},${sy(c.v)}`).join(" ")} />
                  <circle cx={sx(deg)} cy={sy(varAlong)} r={6} fill="var(--c-teal)" stroke="var(--surface)" strokeWidth={2} />
                  <line x1={sx(pc1Deg)} x2={sx(pc1Deg)} y1={12} y2={260} stroke="var(--faint)" strokeDasharray="3 4" />
                </>
              );
            }}
          </Plot>
          <p className="text-xs text-muted mt-1">Variance along direction <i>u</i> is <i>uᵀCu</i>. Its maximum is λ₁ at PC1; its minimum is λ₂, 90° away.</p>
        </div>
      </div>
    </LabFrame>
  );
}
