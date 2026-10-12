"use client";
import { useMemo, useRef, useState } from "react";
import { fmt, gaussian, rng } from "@/lib/ml";
import { diverged, initState, step, SURFACES, type OptimizerId, type OptState, type P2, type Surface } from "@/lib/optim";
import { Button, LabFrame, Legend, linear, Plot, Slider, Stat, svgPoint, Toggle, type TourStep } from "./ui";

const OPTS: { id: OptimizerId; label: string; color: string }[] = [
  { id: "sgd", label: "Gradient descent", color: "var(--c-blue)" },
  { id: "momentum", label: "Momentum (β = 0.9)", color: "var(--c-orange)" },
  { id: "adam", label: "Adam", color: "var(--c-purple)" },
];

type Track = { st: OptState; path: P2[]; dead: boolean };

/** Loss after n noise-free steps from the surface's start (used to write tour captions from real runs). */
const lossAfter = (s: Surface, o: OptimizerId, lr: number, n: number) => {
  let st = initState(s.start);
  for (let i = 0; i < n; i++) st = step(o, st, s.grad(st.p), lr);
  return s.f(st.p);
};
const W = 560, H = 360, M = { t: 12, r: 12, b: 40, l: 48 };

export default function GradientDescentLab() {
  const [surfaceId, setSurfaceId] = useState("bowl");
  const s = SURFACES.find((x) => x.id === surfaceId)!;
  const [lr, setLr] = useState(0.1);
  const [noise, setNoise] = useState(0);
  const [enabled, setEnabled] = useState<Record<OptimizerId, boolean>>({ sgd: true, momentum: false, adam: false });
  const fresh = (p: P2): Record<OptimizerId, Track> => ({
    sgd: { st: initState(p), path: [p], dead: false },
    momentum: { st: initState(p), path: [p], dead: false },
    adam: { st: initState(p), path: [p], dead: false },
  });
  const [start, setStart] = useState<P2>(s.start);
  const [tracks, setTracks] = useState(() => fresh(s.start));
  const [steps, setSteps] = useState(0);
  const noiseRng = useRef(rng(17));
  const svgRef = useRef<SVGSVGElement>(null);

  const sx = linear(s.x, [M.l, W - M.r]), sy = linear(s.y, [H - M.b, M.t]);

  // Heatmap of log-loss so valleys stay visible across scales.
  const heat = useMemo(() => {
    const nx = 56, ny = 36, cells: { x: number; y: number; v: number }[] = [];
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const x = s.x[0] + ((i + 0.5) / nx) * (s.x[1] - s.x[0]);
      const y = s.y[0] + ((j + 0.5) / ny) * (s.y[1] - s.y[0]);
      cells.push({ x, y, v: Math.log(1 + s.f([x, y]) - Math.min(0, s.f(s.minimum))) });
    }
    const max = Math.max(...cells.map((c) => c.v));
    return { cells: cells.map((c) => ({ ...c, v: c.v / max })), cw: (W - M.l - M.r) / nx, ch: (H - M.t - M.b) / ny };
  }, [s]);

  const reset = (p: P2 = start) => { setTracks(fresh(p)); setSteps(0); noiseRng.current = rng(17); };

  const advance = (n: number) => {
    setTracks((cur) => {
      const next = { ...cur };
      for (const o of OPTS) {
        if (!enabled[o.id]) continue;
        let t = next[o.id];
        for (let i = 0; i < n && !t.dead; i++) {
          const g = s.grad(t.st.p);
          const gn: P2 = noise ? [g[0] + gaussian(noiseRng.current) * noise, g[1] + gaussian(noiseRng.current) * noise] : g;
          const st = step(o.id, t.st, gn, lr);
          const dead = diverged(st.p, s.f);
          t = { st, path: [...t.path, st.p], dead };
        }
        next[o.id] = t;
      }
      return next;
    });
    setSteps((x) => x + n);
  };

  const main = tracks.sgd;
  const g = s.grad(main.st.p);
  const gNorm = Math.hypot(g[0], g[1]);
  const anyDead = OPTS.some((o) => enabled[o.id] && tracks[o.id].dead);
  const clampPt = (p: P2): P2 => [Math.min(s.x[1] + 1, Math.max(s.x[0] - 1, p[0])), Math.min(s.y[1] + 1, Math.max(s.y[0] - 1, p[1]))];

  // Guided tour (Watch mode + explainers). Each step sets the full state it needs, then animates.
  const setup = (id: string, eta: number, opts: Record<OptimizerId, boolean>, n = 0) => {
    const sf = SURFACES.find((x) => x.id === id)!;
    setSurfaceId(id); setStart(sf.start); setLr(eta); setNoise(0); setEnabled(opts);
    setTracks(fresh(sf.start)); setSteps(0); noiseRng.current = rng(17);
    return n;
  };
  const walkTo = (n: number) => (t: number) => { const target = Math.round(t * n); if (target > steps) advance(target - steps); };
  const onlyGd = { sgd: true, momentum: false, adam: false };
  const bowl = SURFACES[0];
  const l = (o: OptimizerId, eta: number, n: number) => fmt(lossAfter(bowl, o, eta, n), 2);
  const tour: TourStep[] = [
    { id: "start", caption: "This is a loss surface seen from above. Darker blue means higher loss, and the ring marks the minimum we want to reach. We start at the blue dot.", apply: () => setup("bowl", 0.1, onlyGd) },
    { id: "arrow", caption: "The teal arrow is minus the gradient, scaled by the learning rate: the direction of steepest descent from where we stand.", apply: () => setup("bowl", 0.1, onlyGd) },
    { id: "descend", caption: `Each step moves along that arrow, then recomputes it. Steps shrink on their own as the slope flattens: after 25 steps the loss is down from ${fmt(bowl.f(bowl.start), 1)} to ${l("sgd", 0.1, 25)}.`, apply: () => setup("bowl", 0.1, onlyGd), animate: walkTo(25), animMs: 2600 },
    { id: "zigzag", caption: "Raise the learning rate to 0.18 and the path zigzags across the narrow valley: each step overshoots the steep direction, but by a little less each time.", apply: () => setup("bowl", 0.18, onlyGd), animate: walkTo(25), animMs: 2600 },
    { id: "diverge", caption: `At 0.21 we are past the stability limit (η × 10 > 2 here): every overshoot is bigger than the last, the path flies off the map, and after 20 steps the loss has grown to ${fmt(lossAfter(bowl, "sgd", 0.21, 20), 0)}.`, apply: () => setup("bowl", 0.21, onlyGd), animate: walkTo(20), animMs: 2400 },
    { id: "optimizers", caption: `Three optimizers, η = 0.03, 60 steps. Momentum (orange) builds up speed, overshoots and swirls around the minimum, but still ends closest (loss ${l("momentum", 0.03, 60)}). Gradient descent (blue) crawls (${l("sgd", 0.03, 60)}). Adam (purple) takes similar-sized steps in both directions, so it heads straight for the minimum, but each step is only about η long (${l("adam", 0.03, 60)}).`, apply: () => setup("bowl", 0.03, { sgd: true, momentum: true, adam: true }), animate: walkTo(60), animMs: 3200 },
  ];

  const growing = enabled.sgd && !main.dead && steps > 0 && s.f(main.st.p) > s.f(start);
  const interpretation = anyDead || growing
    ? `${anyDead ? "Diverged" : `Diverging: after ${steps} steps the loss has grown from ${fmt(s.f(start), 2)} to ${fmt(s.f(main.st.p), 2)}`}. Each step overshot the minimum by more than the last, so the loss ${anyDead ? "exploded" : "keeps growing"}. ${s.id === "bowl" ? "On the narrow bowl the steep direction has curvature 10, and plain gradient descent is only stable while η × 10 < 2." : "Lower the learning rate."}`
    : steps === 0
      ? s.note
      : `After ${steps} steps, gradient descent sits at loss ${fmt(s.f(main.st.p), 4)} with gradient norm ${fmt(gNorm, 3)}. The arrow shows −∇L: the direction of steepest descent, scaled by η. ${noise ? "Noisy gradients mimic mini-batches: the path jitters but still trends downhill." : ""}`;

  return (
    <LabFrame
      id="gradient-descent"
      title="Gradient descent lab"
      subtitle={`${s.label}. Click the surface to choose a starting point.`}
      onReset={() => { setSurfaceId("bowl"); setLr(0.1); setNoise(0); setEnabled({ sgd: true, momentum: false, adam: false }); setStart(SURFACES[0].start); setTracks(fresh(SURFACES[0].start)); setSteps(0); }}
      presets={[
        ...SURFACES.map((x) => ({ label: x.label, apply: () => { setSurfaceId(x.id); setStart(x.start); setTracks(fresh(x.start)); setSteps(0); if (x.id !== "bowl") setLr(Math.min(lr, 0.03)); } })),
        { label: "Make it diverge", apply: () => { setSurfaceId("bowl"); setStart(SURFACES[0].start); setTracks(fresh(SURFACES[0].start)); setSteps(0); setLr(0.21); setEnabled({ sgd: true, momentum: false, adam: false }); } },
        { label: "Compare all three", apply: () => { setEnabled({ sgd: true, momentum: true, adam: true }); reset(); } },
      ]}
      controls={
        <>
          <Slider label="learning rate η" value={lr} min={0.001} max={0.25} step={0.001} onChange={(v) => { setLr(v); reset(); }} format={(v) => fmt(v, 3)} />
          <Slider label="gradient noise (mini-batch)" value={noise} min={0} max={3} step={0.1} onChange={(v) => { setNoise(v); reset(); }} format={(v) => fmt(v, 1)} />
          <fieldset className="space-y-1.5">
            <legend className="text-[0.8rem] text-muted mb-1">Optimizers</legend>
            {OPTS.map((o) => (
              <Toggle key={o.id} label={o.label} checked={enabled[o.id]} onChange={(v) => { setEnabled({ ...enabled, [o.id]: v }); reset(); }} />
            ))}
          </fieldset>
          <div className="flex flex-wrap gap-2">
            <Button primary onClick={() => advance(1)}>Step</Button>
            <Button onClick={() => advance(10)}>+10</Button>
            <Button onClick={() => advance(100)}>+100</Button>
          </div>
        </>
      }
      readout={
        <>
          <Stat label="steps" value={steps} />
          {OPTS.filter((o) => enabled[o.id]).map((o) => (
            <Stat key={o.id} label={`loss (${o.id})`} value={tracks[o.id].dead ? "diverged" : fmt(s.f(tracks[o.id].st.p), 4)} color={o.color} />
          ))}
          {enabled.sgd && <Stat label="θ (GD)" value={`(${fmt(main.st.p[0], 2)}, ${fmt(main.st.p[1], 2)})`} />}
          {enabled.sgd && <Stat label="∇L (GD)" value={`(${fmt(g[0], 2)}, ${fmt(g[1], 2)})`} />}
        </>
      }
      interpretation={interpretation}
      tour={tour}
    >
      <Plot
        title="Loss surface with optimizer paths"
        x={s.x} y={s.y} xLabel="θ₁" yLabel="θ₂"
        svgProps={{
          ref: svgRef,
          onPointerDown: (e) => {
            if (!svgRef.current) return;
            const p = svgPoint(e, svgRef.current);
            const np: P2 = [+sx.invert(p.x).toFixed(2), +sy.invert(p.y).toFixed(2)];
            if (np[0] < s.x[0] || np[0] > s.x[1] || np[1] < s.y[0] || np[1] > s.y[1]) return;
            setStart(np); reset(np);
          },
        }}
      >
        {() => (
          <>
            <defs>
              <clipPath id="gd-clip"><rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} /></clipPath>
              <marker id="gd-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--c-teal)" />
              </marker>
            </defs>
            <g clipPath="url(#gd-clip)">
              {heat.cells.map((c, i) => (
                <rect key={i} x={sx(c.x) - heat.cw / 2} y={sy(c.y) - heat.ch / 2} width={heat.cw + 1} height={heat.ch + 1} fill={`color-mix(in srgb, var(--c-blue) ${Math.round(4 + c.v * 42)}%, var(--surface))`} />
              ))}
              <circle cx={sx(s.minimum[0])} cy={sy(s.minimum[1])} r={5} fill="none" stroke="var(--text)" strokeWidth={1.5} />
              {OPTS.filter((o) => enabled[o.id]).map((o) => {
                const pts = tracks[o.id].path.map(clampPt);
                return (
                  <g key={o.id}>
                    <polyline fill="none" stroke={o.color} strokeWidth={2} points={pts.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")} />
                    {pts.length < 80 && pts.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={2.5} fill={o.color} />)}
                    <circle cx={sx(pts[pts.length - 1][0])} cy={sy(pts[pts.length - 1][1])} r={6} fill={o.color} stroke="var(--surface)" strokeWidth={2} />
                  </g>
                );
              })}
              {enabled.sgd && !main.dead && gNorm > 1e-6 && (() => {
                const [px, py] = main.st.p;
                const len = Math.min(1.2, lr * gNorm) / gNorm;
                return <line x1={sx(px)} y1={sy(py)} x2={sx(px - g[0] * len * 1)} y2={sy(py - g[1] * len)} stroke="var(--c-teal)" strokeWidth={2.5} markerEnd="url(#gd-arrow)" />;
              })()}
            </g>
          </>
        )}
      </Plot>
      <Legend items={[
        ...OPTS.filter((o) => enabled[o.id]).map((o) => ({ label: o.label, color: o.color })),
        { label: "−η∇L (next step)", color: "var(--c-teal)" },
        { label: "darker = higher loss; ring = minimum", color: "var(--faint)" },
      ]} />
    </LabFrame>
  );
}
