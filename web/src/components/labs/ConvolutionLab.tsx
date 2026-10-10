"use client";
import { useMemo, useState } from "react";
import type { Mat } from "@/lib/ml";
import { Button, LabFrame, Segmented, Slider, Stat, type TourStep } from "./ui";

const IMAGES: Record<string, Mat> = {
  "vertical edge": Array.from({ length: 7 }, () => [0, 0, 0, 9, 9, 9, 9]),
  "horizontal edge": Array.from({ length: 7 }, (_, i) => Array(7).fill(i < 3 ? 0 : 9)),
  diagonal: Array.from({ length: 7 }, (_, i) => Array.from({ length: 7 }, (_, j) => (j >= i ? 9 : 0))),
  cross: Array.from({ length: 7 }, (_, i) => Array.from({ length: 7 }, (_, j) => (i === 3 || j === 3 ? 9 : 0))),
};
const KERNELS: Record<string, Mat> = {
  "vertical edge": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
  "horizontal edge": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
  blur: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => +(v / 9).toFixed(3))),
  sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
};

/** Cross-correlation with zero padding and stride. */
export function conv2d(x: Mat, k: Mat, stride: number, pad: number): Mat {
  const P = Array.from({ length: x.length + 2 * pad }, (_, i) => Array.from({ length: x[0].length + 2 * pad }, (_, j) => x[i - pad]?.[j - pad] ?? 0));
  const out = Math.floor((P.length - k.length) / stride) + 1;
  return Array.from({ length: out }, (_, i) => Array.from({ length: out }, (_, j) =>
    k.reduce((s, row, u) => s + row.reduce((t, w, v) => t + w * P[i * stride + u][j * stride + v], 0), 0)));
}

export function maxPool(m: Mat): Mat {
  const n = Math.floor(m.length / 2);
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => Math.max(m[2 * i][2 * j], m[2 * i][2 * j + 1], m[2 * i + 1][2 * j], m[2 * i + 1][2 * j + 1])));
}

const r1 = (v: number) => String(+v.toFixed(2));

function Grid({ m, hl, ghost, onPick, title, shape, tone }: { m: Mat; hl?: (i: number, j: number) => boolean; ghost?: (i: number, j: number) => boolean; onPick?: (i: number, j: number) => void; title: string; shape: string; tone: "image" | "signed" }) {
  const max = Math.max(1e-9, ...m.flat().map(Math.abs));
  return (
    <figure>
      <figcaption className="flex justify-between text-xs mb-1"><span className="font-medium">{title}</span><span className="font-mono text-faint">{shape}</span></figcaption>
      <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${m[0]?.length ?? 1}, 2.1rem)` }}>
        {m.flatMap((row, i) => row.map((v, j) => {
          const bg = tone === "image"
            ? `color-mix(in srgb, var(--text) ${Math.round((v / 9) * 70)}%, var(--surface-2))`
            : `color-mix(in srgb, ${v >= 0 ? "var(--c-teal)" : "var(--c-orange)"} ${Math.round((Math.abs(v) / max) * 65)}%, var(--surface-2))`;
          const on = hl?.(i, j);
          return (
            <button key={`${i}-${j}`} type="button" onClick={() => onPick?.(i, j)} disabled={!onPick} aria-label={`row ${i + 1} column ${j + 1}: ${r1(v)}`}
              className={`h-8 rounded text-[0.68rem] font-mono ${on ? "ring-2 ring-[var(--c-purple)]" : ""} ${onPick ? "cursor-pointer" : "cursor-default"} ${ghost?.(i, j) ? "border border-dashed border-line text-faint" : ""}`}
              style={{ background: bg, color: tone === "image" && v > 5 ? "var(--surface)" : "var(--text)" }}>{r1(v)}</button>
          );
        }))}
      </div>
    </figure>
  );
}

export default function ConvolutionLab() {
  const [img, setImg] = useState("vertical edge");
  const [ker, setKer] = useState("vertical edge");
  const [stride, setStride] = useState(1);
  const [pad, setPad] = useState(0);
  const [pos, setPos] = useState(1);
  const X = IMAGES[img], K = KERNELS[ker];
  const Y = useMemo(() => conv2d(X, K, stride, pad), [X, K, stride, pad]);
  const n = Y.length;
  const p = Math.min(pos, n * n - 1);
  const oi = Math.floor(p / n), oj = p % n;
  const r0 = oi * stride - pad, c0 = oj * stride - pad;
  const terms = K.flatMap((row, u) => row.map((w, v) => ({ w, x: X[r0 + u]?.[c0 + v] ?? 0 })));
  const pooled = n >= 2 ? maxPool(Y) : [];
  const Xp = Array.from({ length: 7 + 2 * pad }, (_, i) => Array.from({ length: 7 + 2 * pad }, (_, j) => X[i - pad]?.[j - pad] ?? 0));

  // Guided tour: each step sets image, kernel, stride, padding and window position.
  const setup = (image: string, kernel: string, st = 1, pd = 0, at = 0) => { setImg(image); setKer(kernel); setStride(st); setPad(pd); setPos(at); };
  const slide = (t: number) => setPos(Math.round(t * (n * n - 1)));
  const V = "vertical edge", H = "horizontal edge";
  const tour: TourStep[] = [
    { id: "setup", caption: "A 7×7 image, dark on the left and bright from column 4, and a 3×3 kernel with −1s on its left column and +1s on its right.", apply: () => setup(V, V) },
    { id: "slide", caption: "The purple window slides across the image. At each stop, multiply the 9 overlapping pairs and add them up: that sum is one cell of the feature map.", apply: () => setup(V, V), animate: slide, animMs: 3400 },
    { id: "edge", caption: "Where the window straddles the edge, the −1s sit on 0s and the +1s on 9s, giving 27. On flat regions the two sides cancel to 0.", apply: () => setup(V, V, 1, 0, 1) },
    { id: "mismatch", caption: "Same kernel, horizontal edge. Left and right columns of every window are equal, so every output is 0. This kernel only sees vertical edges.", apply: () => setup(H, V, 1, 0, 7) },
    { id: "stride", caption: "Stride 2 moves the window two pixels at a time, so the feature map shrinks to 3×3 while the 9 weights stay the same.", apply: () => setup(V, V, 2), animate: slide, animMs: 2400 },
    { id: "padding", caption: "Padding 1 adds a ring of zeros around the image. Windows can now centre on border pixels, so the output stays 7×7. The orange cells on the right border are fake edges made by those zeros.", apply: () => setup(V, V, 1, 1), animate: slide, animMs: 3400 },
    { id: "pool", caption: "Max pooling keeps the largest value in each 2×2 block of the feature map, so 5×5 becomes 2×2 (the leftover last row and column are dropped). The 27s survive, so the edge is still found even if it shifts a pixel.", apply: () => setup(V, V, 1, 0, 1) },
  ];

  return (
    <LabFrame
      id="convolution"
      title="Convolution and pooling"
      subtitle="Click any output cell (or use the slider) to see the 3×3 window that produced it."
      tour={tour}
      onReset={() => { setImg("vertical edge"); setKer("vertical edge"); setStride(1); setPad(0); setPos(1); }}
      presets={[
        { label: "Kernel mismatch", apply: () => { setImg("horizontal edge"); setKer("vertical edge"); } },
        { label: "Blur", apply: () => setKer("blur") },
        { label: "Same size (pad 1)", apply: () => setPad(1) },
      ]}
      controls={
        <>
          <label className="block text-[0.8rem] text-muted">Image
            <select value={img} onChange={(e) => setImg(e.target.value)} className="mt-1 w-full rounded-lg border border-line bg-bg px-2 py-1.5 text-sm text-ink">
              {Object.keys(IMAGES).map((k) => <option key={k}>{k}</option>)}
            </select>
          </label>
          <label className="block text-[0.8rem] text-muted">Kernel
            <select value={ker} onChange={(e) => setKer(e.target.value)} className="mt-1 w-full rounded-lg border border-line bg-bg px-2 py-1.5 text-sm text-ink">
              {Object.keys(KERNELS).map((k) => <option key={k}>{k}</option>)}
            </select>
          </label>
          <Segmented label="Stride" value={String(stride)} onChange={(v) => { setStride(+v); setPos(0); }} options={[{ value: "1", label: "1" }, { value: "2", label: "2" }]} />
          <Segmented label="Padding" value={String(pad)} onChange={(v) => { setPad(+v); setPos(0); }} options={[{ value: "0", label: "0" }, { value: "1", label: "1" }]} />
          <Slider label="output position" value={p} min={0} max={n * n - 1} onChange={setPos} format={(v) => `(${Math.floor(v / n) + 1}, ${(v % n) + 1})`} />
          <div className="flex gap-2"><Button onClick={() => setPos(Math.max(0, p - 1))}>Back</Button><Button primary onClick={() => setPos((p + 1) % (n * n))}>Next cell</Button></div>
        </>
      }
      readout={
        <>
          <Stat label="output size" value={`⌊(7 − 3 + 2·${pad}) / ${stride}⌋ + 1 = ${n}`} />
          <Stat label="this cell" value={r1(Y[oi][oj])} color="var(--c-purple)" />
          <Stat label="parameters" value="9 weights + 1 bias" />
        </>
      }
      interpretation={
        <>
          <p className="text-ink font-mono text-[0.8rem] break-words">Y[{oi + 1},{oj + 1}] = {terms.map((t) => `${r1(t.w)}·${r1(t.x)}`).join(" + ")} = {r1(Y[oi][oj])}</p>
          <p className="mt-1">{ker.includes("edge") && Math.max(...Y.flat().map(Math.abs)) === 0
            ? "Every output is 0: this kernel looks for an edge direction that isn't in the image. A CNN learns many kernels so some respond to each pattern."
            : "The same 9 weights slide over every position (weight sharing). Large positive values mean the patch matches the kernel's pattern; that is what a learned filter detects."}</p>
        </>
      }
    >
      <div className="flex flex-wrap gap-6 items-start">
        <Grid m={Xp} title={pad ? "input image + zero padding" : "input image"} shape={pad ? `[7, 7] → [${7 + 2 * pad}, ${7 + 2 * pad}]` : "[7, 7]"} tone="image"
          hl={(i, j) => i >= r0 + pad && i < r0 + pad + 3 && j >= c0 + pad && j < c0 + pad + 3}
          ghost={(i, j) => i < pad || j < pad || i >= 7 + pad || j >= 7 + pad} />
        <Grid m={K} title="kernel" shape="[3, 3]" tone="signed" />
        <Grid m={Y} title="feature map" shape={`[${n}, ${n}]`} tone="signed" hl={(i, j) => i === oi && j === oj} onPick={(i, j) => setPos(i * n + j)} />
        {pooled.length > 0 && <Grid m={pooled} title="2×2 max pool" shape={`[${pooled.length}, ${pooled.length}]`} tone="signed" />}
      </div>
      <p className="text-xs text-muted mt-3">Kernel, feature map and pool: teal is positive, orange is negative, stronger colour means a larger magnitude. The purple outline is the current window and the output cell it produces.{pad > 0 && " Dashed cells are the zero padding added around the image, so the output keeps the input size."}</p>
    </LabFrame>
  );
}
