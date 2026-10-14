"use client";
import { useMemo, useState } from "react";
import { fmt } from "@/lib/ml";
import { Button, LabFrame, Segmented, Slider, Stat, type TourStep } from "./ui";

// ---------- pure logic (exported for tests; same math as guide/code/25-computer-vision/clip_scratch.py) ----------
export type Mat = number[][];

export const normalize = (v: number[]) => {
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
};

/** C[i][j] = cos(image i, caption j). */
export const cosineMatrix = (imgs: Mat, txts: Mat): Mat => {
  const I = imgs.map(normalize), T = txts.map(normalize);
  return I.map((a) => T.map((b) => a.reduce((s, x, k) => s + x * b[k], 0)));
};

const softmax = (z: number[]) => {
  const m = Math.max(...z);
  const e = z.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
};

/** Image -> text: softmax across each row of C / tau. */
export const rowSoftmax = (C: Mat, tau: number): Mat => C.map((r) => softmax(r.map((x) => x / tau)));

/** Text -> image: softmax down each column of C / tau (returned in the same [image][caption] layout). */
export const colSoftmax = (C: Mat, tau: number): Mat => {
  const cols = C[0].map((_, j) => softmax(C.map((r) => r[j] / tau)));
  return C.map((_, i) => cols.map((col) => col[i]));
};

/** Symmetric InfoNCE: the matching pair is always the diagonal. */
export const clipLoss = (C: Mat, tau: number) => {
  const P = rowSoftmax(C, tau), Q = colSoftmax(C, tau), n = C.length;
  const i2t = -C.reduce((s, _, i) => s + Math.log(P[i][i]), 0) / n;
  const t2i = -C.reduce((s, _, i) => s + Math.log(Q[i][i]), 0) / n;
  return { P, Q, i2t, t2i, loss: (i2t + t2i) / 2 };
};

const argmax = (v: number[]) => v.reduce((b, x, i) => (x > v[b] ? i : b), 0);
/** Best caption for each image (zero-shot classification). */
export const bestCaption = (C: Mat) => C.map(argmax);
/** Best image for each caption. */
export const bestImage = (C: Mat) => C[0].map((_, j) => argmax(C.map((r) => r[j])));

/** One gradient step of the symmetric loss on the unit embeddings, then re-normalise. */
export const trainStep = (imgs: Mat, txts: Mat, tau = 0.1, lr = 0.05) => {
  const I = imgs.map(normalize), T = txts.map(normalize), n = I.length;
  const { P, Q } = clipLoss(cosineMatrix(I, T), tau);
  const dS = P.map((r, i) => r.map((p, j) => (p + Q[i][j] - 2 * (i === j ? 1 : 0)) / (2 * n)));
  const dI = I.map((_, i) => I[0].map((__, k) => dS[i].reduce((s, g, j) => s + g * T[j][k], 0) / tau));
  const dT = T.map((_, j) => T[0].map((__, k) => dS.reduce((s, row, i) => s + row[j] * I[i][k], 0) / tau));
  return {
    imgs: I.map((v, i) => normalize(v.map((x, k) => x - lr * dI[i][k]))),
    txts: T.map((v, j) => normalize(v.map((x, k) => x - lr * dT[j][k]))),
  };
};

export const train = (imgs: Mat, txts: Mat, steps: number) => {
  let s = { imgs, txts };
  for (let i = 0; i < steps; i++) s = trainStep(s.imgs, s.txts);
  return s;
};

// ---------- data: SYNTHETIC embeddings, hand-made so you can read them ----------
// feature axes: [animal, feline, vehicle, food, plant]
export const DATASETS = {
  animals: {
    images: ["🐱", "🐶", "🚗", "🍕", "🌳"],
    captions: ["a photo of a cat", "a photo of a dog", "a red car on a street", "a slice of pizza", "a tree in a park"],
    I: [[0.8, 0.2, 0, 0, 0.1], [0.9, 0.1, 0, 0.1, 0.1], [0, 0, 1, 0, 0.2], [0.1, 0, 0, 1, 0.1], [0, 0, 0.2, 0.1, 1]],
    T: [[0.7, 0.5, 0, 0, 0], [0.9, 0, 0, 0.1, 0.2], [0, 0, 0.9, 0.1, 0.3], [0, 0, 0.1, 0.9, 0], [0.1, 0, 0, 0, 0.9]],
  },
  worked: {
    images: ["img 1", "img 2", "img 3"],
    captions: ["caption 1", "caption 2", "caption 3"],
    I: [[1, 0], [0.6, 0.8], [0, 1]],
    T: [[0.8, 0.6], [0.6, 0.8], [0, 1]],
  },
};
type DataId = keyof typeof DATASETS;
type View = "cos" | "i2t" | "t2i";
const TRAIN_TAU = 0.1;

export default function ClipSimilarityLab() {
  const [data, setData] = useState<DataId>("animals");
  const [tau, setTau] = useState(0.1);
  const [steps, setSteps] = useState(0);
  const [view, setView] = useState<View>("i2t");
  const [pick, setPick] = useState(0);

  const d = DATASETS[data];
  const n = d.images.length;
  const emb = useMemo(() => train(d.I, d.T, steps), [d, steps]);
  const C = useMemo(() => cosineMatrix(emb.imgs, emb.txts), [emb]);
  const { P, Q, i2t, t2i, loss } = clipLoss(C, tau);
  const shown = view === "cos" ? C : view === "i2t" ? P : Q;
  const bestT = bestCaption(C), bestI = bestImage(C);
  const accI = bestT.filter((j, i) => j === i).length, accT = bestI.filter((i, j) => i === j).length;
  const chance = Math.log(n);
  const p = Math.min(pick, n - 1);
  const ranked = P[p].map((v, j) => ({ j, v })).sort((a, b) => b.v - a.v);

  const setup = (id: DataId, t: number, v: View, s = 0, pk = 0) => { setData(id); setTau(t); setView(v); setSteps(s); setPick(pk); };

  // Tour: captions computed from the real numbers, so they always match the screen.
  const A = DATASETS.animals;
  const C0 = cosineMatrix(A.I, A.T);
  const L0 = clipLoss(C0, 0.1);
  const A30 = train(A.I, A.T, 30);
  const C30 = cosineMatrix(A30.imgs, A30.txts);
  const L30 = clipLoss(C30, 0.1);
  const flat = rowSoftmax(C0, 1)[0], sharp = rowSoftmax(C0, 0.1)[0];
  const tour: TourStep[] = [
    { id: "similarity", caption: `Rows are images, columns are captions, and every cell is a cosine similarity. The matching pairs sit on the diagonal. The cat image scores ${fmt(C0[0][0], 2)} with "a photo of a cat" but ${fmt(C0[0][1], 2)} with "a photo of a dog".`, apply: () => setup("animals", 0.1, "cos") },
    { id: "temperature", caption: `Before the softmax every cosine is divided by the temperature τ. At τ = 1 the cat row is nearly flat (top probability ${fmt(Math.max(...flat), 2)}); at τ = 0.1 it is decisive (${fmt(Math.max(...sharp), 2)}). Small τ turns small cosine gaps into big probability gaps.`, apply: () => setup("animals", 1, "i2t"), animate: (t) => setTau(Math.round((1 - t * 0.95) * 100) / 100), animMs: 2600 },
    { id: "image-to-text", caption: `The row softmax asks: which caption goes with this image? The cat image puts ${fmt(L0.P[0][1], 2)} on the dog caption and only ${fmt(L0.P[0][0], 2)} on its own, so ${bestCaption(C0).filter((j, i) => j === i).length} of 5 images pick their own caption.`, apply: () => setup("animals", 0.1, "i2t") },
    { id: "text-to-image", caption: `The column softmax asks the reverse: which image goes with this caption? "A photo of a cat" still picks the cat image (${fmt(L0.Q[0][0], 2)}). The two directions disagree, which is why CLIP trains on both.`, apply: () => setup("animals", 0.1, "t2i") },
    { id: "loss", caption: `The loss is minus the log of the diagonal probabilities, averaged over rows (${fmt(L0.i2t, 3)}) and over columns (${fmt(L0.t2i, 3)}). Their mean, ${fmt(L0.loss, 3)}, is the symmetric contrastive loss. Random guessing would score ln 5 = ${fmt(Math.log(5), 3)}.`, apply: () => setup("animals", 0.1, "i2t") },
    { id: "train", caption: `Each train step moves every image toward its own caption and away from the others, and every caption the same way. After 30 steps the loss falls from ${fmt(L0.loss, 3)} to ${fmt(L30.loss, 3)} and the cat image picks its own caption with ${fmt(L30.P[0][0], 2)}.`, apply: () => setup("animals", 0.1, "i2t"), animate: (t) => setSteps(Math.round(t * 30)), animMs: 3000 },
    { id: "zero-shot", caption: `Zero-shot classification is the same row: write one caption per class, embed them, and pick the caption closest to the image. No classifier is trained on these classes. After training, ${bestCaption(C30).filter((j, i) => j === i).length} of 5 images are classified correctly.`, apply: () => setup("animals", 0.1, "i2t", 30, 0) },
  ];

  const viewNote = view === "cos" ? "cosine similarity (before temperature)" : view === "i2t" ? "each row sums to 1: image → text" : "each column sums to 1: text → image";
  const wrong = bestT.map((j, i) => (j !== i ? `${d.images[i]} → "${d.captions[j]}"` : null)).filter(Boolean);

  return (
    <LabFrame
      id="clip-similarity"
      title="Contrastive image-text matching"
      tour={tour}
      subtitle="SYNTHETIC 5-D embeddings stand in for an image encoder and a text encoder. Train steps run gradient descent on the symmetric loss at τ = 0.1."
      onReset={() => setup("animals", 0.1, "i2t")}
      presets={[
        { label: "Cat/dog confusion", apply: () => setup("animals", 0.1, "i2t") },
        { label: "Worked example (3 pairs)", apply: () => setup("worked", 0.1, "i2t") },
        { label: "Very sharp (τ = 0.01)", apply: () => setup("animals", 0.01, "i2t") },
        { label: "Very soft (τ = 1)", apply: () => setup("animals", 1, "i2t") },
        { label: "After 30 train steps", apply: () => setup("animals", 0.1, "i2t", 30) },
      ]}
      controls={
        <>
          <Segmented label="Show" value={view} onChange={setView}
            options={[{ value: "cos", label: "cosine" }, { value: "i2t", label: "image→text" }, { value: "t2i", label: "text→image" }]} />
          <Slider label="temperature τ" value={tau} min={0.01} max={1} step={0.01} onChange={setTau} format={(v) => fmt(v, 2)}
            hint="logits = cosine / τ" />
          <div className="flex flex-wrap items-center gap-2">
            <Button primary onClick={() => setSteps((s) => Math.min(200, s + 1))}>Train 1 step</Button>
            <Button onClick={() => setSteps((s) => Math.min(200, s + 10))}>+10</Button>
          </div>
        </>
      }
      readout={
        <>
          <Stat label="train steps" value={steps} />
          <Stat label="loss image→text" value={fmt(i2t, 3)} />
          <Stat label="loss text→image" value={fmt(t2i, 3)} />
          <Stat label="symmetric loss" value={fmt(loss, 3)} color="var(--c-teal)" />
          <Stat label={`chance (ln ${n})`} value={fmt(chance, 3)} />
          <Stat label="zero-shot image→text" value={`${accI} / ${n}`} color={accI < n ? "var(--c-orange)" : undefined} />
          <Stat label="text→image correct" value={`${accT} / ${n}`} />
        </>
      }
      interpretation={
        tau <= 0.02
          ? `At τ = ${fmt(tau, 2)} the softmax is almost one-hot. A right answer costs nearly nothing, but every near-miss is punished hard: loss ${fmt(loss, 3)}. Real CLIP learns τ and caps it at 0.01.`
          : tau >= 0.7
            ? `At τ = ${fmt(tau, 2)} the probabilities are nearly uniform, so even correct pairs score little and the loss (${fmt(loss, 3)}) sits close to chance (${fmt(chance, 3)}). Gradients are weak and training is slow.`
            : wrong.length
              ? `Wrong best caption: ${wrong.join(", ")}. The loss is ${fmt(loss, 3)}; press Train to pull each image toward its own caption and push it away from the rest.`
              : `Every image picks its own caption and every caption its own image. Loss ${fmt(loss, 3)} (chance ${fmt(chance, 3)}). More training only sharpens the diagonal.`
      }
    >
      <p className="mb-2 text-sm text-muted">{viewNote}. Diagonal = matching pairs (outlined); dot = best match in that {view === "t2i" ? "column" : "row"}.</p>
      <div className="overflow-x-auto">
        <div className="grid gap-1 text-[0.72rem]" style={{ gridTemplateColumns: `5.5rem repeat(${n}, minmax(3.2rem, 1fr))` }}
          role="table" aria-label={`${n} by ${n} matrix: ${viewNote}`}>
          <div />
          {d.captions.map((c) => <div key={c} className="px-1 pb-1 text-center leading-tight text-muted" role="columnheader">&ldquo;{c}&rdquo;</div>)}
          {shown.map((row, i) => (
            <div key={i} className="contents" role="row">
              <button type="button" onClick={() => setPick(i)} aria-pressed={p === i}
                className={`flex items-center justify-center rounded-lg border px-1 py-2 text-base ${p === i ? "border-accent bg-accent-soft" : "border-line bg-surface-2"}`}>
                {d.images[i]}
              </button>
              {row.map((v, j) => {
                const best = view === "t2i" ? bestI[j] === i : bestT[i] === j;
                return (
                  <div key={j} role="cell"
                    className={`relative flex items-center justify-center rounded font-mono ${i === j ? "outline outline-2 -outline-offset-2 outline-[var(--c-teal)]" : ""}`}
                    style={{ background: `color-mix(in srgb, var(--c-blue) ${Math.round(Math.max(0, Math.min(1, v)) * 75)}%, var(--surface))`, color: v > 0.55 ? "white" : undefined }}>
                    {fmt(v, 2)}
                    {view !== "cos" && best && <span className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full" style={{ background: i === j ? "var(--c-teal)" : "var(--c-orange)" }} />}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-5">
        <p className="mb-1 text-sm font-medium">Zero-shot: rank the captions for {d.images[p]} (click an image to change)</p>
        <div className="space-y-1">
          {ranked.map(({ j, v }) => (
            <div key={j} className="flex items-center gap-2 text-xs">
              <span className="w-40 shrink-0 truncate text-muted">&ldquo;{d.captions[j]}&rdquo;</span>
              <div className="h-3 flex-1 overflow-hidden rounded bg-surface-2">
                <div className="h-full" style={{ width: `${(v * 100).toFixed(2)}%`, background: j === p ? "var(--c-teal)" : "var(--c-orange)" }} />
              </div>
              <span className="w-12 text-right font-mono">{fmt(v, 3)}</span>
            </div>
          ))}
        </div>
        <p className="mt-1 text-xs text-faint">Teal = the correct caption. Probabilities are this image&apos;s row softmax at τ = {fmt(tau, 2)}; training uses τ = {TRAIN_TAU}.</p>
      </div>
    </LabFrame>
  );
}
