"use client";
import { useState } from "react";
import { bce, fmt, sigmoid } from "@/lib/ml";
import { LabFrame, Slider, Tex, type TourStep } from "./ui";

const DEFAULTS = { x1: 2, x2: 1, w1: 0.5, w2: -1, b: 0, y: 1, lr: 0.1 };

/** The nine-step backprop walkthrough for a one-neuron binary classifier, recomputed live. */
export function backpropSteps(v: typeof DEFAULTS) {
  const z = v.w1 * v.x1 + v.w2 * v.x2 + v.b;
  const p = sigmoid(z);
  const L = bce(v.y, p);
  const dz = p - v.y;
  const dw1 = dz * v.x1, dw2 = dz * v.x2, db = dz;
  const w1 = v.w1 - v.lr * dw1, w2 = v.w2 - v.lr * dw2, b = v.b - v.lr * db;
  const z2 = w1 * v.x1 + w2 * v.x2 + b;
  return { z, p, L, dz, dw1, dw2, db, w1, w2, b, z2, p2: sigmoid(z2), L2: bce(v.y, sigmoid(z2)) };
}

export default function BackpropLab() {
  const [v, setV] = useState(DEFAULTS);
  const s = backpropSteps(v);
  const set = (k: keyof typeof DEFAULTS) => (n: number) => setV({ ...v, [k]: n });
  const f = (n: number) => fmt(n, 4);
  const paren = (n: number) => (n < 0 ? `(${fmt(n, 4)})` : fmt(n, 4));

  const rows: [string, string, string][] = [
    ["1. Weighted sum", `z = w_1x_1 + w_2x_2 + b = ${paren(v.w1)}\\cdot${paren(v.x1)} + ${paren(v.w2)}\\cdot${paren(v.x2)} + ${paren(v.b)}`, `z = ${f(s.z)}`],
    ["2. Sigmoid output", `\\hat{y} = \\sigma(z) = \\frac{1}{1 + e^{-${paren(s.z)}}}`, `\\hat{y} = ${f(s.p)}`],
    ["3. Binary cross-entropy", `L = -[y\\log\\hat{y} + (1-y)\\log(1-\\hat{y})]`, `L = ${f(s.L)}`],
    ["4. Error signal", `\\frac{\\partial L}{\\partial z} = \\hat{y} - y = ${f(s.p)} - ${v.y}`, `${f(s.dz)}`],
    ["5. Gradient for w₁", `\\frac{\\partial L}{\\partial w_1} = \\frac{\\partial L}{\\partial z}\\,x_1 = ${paren(s.dz)} \\cdot ${paren(v.x1)}`, `${f(s.dw1)}`],
    ["6. Gradient for w₂", `\\frac{\\partial L}{\\partial w_2} = \\frac{\\partial L}{\\partial z}\\,x_2 = ${paren(s.dz)} \\cdot ${paren(v.x2)}`, `${f(s.dw2)}`],
    ["7. Gradient for b", `\\frac{\\partial L}{\\partial b} = \\frac{\\partial L}{\\partial z} \\cdot 1`, `${f(s.db)}`],
    ["8. Update (η = " + v.lr + ")", `w_1 = ${paren(v.w1)} - ${v.lr}\\cdot${paren(s.dw1)},\\; w_2 = ${paren(v.w2)} - ${v.lr}\\cdot${paren(s.dw2)},\\; b = ${paren(v.b)} - ${v.lr}\\cdot${paren(s.db)}`, `${f(s.w1)},\\; ${f(s.w2)},\\; ${f(s.b)}`],
    ["9. New prediction", `z' = ${paren(s.w1)}\\cdot${paren(v.x1)} + ${paren(s.w2)}\\cdot${paren(v.x2)} + ${paren(s.b)} = ${f(s.z2)}`, `\\hat{y}' = ${f(s.p2)}`],
  ];

  const better = s.L2 < s.L;

  // Guided tour: every step starts from the course example, then changes at most one value.
  type Key = keyof typeof DEFAULTS;
  const at = (o: Partial<typeof DEFAULTS> = {}) => setV({ ...DEFAULTS, ...o });
  const swing = (k: Key, amp: number, half = false) => (t: number) =>
    at({ [k]: +(DEFAULTS[k] + amp * Math.sin((half ? 1 : 2) * Math.PI * t)).toFixed(2) });
  const tour: TourStep[] = [
    { id: "forward", caption: "Rows 1 and 2 are the forward pass. Inputs 2 and 1 times weights 0.5 and −1 give z = 0, and the sigmoid turns that into a prediction ŷ = 0.5.", apply: () => at() },
    { id: "loss", caption: "Row 3 scores the guess: L = 0.693 for ŷ = 0.5 when the label is 1. Watch w₁ swing: as ŷ climbs toward 1 the loss shrinks, as ŷ falls it grows.", apply: () => at(), animate: swing("w1", 1.5), animMs: 3200 },
    { id: "chain", caption: "Row 4 starts the backward pass. The chain rule multiplies the loss's slope by the sigmoid's slope, they cancel, and what's left is ŷ − y = −0.5.", apply: () => at() },
    { id: "gradients", caption: "Each weight's gradient is that −0.5 times its own input: −1 for w₁, −0.5 for w₂ and b. Watch x₁ swing and the w₁ gradient follow it.", apply: () => at(), animate: swing("x1", 1), animMs: 3000 },
    { id: "update", caption: "Row 8 moves each weight against its gradient, scaled by η. Row 9 reruns the forward pass: ŷ rises from 0.5 to 0.574. Watch a bigger η take a bigger step.", apply: () => at(), animate: swing("lr", 0.9, true), animMs: 3000 },
    { id: "flip", caption: "Flip the label to y = 0. The forward pass is identical, but the error signal becomes +0.5, so every gradient and every update changes sign.", apply: () => at({ y: 0 }) },
    { id: "cost", caption: "One subtraction in row 4 fed all three gradients. That reuse is why one backward pass costs about as much as a forward pass, however many weights there are.", apply: () => at() },
  ];
  return (
    <LabFrame
      id="backprop"
      title="Backpropagation lab"
      subtitle="One neuron, sigmoid output, binary cross-entropy. Defaults are the course example."
      onReset={() => setV(DEFAULTS)}
      tour={tour}
      presets={[
        { label: "Course example", apply: () => setV(DEFAULTS) },
        { label: "Label y = 0", apply: () => setV({ ...v, y: 0 }) },
        { label: "Huge learning rate", apply: () => setV({ ...v, lr: 5 }) },
      ]}
      controls={
        <>
          <div className="grid grid-cols-2 gap-x-3 gap-y-3">
            <Slider label="x₁" value={v.x1} min={-3} max={3} step={0.1} onChange={set("x1")} format={(n) => fmt(n, 1)} />
            <Slider label="x₂" value={v.x2} min={-3} max={3} step={0.1} onChange={set("x2")} format={(n) => fmt(n, 1)} />
            <Slider label="w₁" value={v.w1} min={-3} max={3} step={0.05} onChange={set("w1")} format={(n) => fmt(n, 2)} />
            <Slider label="w₂" value={v.w2} min={-3} max={3} step={0.05} onChange={set("w2")} format={(n) => fmt(n, 2)} />
            <Slider label="b" value={v.b} min={-3} max={3} step={0.05} onChange={set("b")} format={(n) => fmt(n, 2)} />
            <Slider label="η" value={v.lr} min={0.01} max={5} step={0.01} onChange={set("lr")} format={(n) => fmt(n, 2)} />
          </div>
          <fieldset>
            <legend className="text-[0.8rem] text-muted mb-1">True label y</legend>
            <div className="flex gap-2">
              {[0, 1].map((y) => (
                <button key={y} type="button" aria-pressed={v.y === y} onClick={() => setV({ ...v, y })}
                  className={`flex-1 rounded-lg border px-2 py-1 text-sm ${v.y === y ? "border-accent bg-accent-soft text-accent" : "border-line text-muted"}`}>y = {y}</button>
              ))}
            </div>
          </fieldset>
        </>
      }
      interpretation={
        better
          ? `The update moved the prediction from ${f(s.p)} to ${f(s.p2)}, toward the label ${v.y}, and the loss fell from ${f(s.L)} to ${f(s.L2)}. Each weight moved in proportion to its input: x₁ = ${v.x1} is ${Math.abs(v.x1) > Math.abs(v.x2) ? "larger" : "not larger"} than x₂ = ${v.x2}, so w₁ ${Math.abs(v.x1) > Math.abs(v.x2) ? "changed more" : "did not change more"}.`
          : `The loss went up (from ${f(s.L)} to ${f(s.L2)}): the learning rate is so large that the step jumped past the minimum. Gradients give the direction; the learning rate decides how far to trust it.`
      }
    >
      <ol className="space-y-2">
        {rows.map(([label, work, result]) => (
          <li key={label} className="grid gap-x-4 @lg:grid-cols-[9.5rem_minmax(0,1fr)_auto] items-center rounded-lg border border-line px-3 py-2">
            <span className="text-sm font-medium">{label}</span>
            <span className="overflow-x-auto text-sm"><Tex>{work}</Tex></span>
            <span className="font-mono text-sm text-teal whitespace-nowrap"><Tex>{result}</Tex></span>
          </li>
        ))}
      </ol>
      <p className="text-xs text-muted mt-3">Why is step 4 so simple? <Tex>{"\\frac{\\partial L}{\\partial \\hat{y}} = \\frac{\\hat{y}-y}{\\hat{y}(1-\\hat{y})}"}</Tex> and <Tex>{"\\frac{\\partial \\hat{y}}{\\partial z} = \\hat{y}(1-\\hat{y})"}</Tex>. Multiplying them (the chain rule), the <Tex>{"\\hat{y}(1-\\hat{y})"}</Tex> terms cancel.</p>
    </LabFrame>
  );
}
