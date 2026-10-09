import { describe, expect, it } from "vitest";
import { backpropSteps } from "./BackpropLab";

const base = { x1: 2, x2: 1, w1: 0.5, w2: -1, b: 0, y: 1, lr: 0.1 };

describe("backprop lab", () => {
  it("matches the lesson's worked example", () => {
    const s = backpropSteps(base);
    expect([s.p, s.dw1, s.dw2, s.db]).toEqual([0.5, -1, -0.5, -0.5]);
    expect(+s.p2.toFixed(3)).toBe(0.574);
  });
  // The lesson's Try-it 2 relies on this: for one example, sigmoid + cross-entropy can't overshoot.
  it("never increases the loss on a single example, whatever the learning rate or label", () => {
    for (const y of [0, 1]) for (const lr of [0.01, 0.1, 1, 5, 50]) for (const w1 of [-3, 0.5, 3]) {
      const s = backpropSteps({ ...base, y, lr, w1 });
      expect(s.L2, `y=${y} lr=${lr} w1=${w1}`).toBeLessThanOrEqual(s.L);
    }
  });
});
