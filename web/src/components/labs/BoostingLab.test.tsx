import { describe, expect, it } from "vitest";
import { boost, makeData } from "./BoostingLab";

const curve = (noise: number, lr: number) => {
  const { trainMse, testMse } = boost(makeData(40, 3, noise), makeData(200, 77, noise), lr);
  const best = testMse.indexOf(Math.min(...testMse));
  return { best, rise: testMse[200] / testMse[best] - 1, trainFalls: trainMse[200] < trainMse[best] };
};

describe("boosting lab", () => {
  it("barely overfits at the default settings (σ = 0.35, η = 0.3)", () => {
    const c = curve(0.35, 0.3);
    expect(c.best).toBeGreaterThan(150);
    expect(c.rise).toBeLessThan(0.02);
  });
  // The "Too many rounds (noisy data)" preset, Try-it 3, and the tour's too-many step rely on this.
  it("overfits clearly on noisy data with big steps (σ = 0.6, η = 1)", () => {
    const c = curve(0.6, 1);
    expect(c.best).toBe(8);
    expect(c.rise).toBeGreaterThan(0.15);
    expect(c.trainFalls).toBe(true);
  });
});
