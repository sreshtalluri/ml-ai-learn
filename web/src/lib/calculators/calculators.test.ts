import { describe, expect, it } from "vitest";
import { CALCULATORS } from "./index";

const defaults = (id: string) => {
  const c = CALCULATORS.find((x) => x.id === id)!;
  return { c, v: Object.fromEntries(c.fields.map((f) => [f.key, f.default])) };
};

describe("Math Lab calculators", () => {
  it("cover the spec's list with unique ids and lessons", () => {
    const ids = CALCULATORS.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.length).toBeGreaterThanOrEqual(18);
  });

  it("every calculator computes its default example without errors", () => {
    for (const c of CALCULATORS) {
      const out = c.compute(Object.fromEntries(c.fields.map((f) => [f.key, f.default])));
      expect(out.steps.length, c.id).toBeGreaterThan(0);
      expect(out.result, c.id).toBeTruthy();
      expect(out.meaning.length, c.id).toBeGreaterThan(10);
      expect(c.mistakes.length, c.id).toBeGreaterThan(0);
    }
  });

  it("reproduces the course's worked examples", () => {
    const att = defaults("attention");
    expect(att.c.compute(att.v).result).toContain("6.6976");
    const gd = defaults("gradient-update");
    expect(gd.c.compute(gd.v).result).toContain("w = 1.7333");
    const dot = defaults("dot-product");
    expect(dot.c.compute(dot.v).result).toContain("240");
    const tf = defaults("tf-idf");
    expect(tf.c.compute(tf.v).result).toContain("0.4055");
  });

  it("reports readable errors for bad input", () => {
    const dot = defaults("dot-product");
    expect(() => dot.c.compute({ a: "1, 2", b: "1" })).toThrow(/equal lengths/);
    expect(() => dot.c.compute({ a: "1, x", b: "1, 2" })).toThrow(/must be a number/);
    const sm = defaults("softmax");
    expect(() => sm.c.compute({ z: "1, 2", t: "0" })).toThrow(/greater than 0/);
  });
});
