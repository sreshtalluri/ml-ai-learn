import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import AbTestLab, { normalCdf, normalQuantile, powerAt, sampleSize, simulateAA } from "./AbTestLab";

afterEach(cleanup);

describe("a/b test lab", () => {
  it("renders with buttons", () => {
    render(<AbTestLab />);
    expect(screen.getAllByRole("button").length).toBeGreaterThan(0);
    expect(screen.getByText(/you need 14,749 users per arm: 10 days/)).toBeTruthy();
  });

  it("normal helpers match tables", () => {
    expect(normalQuantile(0.975)).toBeCloseTo(1.959964, 5);
    expect(normalQuantile(0.8)).toBeCloseTo(0.841621, 5);
    expect(normalQuantile(0.01)).toBeCloseTo(-2.326348, 5);
    expect(normalCdf(1.959964)).toBeCloseTo(0.975, 6);
    expect(normalCdf(-1)).toBeCloseTo(0.158655, 6);
  });

  it("reproduces the lesson's sample size (10% baseline, 10% relative MDE, alpha 0.05, power 0.8)", () => {
    expect(sampleSize(0.1, 0.1, 0.05, 0.8)).toBe(14749);
    expect(powerAt(0.1, 0.1, 14749)).toBeCloseTo(0.8, 3);
  });

  it("peeking inflates the A/A false-positive rate; one final look does not", () => {
    const a = simulateAA({ p: 0.1, perArmPerDay: 1500, days: 14, sims: 2000, seed: 7 });
    const b = simulateAA({ p: 0.1, perArmPerDay: 1500, days: 14, sims: 2000, seed: 7 });
    expect(a).toEqual(b);
    expect(a.fprFinal).toBeGreaterThan(0.03);
    expect(a.fprFinal).toBeLessThan(0.07);
    expect(a.fprPeek).toBeGreaterThan(0.15);
    expect(a.curve[0]).toBeCloseTo(a.fprFinal, 1);
  });
});
