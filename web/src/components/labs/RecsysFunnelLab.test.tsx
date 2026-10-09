import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import RecsysFunnelLab, { funnel } from "./RecsysFunnelLab";

afterEach(cleanup);

describe("recsys funnel lab", () => {
  it("renders with buttons and reacts to a preset", () => {
    render(<RecsysFunnelLab />);
    expect(screen.getAllByRole("button").length).toBeGreaterThan(0);
    expect(screen.getByText(/Fits the budget with 25.0 ms to spare/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Rank 5,000" }));
    expect(screen.getByText(/Over budget/)).toBeTruthy();
  });

  it("matches the lesson's worked example (1,000 -> 100 -> 20)", () => {
    const r = funnel({ k1: 1000, k2: 100, k3: 20, perItem: [0.005, 0.06, 0.1] });
    expect(r.stages.map((s) => s.latency)).toEqual([17, 75, 13]);
    expect(r.total).toBe(125);
    expect(r.compute).toBeCloseTo(75, 10);
    expect(r.stages[0].recall).toBeCloseTo(0.9817, 4);
    expect(r.stages[1].recall).toBeCloseTo(0.8647, 4);
    expect(r.recall).toBeCloseTo(0.734, 3);
  });
});
