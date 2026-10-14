import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import MoeRoutingLab, { capacityFor, paramCounts, route, routerLogits, TOKENS } from "./MoeRoutingLab";

afterEach(cleanup);
const T = TOKENS.length;

describe("moe routing math", () => {
  it("capacity = ceil(cf · T · k / E)", () => {
    expect(capacityFor(12, 2, 8, 1)).toBe(3);
    expect(capacityFor(12, 2, 8, 1.25)).toBe(4); // 3.75 → 4
    expect(capacityFor(12, 1, 4, 0.5)).toBe(2); // 1.5 → 2
  });
  it("each token gets k experts, gates sum to 1, slots add up", () => {
    for (const [E, k] of [[4, 1], [8, 2], [6, 3]]) {
      const r = route(routerLogits(T, E), k, 1, false);
      r.slots.forEach((row) => {
        expect(row.length).toBe(k);
        expect(new Set(row.map((s) => s.expert)).size).toBe(k);
        expect(row.reduce((a, s) => a + s.gate, 0)).toBeCloseTo(1, 10);
      });
      expect(r.demand.reduce((a, b) => a + b, 0)).toBe(T * k);
      expect(r.load.reduce((a, b) => a + b, 0) + r.dropped).toBe(T * k);
      expect(Math.max(...r.load)).toBeLessThanOrEqual(r.capacity);
    }
  });
  it("balancing lowers the busiest expert's load and the drops for the default seed", () => {
    const off = route(routerLogits(T, 8), 2, 1, false);
    const on = route(routerLogits(T, 8), 2, 1, true);
    expect(on.maxDemand).toBeLessThan(off.maxDemand);
    expect(on.dropped).toBeLessThan(off.dropped);
  });
  it("synthetic params: 4B shared + 2B per expert", () => {
    expect(paramCounts(8, 2)).toEqual({ total: 20, active: 8 });
  });
  it("renders and responds to presets", () => {
    render(<MoeRoutingLab />);
    expect(screen.getAllByText("20B").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Roomy (cf 2)" }));
    expect(screen.getByText("2 of 24")).toBeTruthy();
  });
});
