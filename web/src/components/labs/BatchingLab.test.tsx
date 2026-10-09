import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import BatchingLab, { EXAMPLE, makeWorkload, metrics, simulate } from "./BatchingLab";

afterEach(cleanup);

describe("batching lab simulation matches the lesson's worked example", () => {
  it("static: finishes 2, 8, 11, 11; 16/11 tokens per step; 27.3% idle", () => {
    const { finish } = simulate(EXAMPLE, 2, "static");
    expect(finish).toEqual([2, 8, 11, 11]);
    const m = metrics(EXAMPLE, 2, finish);
    expect(m.throughput).toBeCloseTo(16 / 11, 10);
    expect(m.mean).toBe(8);
    expect(m.p95).toBe(11);
    expect(m.idle).toBeCloseTo(6 / 22, 10);
  });
  it("continuous: finishes 2, 8, 5, 8; 2 tokens per step; no idle slots", () => {
    const { finish, segments } = simulate(EXAMPLE, 2, "continuous");
    expect(finish).toEqual([2, 8, 5, 8]);
    expect(segments[0].map((g) => g.req)).toEqual([0, 2, 3]);
    const m = metrics(EXAMPLE, 2, finish);
    expect(m.throughput).toBe(2);
    expect(m.mean).toBe(5.75);
    expect(m.idle).toBe(0);
  });
  it("continuous is no worse on a random synthetic workload", () => {
    const w = makeWorkload(30, 0.1, 24, 0.8, 7);
    const st = metrics(w, 4, simulate(w, 4, "static").finish);
    const co = metrics(w, 4, simulate(w, 4, "continuous").finish);
    expect(co.mean).toBeLessThanOrEqual(st.mean);
    expect(co.makespan).toBeLessThanOrEqual(st.makespan);
  });
});

describe("batching lab renders", () => {
  it("has buttons and loads the lesson example", () => {
    render(<BatchingLab />);
    expect(screen.getAllByRole("button").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Lesson example" }));
    expect(screen.getByText(/16\/11 = 1\.45 vs 16\/8 = 2\.00/)).toBeTruthy();
  });
});
