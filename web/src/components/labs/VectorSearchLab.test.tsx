import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import VectorSearchLab, { bruteForce, buildIvf, ivfCost, ivfSearch, makePoints, recallAtK } from "./VectorSearchLab";

afterEach(cleanup);

describe("vector search lab math matches the lesson", () => {
  it("recall@10 worked example is 0.8", () => {
    expect(recallAtK([0, 1, 2, 3, 4, 5, 6, 7, 42, 99], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).toBe(0.8);
  });
  it("IVF cost for N = 1M, nlist = 1024, nprobe = 8", () => {
    expect(ivfCost(1_000_000, 1024, 8)).toBe(8836.5);
    expect(ivfCost(1_000_000, 1024, 8) * 768).toBe(6_786_432);
  });
  it("probing every cell is exact and costs nlist more than brute force", () => {
    const pts = makePoints();
    const idx = buildIvf(pts, 8);
    const q = { x: 4, y: 6 };
    const s = ivfSearch(pts, idx, q, 10, 8);
    expect(recallAtK(s.top, bruteForce(pts, q, 10))).toBe(1);
    expect(s.distances).toBe(pts.length + 8);
  });
  it("renders and presets work", () => {
    render(<VectorSearchLab />);
    fireEvent.click(screen.getByRole("button", { name: "Probe everything" }));
    expect(screen.getByText(/Probing every cell is exact/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Boundary miss" }));
    expect(screen.getByText(/missed neighbor/)).toBeTruthy();
  });
});
