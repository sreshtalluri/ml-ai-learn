import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import AttentionMasksLab, { attendedCount, maskMatrix, receptiveField, type MaskKind } from "./AttentionMasksLab";

afterEach(cleanup);
const o = { w: 4, stride: 4 };
const count = (k: MaskKind, n = 16, opts = o) => attendedCount(maskMatrix(k, n, opts));
const sees = (k: MaskKind) => maskMatrix(k, 16, o)[12].filter(Boolean).length;

describe("attention mask math matches the lesson script", () => {
  it("cell counts at n=16", () => {
    expect(count("full")).toBe(256);
    expect(count("causal")).toBe(136);
    expect(count("sliding")).toBe(58);
    expect(count("strided")).toBe(52);
    expect(count("global")).toBe(100);
    expect(count("sliding", 8)).toBe(26); // 1+2+3+4·5
    expect(count("sliding", 16, { w: 16, stride: 4 })).toBe(136); // w ≥ n recovers causal
  });
  it("query 12 sees", () => {
    expect([sees("causal"), sees("sliding"), sees("strided"), sees("global")]).toEqual([13, 4, 4, 6]);
  });
  it("receptive field grows by w-1 per layer", () => {
    expect([1, 2, 3, 4, 5].map((L) => receptiveField(L, 4, 15))).toEqual([4, 7, 10, 13, 16]);
    expect(receptiveField(6, 4, 15)).toBe(16);
  });
  it("renders and responds to presets", () => {
    render(<AttentionMasksLab />);
    expect(screen.getByText("136 / 256")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Sliding window 4" }));
    expect(screen.getByText("58 / 256")).toBeTruthy();
  });
});
