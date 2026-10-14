import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import ClipSimilarityLab, { bestCaption, bestImage, clipLoss, colSoftmax, cosineMatrix, DATASETS, rowSoftmax, train } from "./ClipSimilarityLab";

afterEach(cleanup);
const W = DATASETS.worked, A = DATASETS.animals;

describe("CLIP lab math matches the lesson's worked example", () => {
  const C = cosineMatrix(W.I, W.T);
  it("cosine matrix", () => {
    expect(C[1][0]).toBeCloseTo(0.96, 6);
    expect(C[2][1]).toBeCloseTo(0.8, 6);
  });
  it("row and column softmax at tau 0.1", () => {
    expect(rowSoftmax(C, 0.1)[1][1]).toBeCloseTo(0.5538, 4);
    expect(colSoftmax(C, 0.1)[0][0]).toBeCloseTo(0.1642, 4);
    for (const r of rowSoftmax(C, 0.1)) expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });
  it("symmetric loss", () => {
    const l = clipLoss(C, 0.1);
    expect(l.i2t).toBeCloseTo(0.287, 3);
    expect(l.t2i).toBeCloseTo(0.6921, 3);
    expect(l.loss).toBeCloseTo(0.4896, 3);
  });
  it("caption 1 picks image 2: the directions disagree", () => {
    expect(bestCaption(C)).toEqual([0, 1, 2]);
    expect(bestImage(C)).toEqual([1, 1, 2]);
  });
});

describe("training", () => {
  it("fixes the cat/dog confusion and lowers the loss (matches clip_scratch.py)", () => {
    const C0 = cosineMatrix(A.I, A.T);
    expect(bestCaption(C0)[0]).toBe(1);
    expect(clipLoss(C0, 0.1).loss).toBeCloseTo(0.2167, 3);
    const t = train(A.I, A.T, 30);
    const C30 = cosineMatrix(t.imgs, t.txts);
    expect(bestCaption(C30)).toEqual([0, 1, 2, 3, 4]);
    expect(clipLoss(C30, 0.1).loss).toBeCloseTo(0.0178, 3);
  });
});

describe("lab UI", () => {
  it("renders, trains, and presets work", () => {
    render(<ClipSimilarityLab />);
    expect(screen.getByText(/Wrong best caption/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "After 30 train steps" }));
    expect(screen.getByText(/Every image picks its own caption/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Very soft (τ = 1)" }));
    expect(screen.getByText(/nearly uniform/)).toBeTruthy();
  });
});
