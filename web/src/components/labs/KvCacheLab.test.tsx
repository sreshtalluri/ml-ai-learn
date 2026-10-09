import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import KvCacheLab, { inferenceBudget, kvBytesPerToken, kvHeadsFor, MODELS } from "./KvCacheLab";

afterEach(cleanup);
const m8b = MODELS.find((m) => m.id === "8b")!;

describe("kv-cache lab math matches the lesson", () => {
  it("8B-style GQA stores 131,072 bytes per token in BF16; MHA 4x, MQA 1/8", () => {
    expect(kvBytesPerToken(32, 8, 128, 2)).toBe(131072);
    expect(kvBytesPerToken(32, kvHeadsFor(m8b, "mha"), 128, 2)).toBe(524288);
    expect(kvBytesPerToken(32, kvHeadsFor(m8b, "mqa"), 128, 2)).toBe(16384);
  });
  it("8k context x batch 16 on an 80 GB, 3.0 TB/s accelerator", () => {
    const r = inferenceBudget({ model: m8b, nKv: 8, ctx: 8192, batch: 16, weightBytes: 2, kvBytes: 2, memGB: 80, bwTBs: 3, gpus: 1 });
    expect(r.weightsGB).toBe(16);
    expect(r.kvGB).toBeCloseTo(17.18, 2);
    expect(r.totalGB).toBeCloseTo(33.18, 2);
    expect(r.fits).toBe(true);
    expect(r.maxBatch).toBe(59);
    expect(r.perSeqTps).toBeCloseTo(90.4, 1);
    expect(Math.round(r.aggTps)).toBe(1447);
  });
  it("batch 1, no context: 3.0e12 / 16e9 = 187.5 tokens/s", () => {
    const r = inferenceBudget({ model: m8b, nKv: 8, ctx: 0, batch: 1, weightBytes: 2, kvBytes: 2, memGB: 80, bwTBs: 3, gpus: 1 });
    expect(r.perSeqTps).toBe(187.5);
  });
});

describe("kv-cache lab renders", () => {
  it("has buttons and reacts to a preset", () => {
    render(<KvCacheLab />);
    expect(screen.getAllByRole("button").length).toBeGreaterThan(0);
    expect(screen.getByText(/= 17\.18 GB/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "70B on one GPU" }));
    expect(screen.getByText(/It does not fit/)).toBeTruthy();
  });
});
