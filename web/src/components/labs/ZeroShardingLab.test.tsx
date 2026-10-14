import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PRESETS } from "./LoraLab";
import ZeroShardingLab, { commBytes, minGpus, perGpu, total } from "./ZeroShardingLab";

afterEach(cleanup);
const c7 = PRESETS["7b"], c70 = PRESETS["70b"];
const gb = (s: Parameters<typeof perGpu>[1], n: number, c = c7) => total(perGpu(c, s, n, "mixed")) / 1e9;

describe("ZeRO lab math matches the lesson and zero_sharding.py", () => {
  it("one GPU: 16 bytes/param plus checkpointed activations = fine-tuning lesson's 108.63 GB", () => {
    for (const s of ["ddp", "zero1", "zero2", "zero3"] as const) expect(gb(s, 1)).toBeCloseTo(108.63, 2);
  });
  it("8 GPUs: DDP 108.63, ZeRO-1 37.88, ZeRO-2 26.09, ZeRO-3 14.30 GB", () => {
    expect(gb("ddp", 8)).toBeCloseTo(108.63, 2);
    expect(gb("zero1", 8)).toBeCloseTo(37.88, 2);
    expect(gb("zero2", 8)).toBeCloseTo(26.09, 2);
    expect(gb("zero3", 8)).toBeCloseTo(14.3, 2);
  });
  it("communication: ZeRO-1/2 equal DDP, ZeRO-3 is 1.5x; one GPU sends nothing", () => {
    const d = commBytes(c7, "ddp", 8, "mixed");
    expect(d / 1e9).toBeCloseTo(23.58, 2);
    expect(commBytes(c7, "zero1", 8, "mixed") / d).toBeCloseTo(1, 10);
    expect(commBytes(c7, "zero2", 8, "mixed") / d).toBeCloseTo(1, 10);
    expect(commBytes(c7, "zero3", 8, "mixed") / d).toBeCloseTo(1.5, 10);
    expect(commBytes(c7, "ddp", 1, "mixed")).toBe(0);
  });
  it("70B-style under ZeRO-3 first fits 80 GB at 16 GPUs", () => {
    expect(minGpus(c70, "zero3", "mixed", 80)).toBe(16);
    expect(gb("zero3", 16, c70)).toBeCloseTo(72.23, 2);
    expect(minGpus(c70, "ddp", "mixed", 80)).toBeNull();
  });
  it("precision changes bytes per parameter", () => {
    const P = 6_738_149_376;
    expect(perGpu(c7, "ddp", 1, "fp32").optimizer).toBe(8 * P);
    expect(perGpu(c7, "ddp", 1, "adam8").optimizer).toBe(6 * P);
  });
});

describe("ZeRO lab renders", () => {
  it("shows the default and reacts to a preset", () => {
    render(<ZeroShardingLab />);
    expect(screen.getAllByText(/37\.88 GB/).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "70B with FSDP" }));
    expect(screen.getAllByText(/72\.23 GB/).length).toBeGreaterThan(0);
  });
});
