import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import LoraLab, { loraParams, memory, paramCounts, PRESETS } from "./LoraLab";

afterEach(cleanup);

const c = PRESETS["7b"];
const opts = { tokens: 2048, checkpointing: true, optim: "adamw" as const };
const gb = (m: Record<string, number>) => Object.values(m).reduce((a, b) => a + b, 0) / 1e9;

describe("LoRA lab math matches the lesson's worked example", () => {
  it("counts parameters", () => {
    expect(paramCounts(c)).toEqual({ block: 6_476_005_376, embed: 262_144_000, total: 6_738_149_376 });
    expect(loraParams(c, 16, "all")).toBe(39_976_960);
    expect(loraParams(c, 8, "attn")).toBe(8_388_608);
  });
  it("computes memory per method", () => {
    const T = loraParams(c, 16, "all");
    expect(gb(memory(c, "full", T, opts))).toBeCloseTo(108.63, 2);
    expect(gb(memory(c, "lora", T, opts))).toBeCloseTo(14.94, 2);
    expect(gb(memory(c, "qlora", T, opts))).toBeCloseTo(5.33, 2);
  });
  it("renders and responds to presets", () => {
    render(<LoraLab />);
    expect(screen.getAllByText(/39\.98 M/).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "70B on 48 GB" }));
    expect(screen.getByText(/On a 48 GB GPU/)).toBeTruthy();
  });
});
