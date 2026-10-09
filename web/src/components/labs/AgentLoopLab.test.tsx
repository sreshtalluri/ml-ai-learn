import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import AgentLoopLab, { BASE, buildTrace, contextPerCall, cumulativeInputTokens } from "./AgentLoopLab";

afterEach(cleanup);
const base = { toolError: false, ambiguous: false, maxSteps: 8, approval: false };

describe("agent loop lab", () => {
  it("matches the lesson's token worked example", () => {
    expect(BASE).toBe(1200);
    expect(cumulativeInputTokens(1200, 400, 6)).toBe(13_200);
    expect([1, 2, 3, 4, 5, 6].map((t) => 1200 + (t - 1) * 400).reduce((a, b) => a + b)).toBe(13_200);
  });

  it("happy path: search, two fare checks, hold, final answer", () => {
    const t = buildTrace(base);
    expect(t.map((x) => x.call?.name ?? x.kind)).toEqual(["search_flights", "get_fare_rules", "get_fare_rules", "hold_booking", "final"]);
    expect(contextPerCall(t)[0]).toBe(1200);
    expect(contextPerCall(t)[1]).toBe(1200 + 80 + 320);
  });

  it("toggles: error adds a retry, step limit stops, approval pauses, ambiguity picks the wrong tool", () => {
    expect(buildTrace({ ...base, toolError: true }).map((x) => x.kind).slice(0, 2)).toEqual(["error", "ok"]);
    const stopped = buildTrace({ ...base, maxSteps: 3 });
    expect(stopped).toHaveLength(4);
    expect(stopped[3].kind).toBe("stopped");
    expect(buildTrace({ ...base, approval: true }).at(-1)?.kind).toBe("awaiting");
    expect(buildTrace({ ...base, approval: true }, "reject").some((x) => x.kind === "declined")).toBe(true);
    const amb = buildTrace({ ...base, ambiguous: true });
    expect(amb[0].call?.name).toBe("search");
    expect(amb.at(-1)?.kind).toBe("bad-final");
  });

  it("steps through the trace", () => {
    render(<AgentLoopLab />);
    fireEvent.click(screen.getByRole("button", { name: "Step" }));
    expect(screen.getByText(/I need candidate flights first/)).toBeTruthy();
  });
});
