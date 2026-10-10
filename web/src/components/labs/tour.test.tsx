import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TourPlayer, type TourStep } from "./tour";

// Fake speech engine: records what was spoken and lets the test finish an utterance.
let spoken: string[], cancels: number, current: { onend: (() => void) | null } | null;
beforeEach(() => {
  spoken = []; cancels = 0; current = null;
  vi.stubGlobal("SpeechSynthesisUtterance", class { text: string; rate = 1; onend: (() => void) | null = null; constructor(t: string) { this.text = t; } });
  Object.defineProperty(window, "speechSynthesis", {
    configurable: true,
    value: { speak: (u: { text: string; onend: (() => void) | null }) => { spoken.push(u.text); current = u; }, cancel: () => { cancels++; } },
  });
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers(); });

const steps = (apply: () => void): TourStep[] => [
  { id: "a", caption: "First step.", apply },
  { id: "b", caption: "Second step.", apply },
];

describe("Watch player", () => {
  it("pause silences narration and never replays or re-applies the step", () => {
    const apply = vi.fn();
    render(<TourPlayer tour={steps(apply)} onExit={() => {}} />);
    act(() => {});
    expect(apply).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByLabelText("Turn narration on"));
    expect(spoken).toEqual(["First step."]);

    const before = cancels;
    fireEvent.click(screen.getByLabelText("Pause"));
    expect(cancels).toBeGreaterThan(before); // audio stopped
    fireEvent.click(screen.getByLabelText("Play"));
    fireEvent.click(screen.getByLabelText("Pause"));
    expect(apply).toHaveBeenCalledTimes(1); // step not restarted by pause/play
    expect(spoken.length).toBeLessThanOrEqual(2); // at most one resume of an unfinished caption
  });

  it("does not speak a caption again after it finished", () => {
    render(<TourPlayer tour={steps(() => {})} onExit={() => {}} />);
    act(() => {});
    fireEvent.click(screen.getByLabelText("Turn narration on"));
    act(() => current?.onend?.());
    fireEvent.click(screen.getByLabelText("Pause"));
    fireEvent.click(screen.getByLabelText("Play"));
    expect(spoken).toEqual(["First step."]);
  });

  it("exiting stops narration", () => {
    const { unmount } = render(<TourPlayer tour={steps(() => {})} onExit={() => {}} />);
    fireEvent.click(screen.getByLabelText("Turn narration on"));
    const before = cancels;
    unmount();
    expect(cancels).toBeGreaterThan(before);
  });
});
