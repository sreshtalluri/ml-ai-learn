import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import QLearningLab, { advance, envStep, GOAL, newSim, pathMoves, PIT_REWARD, START, tdTarget, train } from "./QLearningLab";

afterEach(cleanup);
const base = { alpha: 0.2, gamma: 1, eps: 0.1 };

describe("Q-learning lab logic", () => {
  it("environment: step cost, walls, pits, goal", () => {
    expect(envStep(START, 0)).toEqual({ s2: 15, r: -1, done: false });      // up
    expect(envStep(START, 3)).toEqual({ s2: START, r: -1, done: false });   // left edge: stay put
    expect(envStep(START, 1)).toEqual({ s2: 21, r: PIT_REWARD, done: true }); // right: pit
    expect(envStep(1, 2)).toEqual({ s2: 1, r: -1, done: false });           // (0,1) down into a wall
    expect(envStep(19, 2)).toEqual({ s2: GOAL, r: -1, done: true });        // (3,4) down: goal
  });

  it("TD targets match the lesson's worked example", () => {
    const qNext = [-3, -1.5, -4, -2]; // up, right, down, left
    const tq = tdTarget("q", -1, 0.9, qNext, 2, false);
    const ts = tdTarget("sarsa", -1, 0.9, qNext, 2, false);
    expect(tq).toBeCloseTo(-2.35, 10);
    expect(ts).toBeCloseTo(-4.6, 10);
    expect(-2 + 0.5 * (tq - -2)).toBeCloseTo(-2.175, 10);
    expect(-2 + 0.5 * (ts - -2)).toBeCloseTo(-3.3, 10);
    expect(tdTarget("q", -100, 0.9, qNext, 0, true)).toBe(-100);
  });

  it("one step updates exactly one Q-value and is reproducible", () => {
    const a = advance(newSim(), { ...base, algo: "q" }), b = advance(newSim(), { ...base, algo: "q" });
    expect(a.Q).toEqual(b.Q);
    expect(a.Q.filter((v) => v !== 0)).toHaveLength(1);
    const u = a.last!;
    expect(u.next).toBeCloseTo(u.old + 0.2 * (u.target - u.old), 10);
  });

  it("cliff: Q-learning hugs the pits, SARSA keeps a row away", () => {
    for (const seed of [1, 2, 3, 7]) {
      expect(pathMoves(train({ ...base, algo: "q" }, 300, seed).Q)).toBe(6);
      expect(pathMoves(train({ ...base, algo: "sarsa" }, 300, seed).Q)).toBe(8);
    }
    const mean = (xs: number[]) => xs.reduce((x, y) => x + y, 0) / xs.length;
    const q = train({ ...base, algo: "q" }, 300), s = train({ ...base, algo: "sarsa" }, 300);
    expect(mean(s.returns.slice(-50))).toBeGreaterThan(mean(q.returns.slice(-50))); // safer while exploring
  });

  it("renders, steps, and runs a preset", () => {
    render(<QLearningLab />);
    fireEvent.click(screen.getByRole("button", { name: "Step" }));
    expect(screen.getByText(/Q ←/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "SARSA, 300 episodes" }));
    expect(screen.getByText(/SARSA's greedy path takes 8 moves/)).toBeTruthy();
  });
});
