import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { LABS } from "@/lib/labs";
import { STORAGE_KEY } from "@/lib/progress";
import BackpropLab, { backpropSteps } from "./BackpropLab";
import KMeansLab from "./KMeansLab";
import { LAB_COMPONENTS } from "./registry";

beforeEach(() => window.localStorage.clear());
afterEach(cleanup);

describe("lab registry", () => {
  it("every lab in the metadata has a component, and vice versa", () => {
    expect(Object.keys(LAB_COMPONENTS).sort()).toEqual(LABS.map((l) => l.id).sort());
  });
});

describe("backprop lab", () => {
  it("reproduces the course's nine-step example", () => {
    const s = backpropSteps({ x1: 2, x2: 1, w1: 0.5, w2: -1, b: 0, y: 1, lr: 0.1 });
    expect(s.z).toBe(0);
    expect(s.p).toBe(0.5);
    expect(s.dz).toBe(-0.5);
    expect([s.dw1, s.dw2, s.db]).toEqual([-1, -0.5, -0.5]);
    expect(s.w1).toBeCloseTo(0.6, 12);
    expect(s.w2).toBeCloseTo(-0.95, 12);
    expect(s.b).toBeCloseTo(0.05, 12);
    expect(s.z2).toBeCloseTo(0.3, 12);
    expect(s.p2).toBeCloseTo(0.5744, 4);
  });

  it("recomputes when the learner changes the label, and marks the lab explored", () => {
    render(<BackpropLab />);
    expect(screen.getByText(/moved the prediction from 0.5000 to 0.5744/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "y = 0" }));
    expect(screen.getByText(/moved the prediction from 0.5000 to 0.4256/)).toBeTruthy();
    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY)!).labs.backprop).toBeTypeOf("number");
  });
});

describe("k-means lab", () => {
  it("alternates assignment and update steps", () => {
    render(<KMeansLab />);
    const assign = screen.getByRole("button", { name: "Assign" });
    const update = screen.getByRole("button", { name: "Update" }) as HTMLButtonElement;
    expect(update.disabled).toBe(true);
    fireEvent.click(assign);
    expect(screen.getByText(/Assignment step: 90 points joined/)).toBeTruthy();
    expect(update.disabled).toBe(false);
    fireEvent.click(update);
    expect(screen.getByText(/Update step: each centroid moved/)).toBeTruthy();
  });

  it("runs to convergence", () => {
    render(<KMeansLab />);
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(screen.getByText(/Converged after \d+ update steps/)).toBeTruthy();
  });
});
