import { describe, expect, it } from "vitest";
import { diverged, initState, step, SURFACES, type OptimizerId, type P2 } from "./optim";

const bowl = SURFACES.find((s) => s.id === "bowl")!;

function run(opt: OptimizerId, lr: number, n: number, surface = bowl) {
  let st = initState(surface.start);
  for (let i = 0; i < n; i++) st = step(opt, st, surface.grad(st.p), lr);
  return st;
}

describe("optimizers", () => {
  it("analytic gradients match finite differences", () => {
    for (const s of SURFACES) {
      const p: P2 = [0.37, -0.42], h = 1e-6;
      const g = s.grad(p);
      expect(g[0]).toBeCloseTo((s.f([p[0] + h, p[1]]) - s.f([p[0] - h, p[1]])) / (2 * h), 4);
      expect(g[1]).toBeCloseTo((s.f([p[0], p[1] + h]) - s.f([p[0], p[1] - h])) / (2 * h), 4);
    }
  });
  it("one SGD step is θ − η∇L", () => {
    const st = step("sgd", initState([2, 1]), bowl.grad([2, 1]), 0.1);
    expect(st.p[0]).toBeCloseTo(2 - 0.1 * 2, 12);
    expect(st.p[1]).toBeCloseTo(1 - 0.1 * 10, 12);
  });
  it("SGD converges on the bowl below η = 0.2 and diverges above it", () => {
    expect(bowl.f(run("sgd", 0.15, 200).p)).toBeLessThan(1e-6);
    expect(diverged(run("sgd", 0.21, 400).p, bowl.f)).toBe(true);
  });
  it("momentum reaches the bowl minimum faster than SGD at the same small learning rate", () => {
    expect(bowl.f(run("momentum", 0.02, 60).p)).toBeLessThan(bowl.f(run("sgd", 0.02, 60).p));
  });
  it("Adam's first step moves each coordinate by about η regardless of gradient scale", () => {
    const st = step("adam", initState([2, 1]), bowl.grad([2, 1]), 0.1);
    expect(2 - st.p[0]).toBeCloseTo(0.1, 6);
    expect(1 - st.p[1]).toBeCloseTo(0.1, 6);
  });
});
