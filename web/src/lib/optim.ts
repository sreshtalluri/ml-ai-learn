// 2D loss surfaces and optimizers for the gradient-descent lab. Pure functions, tested in optim.test.ts.

export type P2 = [number, number];

export interface Surface {
  id: string;
  label: string;
  f: (p: P2) => number;
  grad: (p: P2) => P2;
  x: [number, number];
  y: [number, number];
  start: P2;
  minimum: P2;
  note: string;
}

export const SURFACES: Surface[] = [
  {
    id: "bowl",
    label: "Narrow bowl",
    f: ([x, y]) => 0.5 * (x * x + 10 * y * y),
    grad: ([x, y]) => [x, 10 * y],
    x: [-3.5, 3.5], y: [-2, 2], start: [-3, 1.6], minimum: [0, 0],
    note: "f = ½(x² + 10y²). Curvature is 10× steeper in y than x, so one learning rate is too big for y and too small for x. Plain gradient descent diverges once η > 2/10 = 0.2.",
  },
  {
    id: "rosenbrock",
    label: "Curved valley",
    f: ([x, y]) => (1 - x) ** 2 + 5 * (y - x * x) ** 2,
    grad: ([x, y]) => [-2 * (1 - x) - 20 * x * (y - x * x), 10 * (y - x * x)],
    x: [-2, 2], y: [-1, 3], start: [-1.5, 2.5], minimum: [1, 1],
    note: "A Rosenbrock-style valley: finding the valley is easy, following its curved floor to (1, 1) is slow. Momentum and Adam help.",
  },
  {
    id: "two-minima",
    label: "Two minima",
    f: ([x, y]) => (x * x - 1) ** 2 + y * y + 0.3 * x,
    grad: ([x, y]) => [4 * x * (x * x - 1) + 0.3, 2 * y],
    x: [-2, 2], y: [-1.5, 1.5], start: [1.6, 1.2], minimum: [-1.036, 0],
    note: "Non-convex: a global minimum on the left, a worse local minimum on the right. Where you start decides where you end up.",
  },
];

export type OptimizerId = "sgd" | "momentum" | "adam";

export interface OptState { p: P2; v: P2; m: P2; s: P2; t: number }

export const initState = (p: P2): OptState => ({ p, v: [0, 0], m: [0, 0], s: [0, 0], t: 0 });

/** One update. `g` is the (possibly noisy) gradient at state.p. */
export function step(opt: OptimizerId, st: OptState, g: P2, lr: number, beta = 0.9, beta2 = 0.999, eps = 1e-8): OptState {
  const t = st.t + 1;
  if (opt === "sgd") {
    return { ...st, t, p: [st.p[0] - lr * g[0], st.p[1] - lr * g[1]] };
  }
  if (opt === "momentum") {
    // v ← βv + g ;  θ ← θ − ηv
    const v: P2 = [beta * st.v[0] + g[0], beta * st.v[1] + g[1]];
    return { ...st, t, v, p: [st.p[0] - lr * v[0], st.p[1] - lr * v[1]] };
  }
  // Adam: bias-corrected first and second moment estimates, per-parameter step size.
  const m: P2 = [beta * st.m[0] + (1 - beta) * g[0], beta * st.m[1] + (1 - beta) * g[1]];
  const s: P2 = [beta2 * st.s[0] + (1 - beta2) * g[0] ** 2, beta2 * st.s[1] + (1 - beta2) * g[1] ** 2];
  const mh: P2 = [m[0] / (1 - beta ** t), m[1] / (1 - beta ** t)];
  const sh: P2 = [s[0] / (1 - beta2 ** t), s[1] / (1 - beta2 ** t)];
  return { ...st, t, m, s, p: [st.p[0] - (lr * mh[0]) / (Math.sqrt(sh[0]) + eps), st.p[1] - (lr * mh[1]) / (Math.sqrt(sh[1]) + eps)] };
}

export const diverged = (p: P2, f: (p: P2) => number) => !Number.isFinite(f(p)) || Math.abs(p[0]) > 1e4 || Math.abs(p[1]) > 1e4;
