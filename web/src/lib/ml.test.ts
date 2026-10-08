import { describe, expect, it } from "vitest";
import * as ml from "./ml";

// Fixtures are the worked examples printed in the guide, so the site and the text cannot disagree.

describe("activations", () => {
  it("sigmoid", () => {
    expect(ml.sigmoid(0)).toBe(0.5);
    expect(ml.sigmoid(0.3)).toBeCloseTo(0.5744, 4); // backprop worked example, new prediction
    expect(ml.sigmoid(-50)).toBeGreaterThan(0);
  });
  it("softmax sums to 1, is stable, and respects temperature", () => {
    const p = ml.softmax([2, 1, 0.1]);
    expect(ml.sum(p)).toBeCloseTo(1, 12);
    expect(p[0]).toBeCloseTo(0.659, 3);
    expect(ml.softmax([1000, 1000])).toEqual([0.5, 0.5]);
    const cold = ml.softmax([2, 1], 0.5), hot = ml.softmax([2, 1], 2);
    expect(cold[0]).toBeGreaterThan(p[0]);
    expect(hot[0]).toBeLessThan(cold[0]);
  });
});

describe("losses", () => {
  it("MSE / MAE / RMSE / R²", () => {
    expect(ml.mse([2, 4, 5], [1, 2, 3])).toBe(3);
    expect(ml.mae([2, 4, 5], [1, 2, 3])).toBeCloseTo(5 / 3, 12);
    expect(ml.rmse([2, 4, 5], [1, 2, 3])).toBeCloseTo(Math.sqrt(3), 12);
    expect(ml.r2([1, 2, 3], [1, 2, 3])).toBe(1);
  });
  it("binary cross-entropy", () => {
    expect(ml.bce(1, 0.5)).toBeCloseTo(Math.log(2), 12);
    expect(ml.bce(0, 0.9)).toBeCloseTo(-Math.log(0.1), 12);
    expect(Number.isFinite(ml.bce(1, 0))).toBe(true); // clamped, no Infinity
  });
  it("categorical cross-entropy", () => {
    expect(ml.cce([0.7, 0.2, 0.1], 0)).toBeCloseTo(-Math.log(0.7), 12);
  });
});

describe("linear regression worked example (x=[1,2,3], y=[2,4,5], w=1, b=0, lr=0.1)", () => {
  const x = [1, 2, 3], y = [2, 4, 5];
  it("computes predictions, residuals, MSE and gradients", () => {
    const g = ml.linearGrad(x, y, 1, 0);
    expect(g.yHat).toEqual([1, 2, 3]);
    expect(g.residuals).toEqual([1, 2, 2]);
    expect(g.mse).toBe(3);
    expect(g.dw).toBeCloseTo(-22 / 3, 12);
    expect(g.db).toBeCloseTo(-10 / 3, 12);
  });
  it("takes one gradient-descent step", () => {
    const s = ml.gradientStep(x, y, 1, 0, 0.1);
    expect(s.w).toBeCloseTo(1.7333, 4);
    expect(s.b).toBeCloseTo(0.3333, 4);
    expect(ml.linearGrad(x, y, s.w, s.b).mse).toBeCloseTo(0.1096, 4);
  });
  it("closed form matches the normal equation", () => {
    const f = ml.linearFit(x, y);
    expect(f.w).toBeCloseTo(1.5, 12);
    expect(f.b).toBeCloseTo(2 / 3, 12);
  });
  it("L2 shrinks the slope, L1 can zero it", () => {
    const ols = ml.linearFit(x, y).w;
    expect(Math.abs(ml.regularizedFit(x, y, 1, "l2").w)).toBeLessThan(ols);
    expect(ml.regularizedFit(x, y, 100, "l1").w).toBe(0);
  });
});

describe("classification metrics", () => {
  const scores = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1];
  const labels = [1, 1, 0, 1, 0, 0];
  it("confusion matrix at a threshold", () => {
    expect(ml.confusionAt(scores, labels, 0.5)).toEqual({ tp: 2, fp: 1, tn: 2, fn: 1 });
  });
  it("precision, recall, F1, accuracy", () => {
    const c = ml.confusionAt(scores, labels, 0.5);
    expect(ml.precision(c)).toBeCloseTo(2 / 3, 12);
    expect(ml.recall(c)).toBeCloseTo(2 / 3, 12);
    expect(ml.f1(c)).toBeCloseTo(2 / 3, 12);
    expect(ml.accuracy(c)).toBeCloseTo(4 / 6, 12);
  });
  it("handles empty denominators without NaN", () => {
    const c = ml.confusionAt(scores, labels, 1.1); // predicts nothing positive
    expect(ml.precision(c)).toBe(0);
    expect(ml.f1(c)).toBe(0);
  });
});

describe("trees", () => {
  it("Gini impurity", () => {
    expect(ml.gini([5, 5])).toBe(0.5);
    expect(ml.gini([10, 0])).toBe(0);
    expect(ml.gini([2, 1, 1])).toBeCloseTo(0.625, 12);
  });
  it("entropy in bits", () => {
    expect(ml.entropy([5, 5])).toBe(1);
    expect(ml.entropy([4, 0])).toBe(0);
  });
  it("impurity decrease of a perfect split equals parent impurity", () => {
    expect(ml.impurityDecrease([[5, 0], [0, 5]])).toBeCloseTo(0.5, 12);
  });
});

describe("vectors", () => {
  it("dot product and matmul shapes", () => {
    expect(ml.dot([2, 1.5], [30, 120])).toBe(240); // house-price example from Module 0
    expect(ml.matmul([[1, 2], [3, 4]], [[5], [6]])).toEqual([[17], [39]]);
    expect(() => ml.matmul([[1, 2]], [[1, 2]])).toThrow();
  });
  it("cosine similarity", () => {
    expect(ml.cosineSimilarity([1, 0], [0, 1])).toBe(0);
    expect(ml.cosineSimilarity([1, 2], [2, 4])).toBeCloseTo(1, 12);
    expect(ml.cosineSimilarity([0, 0], [1, 1])).toBe(0);
  });
  it("distances and standardization", () => {
    expect(ml.euclidean([0, 0], [3, 4])).toBe(5);
    expect(ml.manhattan([0, 0], [3, 4])).toBe(7);
    const z = ml.standardize([1, 2, 3]);
    expect(ml.mean(z)).toBeCloseTo(0, 12);
    expect(ml.variance(z)).toBeCloseTo(1, 12);
  });
});

describe("TF-IDF", () => {
  it("matches tf * ln(N/df)", () => {
    const r = ml.tfidf(["machine learning is useful", "machine learning learning", "cats are useful"]);
    const j = r.vocab.indexOf("learning");
    expect(r.tf[1][j]).toBe(2);
    expect(r.df[j]).toBe(2);
    expect(r.idf[j]).toBeCloseTo(Math.log(3 / 2), 12);
    expect(r.tfidf[1][j]).toBeCloseTo(2 * Math.log(1.5), 12);
    // a term in every document gets zero weight
    const r2 = ml.tfidf(["a b", "a c"]);
    expect(r2.tfidf[0][r2.vocab.indexOf("a")]).toBe(0);
  });
});

describe("scaled dot-product attention (course example)", () => {
  it("q=[1,0], k1=[1,0], k2=[0,1], v1=[10,0], v2=[0,6]", () => {
    const r = ml.attention([[1, 0]], [[1, 0], [0, 1]], [[10, 0], [0, 6]]);
    expect(r.scores[0]).toEqual([1, 0]);
    expect(r.scaled[0][0]).toBeCloseTo(1 / Math.SQRT2, 12);
    expect(r.weights[0][0]).toBeCloseTo(0.6698, 4);
    expect(r.weights[0][1]).toBeCloseTo(0.3302, 4);
    expect(r.output[0][0]).toBeCloseTo(6.698, 3);
    expect(r.output[0][1]).toBeCloseTo(1.981, 3);
  });
  it("causal mask zeroes attention to future tokens", () => {
    const X = [[1, 0], [0, 1], [1, 1]];
    const r = ml.attention(X, X, X, true);
    expect(r.weights[0]).toEqual([1, 0, 0]);
    expect(r.weights[1][2]).toBe(0);
    r.weights.forEach((row) => expect(ml.sum(row)).toBeCloseTo(1, 12));
  });
});

describe("KNN and K-means", () => {
  const pts = [
    { x: 0, y: 0, label: 0 }, { x: 1, y: 0, label: 0 }, { x: 5, y: 5, label: 1 }, { x: 6, y: 5, label: 1 }, { x: 5, y: 6, label: 1 },
  ];
  it("KNN votes among nearest neighbors", () => {
    expect(ml.knnPredict(pts, { x: 0.5, y: 0.2 }, 1)).toBe(0);
    expect(ml.knnPredict(pts, { x: 4, y: 4 }, 3)).toBe(1);
    expect(ml.knnPredict(pts, { x: 0.5, y: 0.2 }, 5)).toBe(1); // large K: majority class wins
  });
  it("K-means assign/update decreases inertia", () => {
    let c = [{ x: 0, y: 5 }, { x: 6, y: 0 }];
    let a = ml.kmeansAssign(pts, c);
    const before = ml.kmeansInertia(pts, a, c);
    c = ml.kmeansUpdate(pts, a, c);
    a = ml.kmeansAssign(pts, c);
    expect(ml.kmeansInertia(pts, a, c)).toBeLessThanOrEqual(before);
  });
});

describe("decoding filters", () => {
  const p = [0.5, 0.3, 0.15, 0.05];
  it("top-k keeps k tokens and renormalizes", () => {
    const q = ml.topK(p, 2);
    expect(q[2]).toBe(0);
    expect(q[0]).toBeCloseTo(0.625, 12);
  });
  it("top-p keeps the smallest nucleus", () => {
    const q = ml.topP(p, 0.8);
    expect(q.filter((v) => v > 0).length).toBe(2);
    expect(ml.sum(q)).toBeCloseTo(1, 12);
  });
});

describe("seeded rng", () => {
  it("is deterministic", () => {
    const a = ml.rng(7), b = ml.rng(7);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
});

describe("polynomial regression", () => {
  it("recovers an exact quadratic", () => {
    const x = [-1, -0.5, 0, 0.5, 1], y = x.map((v) => 1 + 2 * v - 3 * v * v);
    const c = ml.polyFit(x, y, 2);
    expect(c[0]).toBeCloseTo(1, 6);
    expect(c[1]).toBeCloseTo(2, 6);
    expect(c[2]).toBeCloseTo(-3, 6);
    expect(ml.polyEval(c, 0.25)).toBeCloseTo(1 + 0.5 - 0.1875, 6);
  });
  it("degree 1 matches the closed-form line", () => {
    const x = [1, 2, 3], y = [2, 4, 5];
    const c = ml.polyFit(x, y, 1, 0);
    expect(c[1]).toBeCloseTo(1.5, 9);
    expect(c[0]).toBeCloseTo(2 / 3, 9);
  });
});
