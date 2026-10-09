// Pure, dependency-free ML math used by labs, the Math Lab, and tests.
// Every function is deterministic and works on plain number arrays.

export type Vec = number[];
export type Mat = number[][];

// ---------- basics ----------

export const sum = (a: Vec) => a.reduce((s, v) => s + v, 0);
export const mean = (a: Vec) => (a.length ? sum(a) / a.length : NaN);

/** Population variance (divide by n). */
export function variance(a: Vec): number {
  const m = mean(a);
  return mean(a.map((v) => (v - m) ** 2));
}

export function standardize(a: Vec): Vec {
  const m = mean(a);
  const s = Math.sqrt(variance(a));
  return a.map((v) => (s === 0 ? 0 : (v - m) / s));
}

export function dot(a: Vec, b: Vec): number {
  if (a.length !== b.length) throw new Error(`dot: length ${a.length} vs ${b.length}`);
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

export const norm = (a: Vec) => Math.sqrt(dot(a, a));

export function matmul(A: Mat, B: Mat): Mat {
  const inner = A[0]?.length ?? 0;
  if (inner !== B.length) throw new Error(`matmul: [${A.length},${inner}] x [${B.length},${B[0]?.length ?? 0}]`);
  return A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
}

export const transpose = (A: Mat): Mat => (A[0] ?? []).map((_, j) => A.map((row) => row[j]));

export const euclidean = (a: Vec, b: Vec) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
export const manhattan = (a: Vec, b: Vec) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);

export function cosineSimilarity(a: Vec, b: Vec): number {
  const d = norm(a) * norm(b);
  return d === 0 ? 0 : dot(a, b) / d;
}

// ---------- activations ----------

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
export const relu = (z: number) => Math.max(0, z);
export const tanh = (z: number) => Math.tanh(z);
/** GELU, tanh approximation (as used in GPT-2). */
export const gelu = (z: number) => 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z ** 3)));

/** Numerically stable softmax with optional temperature T (> 0). */
export function softmax(z: Vec, T = 1): Vec {
  const scaled = z.map((v) => v / T);
  const m = Math.max(...scaled);
  const e = scaled.map((v) => Math.exp(v - m));
  const s = sum(e);
  return e.map((v) => v / s);
}

// ---------- losses ----------

export const mse = (y: Vec, yHat: Vec) => mean(y.map((v, i) => (v - yHat[i]) ** 2));
export const mae = (y: Vec, yHat: Vec) => mean(y.map((v, i) => Math.abs(v - yHat[i])));
export const rmse = (y: Vec, yHat: Vec) => Math.sqrt(mse(y, yHat));

export function r2(y: Vec, yHat: Vec): number {
  const m = mean(y);
  const sse = sum(y.map((v, i) => (v - yHat[i]) ** 2));
  const sst = sum(y.map((v) => (v - m) ** 2));
  return 1 - sse / sst;
}

const EPS = 1e-12;
const clampP = (p: number) => Math.min(1 - EPS, Math.max(EPS, p));

/** Binary cross-entropy for one example (natural log). */
export const bce = (y: number, p: number) => -(y * Math.log(clampP(p)) + (1 - y) * Math.log(1 - clampP(p)));

/** Categorical cross-entropy: -log of the probability on the true class. */
export const cce = (probs: Vec, trueIndex: number) => -Math.log(clampP(probs[trueIndex]));

// ---------- linear regression ----------

export interface LinearGrad { yHat: Vec; residuals: Vec; mse: number; dw: number; db: number }

/** Forward pass + MSE gradients for y_hat = w*x + b. */
export function linearGrad(x: Vec, y: Vec, w: number, b: number): LinearGrad {
  const n = x.length;
  const yHat = x.map((v) => w * v + b);
  const err = yHat.map((v, i) => v - y[i]);
  return {
    yHat,
    residuals: y.map((v, i) => v - yHat[i]),
    mse: mean(err.map((e) => e * e)),
    dw: (2 / n) * sum(err.map((e, i) => e * x[i])),
    db: (2 / n) * sum(err),
  };
}

export function gradientStep(x: Vec, y: Vec, w: number, b: number, lr: number) {
  const g = linearGrad(x, y, w, b);
  return { w: w - lr * g.dw, b: b - lr * g.db, grad: g };
}

/** Closed-form least squares for one feature. */
export function linearFit(x: Vec, y: Vec): { w: number; b: number } {
  const mx = mean(x), my = mean(y);
  const sxy = sum(x.map((v, i) => (v - mx) * (y[i] - my)));
  const sxx = sum(x.map((v) => (v - mx) ** 2));
  const w = sxx === 0 ? 0 : sxy / sxx;
  return { w, b: my - w * mx };
}

/**
 * One-feature regularized fit by coordinate descent on standardized-free data.
 * Loss: MSE + lambda * penalty(w) (bias not penalized). Returns w, b.
 */
export function regularizedFit(x: Vec, y: Vec, lambda: number, kind: "l1" | "l2" | "none"): { w: number; b: number } {
  if (kind === "none" || lambda === 0) return linearFit(x, y);
  const mx = mean(x), my = mean(y);
  const n = x.length;
  const sxy = sum(x.map((v, i) => (v - mx) * (y[i] - my))) / n;
  const sxx = sum(x.map((v) => (v - mx) ** 2)) / n;
  let w: number;
  if (kind === "l2") {
    // d/dw [ (1/n)Σ(y - wx - b)^2 + λw^2 ] = 0  ->  w = sxy / (sxx + λ)
    w = sxy / (sxx + lambda);
  } else {
    // soft-thresholding: w = sign(sxy) * max(|sxy| - λ/2, 0) / sxx
    w = (Math.sign(sxy) * Math.max(Math.abs(sxy) - lambda / 2, 0)) / sxx;
  }
  return { w, b: my - w * mx };
}

// ---------- classification metrics ----------

export interface Confusion { tp: number; fp: number; tn: number; fn: number }

export function confusionAt(scores: Vec, labels: number[], threshold: number): Confusion {
  const c = { tp: 0, fp: 0, tn: 0, fn: 0 };
  scores.forEach((s, i) => {
    const pred = s >= threshold ? 1 : 0;
    if (pred === 1 && labels[i] === 1) c.tp++;
    else if (pred === 1) c.fp++;
    else if (labels[i] === 0) c.tn++;
    else c.fn++;
  });
  return c;
}

const safeDiv = (a: number, b: number) => (b === 0 ? 0 : a / b);
export const precision = (c: Confusion) => safeDiv(c.tp, c.tp + c.fp);
export const recall = (c: Confusion) => safeDiv(c.tp, c.tp + c.fn);
export const specificity = (c: Confusion) => safeDiv(c.tn, c.tn + c.fp);
export const accuracy = (c: Confusion) => safeDiv(c.tp + c.tn, c.tp + c.tn + c.fp + c.fn);
export function f1(c: Confusion): number {
  const p = precision(c), r = recall(c);
  return safeDiv(2 * p * r, p + r);
}

// ---------- trees ----------

/** Gini impurity from class counts: 1 - Σ p_k^2. */
export function gini(counts: number[]): number {
  const n = sum(counts);
  return n === 0 ? 0 : 1 - sum(counts.map((c) => (c / n) ** 2));
}

/** Shannon entropy (bits) from class counts. */
export function entropy(counts: number[]): number {
  const n = sum(counts);
  return n === 0 ? 0 : sum(counts.filter((c) => c > 0).map((c) => (c / n) * Math.log2(n / c)));
}

/** Weighted impurity decrease of splitting parent into children (each a counts array). */
export function impurityDecrease(children: number[][], impurity: (c: number[]) => number = gini): number {
  const parent = children[0].map((_, k) => sum(children.map((c) => c[k])));
  const n = sum(parent);
  return impurity(parent) - sum(children.map((c) => (sum(c) / n) * impurity(c)));
}

// ---------- KNN ----------

export interface LabeledPoint { x: number; y: number; label: number }

export function knnNeighbors(points: LabeledPoint[], q: { x: number; y: number }, k: number, metric: "euclidean" | "manhattan" = "euclidean") {
  const d = metric === "euclidean" ? euclidean : manhattan;
  return points
    .map((p, i) => ({ index: i, point: p, distance: d([p.x, p.y], [q.x, q.y]) }))
    .sort((a, b) => a.distance - b.distance || a.index - b.index)
    .slice(0, k);
}

/** Majority vote; ties broken by the nearest neighbor among tied classes. */
export function knnPredict(points: LabeledPoint[], q: { x: number; y: number }, k: number, metric: "euclidean" | "manhattan" = "euclidean"): number {
  const nb = knnNeighbors(points, q, k, metric);
  const votes = new Map<number, number>();
  nb.forEach((n) => votes.set(n.point.label, (votes.get(n.point.label) ?? 0) + 1));
  const best = Math.max(...votes.values());
  return nb.find((n) => votes.get(n.point.label) === best)!.point.label;
}

// ---------- K-means ----------

export type Point2 = { x: number; y: number };

export function kmeansAssign(points: Point2[], centroids: Point2[]): number[] {
  return points.map((p) => {
    let best = 0, bestD = Infinity;
    centroids.forEach((c, j) => {
      const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
      if (d < bestD) { bestD = d; best = j; }
    });
    return best;
  });
}

/** Mean of assigned points; an empty cluster keeps its old centroid. */
export function kmeansUpdate(points: Point2[], assign: number[], centroids: Point2[]): Point2[] {
  return centroids.map((c, j) => {
    const mine = points.filter((_, i) => assign[i] === j);
    return mine.length ? { x: mean(mine.map((p) => p.x)), y: mean(mine.map((p) => p.y)) } : c;
  });
}

/** K-means objective: total squared distance to assigned centroid (inertia). */
export function kmeansInertia(points: Point2[], assign: number[], centroids: Point2[]): number {
  return sum(points.map((p, i) => (p.x - centroids[assign[i]].x) ** 2 + (p.y - centroids[assign[i]].y) ** 2));
}

// ---------- NLP ----------

export const tokenize = (text: string) => text.toLowerCase().normalize("NFKC").match(/[\p{L}\p{N}']+/gu) ?? [];

export interface TfidfResult {
  vocab: string[];
  tf: number[][];        // [doc][term] raw counts
  df: number[];          // [term] number of docs containing term
  idf: number[];         // [term] ln(N / df)
  tfidf: number[][];     // [doc][term]
}

/** TF-IDF with raw counts and idf = ln(N / df), matching the course formula. */
export function tfidf(docs: string[]): TfidfResult {
  const toks = docs.map(tokenize);
  const vocab = [...new Set(toks.flat())].sort();
  const N = docs.length;
  const tf = toks.map((t) => vocab.map((w) => t.filter((x) => x === w).length));
  const df = vocab.map((_, j) => tf.filter((row) => row[j] > 0).length);
  const idf = df.map((d) => Math.log(N / d));
  return { vocab, tf, df, idf, tfidf: tf.map((row) => row.map((c, j) => c * idf[j])) };
}

// ---------- attention ----------

export interface AttentionResult {
  scores: Mat;   // QK^T            [nq, nk]
  scaled: Mat;   // / sqrt(dk)
  masked: Mat;   // causal mask applied (-Infinity above diagonal)
  weights: Mat;  // softmax rows
  output: Mat;   // weights @ V     [nq, dv]
}

/** Scaled dot-product attention, optionally with a causal mask. */
export function attention(Q: Mat, K: Mat, V: Mat, causal = false): AttentionResult {
  const dk = K[0].length;
  const scores = matmul(Q, transpose(K));
  const scaled = scores.map((r) => r.map((v) => v / Math.sqrt(dk)));
  const masked = scaled.map((r, i) => r.map((v, j) => (causal && j > i ? -Infinity : v)));
  const weights = masked.map((r) => softmax(r));
  return { scores, scaled, masked, weights, output: matmul(weights, V) };
}

// ---------- decoding ----------

/** Keep the k most likely tokens, renormalize. */
export function topK(probs: Vec, k: number): Vec {
  const keep = new Set(probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i));
  const kept = probs.map((p, i) => (keep.has(i) ? p : 0));
  const s = sum(kept);
  return kept.map((p) => p / s);
}

/** Nucleus sampling: smallest set whose cumulative probability >= p, renormalized. */
export function topP(probs: Vec, p: number): Vec {
  const order = probs.map((q, i) => [q, i]).sort((a, b) => b[0] - a[0]);
  const keep = new Set<number>();
  let cum = 0;
  for (const [q, i] of order) {
    keep.add(i);
    cum += q;
    if (cum >= p) break;
  }
  const kept = probs.map((q, i) => (keep.has(i) ? q : 0));
  const s = sum(kept);
  return kept.map((q) => q / s);
}

// ---------- deterministic randomness ----------

/** mulberry32: tiny seeded PRNG so synthetic datasets are identical on every load. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box-Muller using a seeded uniform source. */
export function gaussian(rand: () => number) {
  const u = 1 - rand(), v = rand();
  // Math.log/cos differ in the last bits between Node and browsers; rounding keeps
  // server-rendered lab SVGs identical to the client render (no hydration mismatch).
  return Math.round(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * 1e9) / 1e9;
}

/** Format a number for display: fixed decimals, trims -0. */
export function fmt(v: number, d = 3): string {
  if (!Number.isFinite(v)) return v > 0 ? "∞" : v < 0 ? "−∞" : "NaN";
  const s = v.toFixed(d);
  return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
}

// ---------- polynomial regression ----------

/** Solve A x = b by Gaussian elimination with partial pivoting (small dense systems). */
export function solve(A: Mat, b: Vec): Vec {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) x[r] = (M[r][n] - M[r].slice(r + 1, n).reduce((s, v, k) => s + v * x[r + 1 + k], 0)) / M[r][r];
  return x;
}

/** Least-squares polynomial fit (tiny ridge for numerical stability). Returns coefficients, lowest power first. */
export function polyFit(x: Vec, y: Vec, degree: number, ridge = 1e-9): Vec {
  const d = degree + 1;
  const X = x.map((v) => Array.from({ length: d }, (_, k) => v ** k));
  const XtX = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[i] * r[j], 0) + (i === j && i > 0 ? ridge : 0)));
  const Xty = Array.from({ length: d }, (_, i) => X.reduce((s, r, n) => s + r[i] * y[n], 0));
  return solve(XtX, Xty);
}

export const polyEval = (c: Vec, x: number) => c.reduce((s, v, k) => s + v * x ** k, 0);
