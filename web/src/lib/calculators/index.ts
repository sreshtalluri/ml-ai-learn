import * as ml from "../ml";
import { type Calculator, n, num, vec } from "./types";
import { MORE_CALCULATORS } from "./more";

export type { Calculator } from "./types";

const dotProduct: Calculator = {
  id: "dot-product",
  title: "Dot product",
  group: "Linear algebra",
  lesson: "vectors-and-matrices",
  formula: "a \\cdot b = \\sum_{i=1}^{n} a_i b_i",
  symbols: [["a, b", "vectors of the same length n"], ["a_i", "the i-th entry of a"], ["n", "number of entries"]],
  fields: [
    { key: "a", label: "a", kind: "vector", default: "2, 1.5", hint: "Module 0 example: 2 bedrooms, 1.5 thousand sq ft" },
    { key: "b", label: "b", kind: "vector", default: "30, 120", hint: "weights: $k per bedroom, $k per thousand sq ft" },
  ],
  compute(v) {
    const a = vec(v.a, "a"), b = vec(v.b, "b");
    if (a.length !== b.length) throw new Error(`a has ${a.length} entries but b has ${b.length}. A dot product needs equal lengths.`);
    const prods = a.map((x, i) => x * b[i]);
    const d = ml.sum(prods);
    return {
      steps: [
        { label: "Multiply matching entries", tex: a.map((x, i) => `${n(x, true)} \\times ${n(b[i], true)}`).join(" + ") },
        { label: "Evaluate each product", tex: prods.map((p) => n(p, true)).join(" + ") },
        { label: "Add them up", tex: `= ${n(d)}` },
      ],
      result: `a \\cdot b = ${n(d)}`,
      meaning: `A weighted sum: each entry of a is scaled by the matching weight in b. Geometrically it equals ‖a‖‖b‖cos θ = ${n(ml.norm(a))} × ${n(ml.norm(b))} × cos θ, so its sign tells you whether the vectors point the same way.`,
    };
  },
  mistakes: [
    "Mixing up the dot product (one number) with element-wise multiplication (a vector).",
    "Forgetting that lengths must match. In code, a shape error here usually means a transposed matrix upstream.",
  ],
};

const sigmoid: Calculator = {
  id: "sigmoid",
  title: "Sigmoid",
  group: "Activations",
  lesson: "logistic-regression",
  formula: "\\sigma(z) = \\frac{1}{1 + e^{-z}}",
  symbols: [["z", "a logit: any real number"], ["e", "Euler's number, about 2.71828"], ["\\sigma(z)", "a probability between 0 and 1"]],
  fields: [{ key: "z", label: "z (logit)", kind: "number", default: "0.3", hint: "Backprop example: new z = 0.3" }],
  compute(v) {
    const z = num(v.z, "z");
    const e = Math.exp(-z);
    const s = 1 / (1 + e);
    return {
      steps: [
        { label: "Negate the logit", tex: `-z = ${n(-z)}` },
        { label: "Exponentiate", tex: `e^{${n(-z)}} = ${n(e)}` },
        { label: "Add one", tex: `1 + ${n(e)} = ${n(1 + e)}` },
        { label: "Take the reciprocal", tex: `\\frac{1}{${n(1 + e)}} = ${n(s)}` },
      ],
      result: `\\sigma(${n(z)}) = ${n(s)}`,
      meaning: `A logit of ${n(z)} becomes a probability of ${n(s)}. z = 0 maps to exactly 0.5; each unit of z multiplies the odds p/(1−p) by e ≈ 2.718.`,
    };
  },
  mistakes: [
    "Reading σ(z) as a calibrated probability without checking calibration on held-out data.",
    "Computing log(σ(z)) directly for very negative z. It underflows; libraries use log-sum-exp tricks (e.g. BCEWithLogitsLoss).",
  ],
};

const softmax: Calculator = {
  id: "softmax",
  title: "Softmax (with temperature)",
  group: "Activations",
  lesson: "decoding",
  formula: "p_i = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}",
  symbols: [["z_i", "logit (raw score) for class or token i"], ["T", "temperature, T > 0 (T = 1 is standard softmax)"], ["p_i", "probability of i; all p_i sum to 1"]],
  fields: [
    { key: "z", label: "logits z", kind: "vector", default: "2, 1, 0.1" },
    { key: "t", label: "temperature T", kind: "number", default: "1", hint: "Below 1 sharpens, above 1 flattens" },
  ],
  compute(v) {
    const z = vec(v.z, "z"), T = num(v.t, "T");
    if (T <= 0) throw new Error("Temperature must be greater than 0.");
    const scaled = z.map((x) => x / T);
    const m = Math.max(...scaled);
    const ex = scaled.map((x) => Math.exp(x - m));
    const s = ml.sum(ex);
    const p = ex.map((x) => x / s);
    return {
      steps: [
        { label: "Divide by temperature", tex: `z / T = [${scaled.map((x) => n(x)).join(", ")}]` },
        { label: "Subtract the max for numerical stability (does not change the result)", tex: `[${scaled.map((x) => n(x - m)).join(", ")}]` },
        { label: "Exponentiate", tex: `[${ex.map((x) => n(x)).join(", ")}]` },
        { label: "Sum", tex: `\\sum = ${n(s)}` },
        { label: "Divide each by the sum", tex: `p = [${p.map((x) => n(x)).join(", ")}]` },
      ],
      result: `p = [${p.map((x) => n(x)).join(", ")}]`,
      meaning: `The largest logit gets ${n(Math.max(...p) * 100)}% of the probability. Only differences between logits matter: adding a constant to every z leaves p unchanged.`,
    };
  },
  mistakes: [
    "Exponentiating large logits directly. e^1000 overflows; subtract the max first.",
    "Treating softmax probabilities from an LLM as factual confidence. They describe the next token, not the truth of a claim.",
  ],
};

export const CALCULATORS: Calculator[] = [dotProduct, sigmoid, softmax, ...MORE_CALCULATORS];
