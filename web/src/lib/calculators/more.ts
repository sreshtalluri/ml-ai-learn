import * as ml from "../ml";
import { type Calculator, mat, n, num, vec } from "./types";

const sameLength = (a: number[], b: number[], an: string, bn: string) => {
  if (a.length !== b.length) throw new Error(`${an} has ${a.length} values but ${bn} has ${b.length}.`);
};
const texRow = (v: number[]) => `[${v.map((x) => n(x)).join(", ")}]`;
const texMat = (m: number[][]) => `\\begin{bmatrix}${m.map((r) => r.map((x) => n(x)).join(" & ")).join(" \\\\ ")}\\end{bmatrix}`;

const matmul: Calculator = {
  id: "matrix-multiplication",
  title: "Matrix multiplication",
  group: "Linear algebra",
  lesson: "vectors-and-matrices",
  formula: "C_{ij} = \\sum_{k} A_{ik} B_{kj}",
  symbols: [["A", "matrix of shape [m, n]"], ["B", "matrix of shape [n, p]"], ["C", "result of shape [m, p]"]],
  fields: [
    { key: "a", label: "A (rows separated by ;)", kind: "matrix", default: "1, 2; 3, 4; 5, 6" },
    { key: "b", label: "B (rows separated by ;)", kind: "matrix", default: "1, 0, 2; 0, 1, 1" },
  ],
  compute(v) {
    const A = mat(v.a, "A"), B = mat(v.b, "B");
    if (A[0].length !== B.length) throw new Error(`Shapes [${A.length}, ${A[0].length}] and [${B.length}, ${B[0].length}] don't line up: inner dimensions must match.`);
    const C = ml.matmul(A, B);
    const c00 = A[0].map((x, k) => `${n(x, true)}\\cdot${n(B[k][0], true)}`).join(" + ");
    return {
      steps: [
        { label: "Check shapes: inner dimensions match and disappear", tex: `[${A.length}, ${A[0].length}] \\times [${B.length}, ${B[0].length}] \\to [${A.length}, ${B[0].length}]` },
        { label: "Each entry is a row of A dotted with a column of B; for example C₁₁", tex: `C_{11} = ${c00} = ${n(C[0][0])}` },
        { label: "Full result", tex: `C = ${texMat(C)}` },
      ],
      result: `C = ${texMat(C)}`,
      meaning: `${A.length * B[0].length} dot products of length ${B.length}, so ${A.length * B[0].length * B.length} multiply-adds. In a neural network, A would be a batch of inputs and B a weight matrix.`,
    };
  },
  mistakes: ["Assuming AB = BA. Usually it isn't, and BA may not even be defined.", "Confusing the matrix product (@) with element-wise multiplication (*)."],
};

const distance: Calculator = {
  id: "euclidean-distance",
  title: "Euclidean and Manhattan distance",
  group: "Linear algebra",
  lesson: "k-nearest-neighbors",
  formula: "d_2(q, x) = \\sqrt{\\sum_j (q_j - x_j)^2}, \\quad d_1(q, x) = \\sum_j |q_j - x_j|",
  symbols: [["q, x", "two points with the same number of features"], ["d_2", "straight-line distance"], ["d_1", "city-block distance"]],
  fields: [
    { key: "q", label: "q", kind: "vector", default: "45, 56", hint: "KNN example: age 45, income 56" },
    { key: "x", label: "x", kind: "vector", default: "60, 57" },
  ],
  compute(v) {
    const q = vec(v.q, "q"), x = vec(v.x, "x");
    sameLength(q, x, "q", "x");
    const diff = q.map((a, i) => x[i] - a);
    const sq = diff.map((d) => d * d);
    return {
      steps: [
        { label: "Differences per feature", tex: texRow(diff) },
        { label: "Square and add", tex: `${sq.map((s) => n(s)).join(" + ")} = ${n(ml.sum(sq))}` },
        { label: "Square root (Euclidean)", tex: `\\sqrt{${n(ml.sum(sq))}} = ${n(Math.sqrt(ml.sum(sq)))}` },
        { label: "Absolute values added (Manhattan)", tex: `${diff.map((d) => n(Math.abs(d))).join(" + ")} = ${n(ml.sum(diff.map(Math.abs)))}` },
      ],
      result: `d_2 = ${n(Math.sqrt(ml.sum(sq)))}, \\; d_1 = ${n(ml.sum(diff.map(Math.abs)))}`,
      meaning: "Distance is only meaningful when features share a scale. A feature with a large range dominates both distances, so standardize first.",
    };
  },
  mistakes: ["Computing distances on unscaled features.", "Using Euclidean distance on very high-dimensional data, where distances concentrate."],
};

const meanVar: Calculator = {
  id: "mean-variance",
  title: "Mean and variance",
  group: "Statistics",
  lesson: "probability-and-statistics",
  formula: "\\mu = \\frac{1}{n}\\sum_i x_i, \\qquad \\sigma^2 = \\frac{1}{n}\\sum_i (x_i - \\mu)^2",
  symbols: [["x_i", "the i-th value"], ["n", "number of values"], ["\\mu", "mean"], ["\\sigma^2, \\sigma", "variance and standard deviation (population form)"]],
  fields: [{ key: "x", label: "values", kind: "vector", default: "2, 4, 4, 4, 5, 5, 7, 9" }],
  compute(v) {
    const x = vec(v.x, "values");
    const m = ml.mean(x);
    const dev = x.map((a) => (a - m) ** 2);
    const varp = ml.mean(dev);
    return {
      steps: [
        { label: "Mean", tex: `\\mu = \\frac{${x.map((a) => n(a)).join(" + ")}}{${x.length}} = ${n(m)}` },
        { label: "Squared deviations from the mean", tex: texRow(dev) },
        { label: "Average them", tex: `\\sigma^2 = \\frac{${n(ml.sum(dev))}}{${x.length}} = ${n(varp)}` },
        { label: "Standard deviation", tex: `\\sigma = \\sqrt{${n(varp)}} = ${n(Math.sqrt(varp))}` },
      ],
      result: `\\mu = ${n(m)},\\; \\sigma^2 = ${n(varp)},\\; \\sigma = ${n(Math.sqrt(varp))}`,
      meaning: `Values typically sit about ${n(Math.sqrt(varp))} away from the mean. The sample variance (dividing by n − 1 = ${x.length - 1}) would be ${n(ml.sum(dev) / Math.max(1, x.length - 1))}.`,
    };
  },
  mistakes: ["Mixing up population (÷n) and sample (÷(n−1)) variance; libraries differ (NumPy defaults to ÷n, pandas to ÷(n−1)).", "Reporting variance in squared units when the standard deviation is more readable."],
};

const standardize: Calculator = {
  id: "standardization",
  title: "Standardization (z-scores)",
  group: "Statistics",
  lesson: "python-toolkit",
  formula: "z_i = \\frac{x_i - \\mu}{\\sigma}",
  symbols: [["x_i", "original value"], ["\\mu, \\sigma", "mean and standard deviation from the TRAINING data"], ["z_i", "standardized value"]],
  fields: [
    { key: "x", label: "training values (used to fit μ and σ)", kind: "vector", default: "22, 26, 44, 47, 60" },
    { key: "q", label: "new value to transform", kind: "number", default: "45" },
  ],
  compute(v) {
    const x = vec(v.x, "training values"), q = num(v.q, "new value");
    const m = ml.mean(x), s = Math.sqrt(ml.variance(x));
    if (s === 0) throw new Error("All training values are equal, so the standard deviation is 0.");
    return {
      steps: [
        { label: "Fit the mean on training data", tex: `\\mu = ${n(m)}` },
        { label: "Fit the standard deviation on training data", tex: `\\sigma = ${n(s)}` },
        { label: "Transform the new value", tex: `z = \\frac{${n(q)} - ${n(m)}}{${n(s)}} = ${n((q - m) / s)}` },
        { label: "Training values after transformation", tex: texRow(x.map((a) => (a - m) / s)) },
      ],
      result: `z = ${n((q - m) / s)}`,
      meaning: `The new value is ${n(Math.abs((q - m) / s))} standard deviations ${q >= m ? "above" : "below"} the training mean. Every feature now speaks in the same unit.`,
    };
  },
  mistakes: ["Fitting μ and σ on all data, including test data (leakage).", "Re-fitting the scaler in production instead of reusing the training statistics."],
};

const mseCalc: Calculator = {
  id: "mse",
  title: "Mean squared error",
  group: "Losses",
  lesson: "linear-regression",
  formula: "\\text{MSE} = \\frac{1}{n}\\sum_i (y_i - \\hat{y}_i)^2",
  symbols: [["y_i", "true value"], ["\\hat{y}_i", "prediction"], ["n", "number of examples"]],
  fields: [
    { key: "y", label: "true values y", kind: "vector", default: "2, 4, 5" },
    { key: "p", label: "predictions ŷ", kind: "vector", default: "1, 2, 3", hint: "Linear-regression worked example" },
  ],
  compute(v) {
    const y = vec(v.y, "y"), p = vec(v.p, "ŷ");
    sameLength(y, p, "y", "ŷ");
    const r = y.map((a, i) => a - p[i]);
    const sq = r.map((e) => e * e);
    return {
      steps: [
        { label: "Residuals y − ŷ", tex: texRow(r) },
        { label: "Square each", tex: texRow(sq) },
        { label: "Average", tex: `\\frac{${n(ml.sum(sq))}}{${y.length}} = ${n(ml.mse(y, p))}` },
      ],
      result: `\\text{MSE} = ${n(ml.mse(y, p))},\\; \\text{RMSE} = ${n(Math.sqrt(ml.mse(y, p)))}`,
      meaning: "Squared units are hard to read, so report RMSE (same units as y). Large errors dominate because they are squared.",
    };
  },
  mistakes: ["Forgetting to square (that would be mean error, where positive and negative errors cancel).", "Comparing MSE across targets with different units."],
};

const maeCalc: Calculator = {
  id: "mae",
  title: "Mean absolute error",
  group: "Losses",
  lesson: "regularization-and-regression-metrics",
  formula: "\\text{MAE} = \\frac{1}{n}\\sum_i |y_i - \\hat{y}_i|",
  symbols: [["y_i", "true value"], ["\\hat{y}_i", "prediction"]],
  fields: [
    { key: "y", label: "true values y", kind: "vector", default: "200, 250, 300, 350, 400" },
    { key: "p", label: "predictions ŷ", kind: "vector", default: "210, 240, 310, 340, 300", hint: "Last prediction is a big miss" },
  ],
  compute(v) {
    const y = vec(v.y, "y"), p = vec(v.p, "ŷ");
    sameLength(y, p, "y", "ŷ");
    const a = y.map((t, i) => Math.abs(t - p[i]));
    return {
      steps: [
        { label: "Absolute errors", tex: texRow(a) },
        { label: "Average", tex: `\\frac{${n(ml.sum(a))}}{${y.length}} = ${n(ml.mae(y, p))}` },
        { label: "Compare with RMSE", tex: `\\text{RMSE} = ${n(ml.rmse(y, p))}` },
      ],
      result: `\\text{MAE} = ${n(ml.mae(y, p))}`,
      meaning: "MAE is in the target's units and treats every unit of error equally, so a single big miss moves it far less than RMSE.",
    };
  },
  mistakes: ["Choosing MAE when large errors are genuinely much worse (use RMSE).", "Optimizing MAE with plain gradient descent near zero error, where its gradient is not defined."],
};

const bceCalc: Calculator = {
  id: "binary-cross-entropy",
  title: "Binary cross-entropy",
  group: "Losses",
  lesson: "logistic-regression",
  formula: "L = -\\big[y\\ln p + (1-y)\\ln(1-p)\\big]",
  symbols: [["y", "true label, 0 or 1"], ["p", "predicted probability of class 1"]],
  fields: [
    { key: "y", label: "true label y (0 or 1)", kind: "number", default: "1" },
    { key: "p", label: "predicted probability p", kind: "number", default: "0.5", hint: "Backprop example: p = 0.5" },
  ],
  compute(v) {
    const y = num(v.y, "y"), p = num(v.p, "p");
    if (y !== 0 && y !== 1) throw new Error("y must be 0 or 1.");
    if (p <= 0 || p >= 1) throw new Error("p must be strictly between 0 and 1.");
    const L = ml.bce(y, p);
    return {
      steps: [
        { label: "Pick the term for the true label", tex: y === 1 ? `L = -\\ln p = -\\ln(${n(p)})` : `L = -\\ln(1 - p) = -\\ln(${n(1 - p)})` },
        { label: "Evaluate", tex: `L = ${n(L)}` },
        { label: "Gradient with respect to the logit", tex: `\\frac{\\partial L}{\\partial z} = p - y = ${n(p - y)}` },
      ],
      result: `L = ${n(L)}`,
      meaning: `The model gave ${n(y === 1 ? p : 1 - p)} probability to the correct label. Confident mistakes cost a lot: probability 0.01 on the truth costs 4.6.`,
    };
  },
  mistakes: ["Passing probabilities to a loss that expects logits (or vice versa).", "Computing log(sigmoid(z)) directly for large |z|; use library functions that work from logits."],
};

const cceCalc: Calculator = {
  id: "categorical-cross-entropy",
  title: "Categorical cross-entropy",
  group: "Losses",
  lesson: "tokenization-and-pretraining",
  formula: "L = -\\ln p_{c}, \\quad p = \\text{softmax}(z)",
  symbols: [["z", "logits for K classes"], ["c", "index of the true class (starting at 0)"], ["p_c", "probability assigned to the true class"]],
  fields: [
    { key: "z", label: "logits z", kind: "vector", default: "2, 1, 0.1" },
    { key: "c", label: "true class index c", kind: "number", default: "0" },
  ],
  compute(v) {
    const z = vec(v.z, "z"), c = num(v.c, "c");
    if (!Number.isInteger(c) || c < 0 || c >= z.length) throw new Error(`c must be an integer from 0 to ${z.length - 1}.`);
    const p = ml.softmax(z);
    return {
      steps: [
        { label: "Softmax the logits", tex: `p = ${texRow(p)}` },
        { label: "Take the probability of the true class", tex: `p_{${c}} = ${n(p[c])}` },
        { label: "Negative log", tex: `L = -\\ln(${n(p[c])}) = ${n(-Math.log(p[c]))}` },
      ],
      result: `L = ${n(-Math.log(p[c]))}`,
      meaning: `Perfect confidence in the right class gives 0; uniform guessing over ${z.length} classes gives ln(${z.length}) = ${n(Math.log(z.length))}. LLMs are trained with exactly this loss over their vocabulary.`,
    };
  },
  mistakes: ["Applying softmax twice (once in the model, again inside the loss).", "Using categorical cross-entropy for multi-label problems (use one sigmoid per label)."],
};

const gdCalc: Calculator = {
  id: "gradient-update",
  title: "Gradient-descent update",
  group: "Optimization",
  lesson: "gradient-descent",
  formula: "w \\leftarrow w - \\eta\\frac{\\partial \\text{MSE}}{\\partial w}, \\quad \\frac{\\partial \\text{MSE}}{\\partial w} = \\frac{2}{n}\\sum_i(\\hat{y}_i - y_i)x_i",
  symbols: [["x, y", "data"], ["w, b", "slope and bias"], ["\\eta", "learning rate"], ["\\hat{y}_i = w x_i + b", "prediction"]],
  fields: [
    { key: "x", label: "x", kind: "vector", default: "1, 2, 3" },
    { key: "y", label: "y", kind: "vector", default: "2, 4, 5" },
    { key: "w", label: "w", kind: "number", default: "1" },
    { key: "b", label: "b", kind: "number", default: "0" },
    { key: "lr", label: "learning rate η", kind: "number", default: "0.1" },
  ],
  compute(v) {
    const x = vec(v.x, "x"), y = vec(v.y, "y"), w = num(v.w, "w"), b = num(v.b, "b"), lr = num(v.lr, "η");
    sameLength(x, y, "x", "y");
    const s = ml.gradientStep(x, y, w, b, lr);
    const g = s.grad;
    const err = g.yHat.map((p, i) => p - y[i]);
    return {
      steps: [
        { label: "Predictions", tex: `\\hat{y} = ${texRow(g.yHat)}` },
        { label: "Errors ŷ − y", tex: texRow(err) },
        { label: "Loss before the step", tex: `\\text{MSE} = ${n(g.mse)}` },
        { label: "Gradient for w", tex: `\\frac{2}{${x.length}}(${err.map((e, i) => `${n(e, true)}\\cdot${n(x[i], true)}`).join(" + ")}) = ${n(g.dw)}` },
        { label: "Gradient for b", tex: `\\frac{2}{${x.length}}(${err.map((e) => n(e, true)).join(" + ")}) = ${n(g.db)}` },
        { label: "Update", tex: `w = ${n(w)} - ${n(lr)}\\cdot${n(g.dw, true)} = ${n(s.w)}, \\quad b = ${n(b)} - ${n(lr)}\\cdot${n(g.db, true)} = ${n(s.b)}` },
        { label: "Loss after the step", tex: `\\text{MSE} = ${n(ml.linearGrad(x, y, s.w, s.b).mse)}` },
      ],
      result: `w = ${n(s.w)},\\; b = ${n(s.b)}`,
      meaning: ml.linearGrad(x, y, s.w, s.b).mse < g.mse
        ? "The loss decreased: the step moved against the gradient by a safe amount."
        : "The loss increased: the learning rate is too large for this data and the step overshot.",
    };
  },
  mistakes: ["Adding the gradient instead of subtracting it.", "Using a learning rate larger than 2 divided by the steepest curvature, which diverges."],
};

const cosineCalc: Calculator = {
  id: "cosine-similarity",
  title: "Cosine similarity",
  group: "NLP",
  lesson: "text-to-vectors",
  formula: "\\cos(a, b) = \\frac{a \\cdot b}{\\lVert a\\rVert\\,\\lVert b\\rVert}",
  symbols: [["a, b", "vectors of equal length"], ["\\lVert a\\rVert", "length of a"]],
  fields: [
    { key: "a", label: "a", kind: "vector", default: "1, 2, 0" },
    { key: "b", label: "b", kind: "vector", default: "2, 1, 1" },
  ],
  compute(v) {
    const a = vec(v.a, "a"), b = vec(v.b, "b");
    sameLength(a, b, "a", "b");
    const d = ml.dot(a, b), na = ml.norm(a), nb = ml.norm(b);
    if (na === 0 || nb === 0) throw new Error("A zero vector has no direction, so cosine is undefined.");
    return {
      steps: [
        { label: "Dot product", tex: `a\\cdot b = ${n(d)}` },
        { label: "Lengths", tex: `\\lVert a\\rVert = ${n(na)},\\; \\lVert b\\rVert = ${n(nb)}` },
        { label: "Divide", tex: `\\frac{${n(d)}}{${n(na)} \\times ${n(nb)}} = ${n(d / (na * nb))}` },
      ],
      result: `\\cos = ${n(d / (na * nb))}`,
      meaning: `Angle ≈ ${n((Math.acos(Math.max(-1, Math.min(1, d / (na * nb)))) * 180) / Math.PI)}°. 1 means same direction, 0 unrelated (perpendicular), −1 opposite. Length doesn't matter, so long and short documents can still match.`,
    };
  },
  mistakes: ["Using dot products on unnormalized vectors when the model expects cosine.", "Comparing vectors from different embedding models."],
};

const tfidfCalc: Calculator = {
  id: "tf-idf",
  title: "TF-IDF",
  group: "NLP",
  lesson: "text-to-vectors",
  formula: "\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\ln\\frac{N}{\\text{DF}(t)}",
  symbols: [["t", "term"], ["d", "document"], ["N", "number of documents"], ["\\text{TF}", "count of t in d"], ["\\text{DF}", "documents containing t"]],
  fields: [
    { key: "docs", label: "corpus (one document per line)", kind: "text", default: "the model learns from data\ngradient descent trains the model\nthe cat sat on the mat" },
    { key: "term", label: "term", kind: "word", default: "model" },
    { key: "doc", label: "document number (1-based)", kind: "number", default: "2" },
  ],
  compute(v) {
    const docs = v.docs.split("\n").map((d) => d.trim()).filter(Boolean);
    if (!docs.length) throw new Error("Add at least one document.");
    const term = v.term.trim().toLowerCase();
    const d = num(v.doc, "document number");
    if (!Number.isInteger(d) || d < 1 || d > docs.length) throw new Error(`Document number must be between 1 and ${docs.length}.`);
    const r = ml.tfidf(docs);
    const j = r.vocab.indexOf(term);
    if (j < 0) throw new Error(`"${term}" does not appear in the corpus.`);
    const tf = r.tf[d - 1][j], df = r.df[j], N = docs.length;
    return {
      steps: [
        { label: "Term frequency in the document", tex: `\\text{TF} = ${tf}` },
        { label: "Document frequency", tex: `\\text{DF} = ${df} \\text{ of } N = ${N}` },
        { label: "Inverse document frequency", tex: `\\text{IDF} = \\ln\\frac{${N}}{${df}} = ${n(r.idf[j])}` },
        { label: "Multiply", tex: `\\text{TF-IDF} = ${tf} \\times ${n(r.idf[j])} = ${n(r.tfidf[d - 1][j])}` },
      ],
      result: `\\text{TF-IDF}(\\text{${term}}, d_{${d}}) = ${n(r.tfidf[d - 1][j])}`,
      meaning: df === N ? `"${term}" is in every document, so it carries no distinguishing information (IDF = 0).` : `Rarer terms get higher IDF; "${term}" appears in ${df} of ${N} documents.`,
    };
  },
  mistakes: ["Comparing hand results with scikit-learn's default (smoothed IDF and L2-normalized rows).", "Fitting the vocabulary on test documents."],
};

const counts = (s: string) => {
  const c = vec(s, "class counts");
  if (c.some((x) => x < 0)) throw new Error("Counts must be non-negative.");
  if (ml.sum(c) === 0) throw new Error("At least one count must be positive.");
  return c;
};

const giniCalc: Calculator = {
  id: "gini-impurity",
  title: "Gini impurity",
  group: "Trees",
  lesson: "decision-trees",
  formula: "G = 1 - \\sum_k p_k^2",
  symbols: [["p_k", "fraction of the node's examples in class k"]],
  fields: [{ key: "c", label: "class counts in the node", kind: "vector", default: "4, 1" }],
  compute(v) {
    const c = counts(v.c);
    const total = ml.sum(c);
    const p = c.map((x) => x / total);
    return {
      steps: [
        { label: "Class proportions", tex: `p = ${texRow(p)}` },
        { label: "Squares", tex: `${p.map((x) => `${n(x)}^2`).join(" + ")} = ${n(ml.sum(p.map((x) => x * x)))}` },
        { label: "Subtract from 1", tex: `G = 1 - ${n(ml.sum(p.map((x) => x * x)))} = ${n(ml.gini(c))}` },
      ],
      result: `G = ${n(ml.gini(c))}`,
      meaning: `0 means a pure node; the maximum for ${c.length} classes is ${n(1 - 1 / c.length)}. A split is good if the weighted Gini of its children is much lower than the parent's.`,
    };
  },
  mistakes: ["Forgetting to weight child impurities by their sizes when scoring a split.", "Comparing Gini values across nodes with different numbers of classes."],
};

const entropyCalc: Calculator = {
  id: "entropy",
  title: "Entropy",
  group: "Trees",
  lesson: "decision-trees",
  formula: "H = -\\sum_k p_k \\log_2 p_k",
  symbols: [["p_k", "fraction in class k (terms with p_k = 0 contribute 0)"], ["H", "impurity in bits"]],
  fields: [{ key: "c", label: "class counts", kind: "vector", default: "5, 5" }],
  compute(v) {
    const c = counts(v.c);
    const total = ml.sum(c);
    const p = c.map((x) => x / total).filter((x) => x > 0);
    return {
      steps: [
        { label: "Proportions (nonzero)", tex: texRow(p) },
        { label: "Terms −p log₂ p", tex: `${p.map((x) => n(-x * Math.log2(x))).join(" + ")}` },
        { label: "Sum", tex: `H = ${n(ml.entropy(c))} \\text{ bits}` },
      ],
      result: `H = ${n(ml.entropy(c))}\\text{ bits}`,
      meaning: `Maximum uncertainty for ${c.length} classes is log₂(${c.length}) = ${n(Math.log2(c.length))} bits. Information gain = parent entropy − weighted child entropy.`,
    };
  },
  mistakes: ["Using natural log and comparing with values in bits.", "Taking log(0) instead of treating 0·log 0 as 0."],
};

const prfCalc: Calculator = {
  id: "precision-recall-f1",
  title: "Precision, recall, and F1",
  group: "Classification metrics",
  lesson: "classification-metrics",
  formula: "P = \\frac{TP}{TP + FP}, \\quad R = \\frac{TP}{TP + FN}, \\quad F_1 = \\frac{2PR}{P + R}",
  symbols: [["TP", "true positives"], ["FP", "false positives"], ["FN", "false negatives"], ["TN", "true negatives"]],
  fields: [
    { key: "tp", label: "TP", kind: "number", default: "3" },
    { key: "fp", label: "FP", kind: "number", default: "2" },
    { key: "fn", label: "FN", kind: "number", default: "2" },
    { key: "tn", label: "TN", kind: "number", default: "3" },
  ],
  compute(v) {
    const c = { tp: num(v.tp, "TP"), fp: num(v.fp, "FP"), fn: num(v.fn, "FN"), tn: num(v.tn, "TN") };
    if (Object.values(c).some((x) => x < 0 || !Number.isInteger(x))) throw new Error("Counts must be non-negative integers.");
    const P = ml.precision(c), R = ml.recall(c), F = ml.f1(c);
    return {
      steps: [
        { label: "Precision", tex: `\\frac{${c.tp}}{${c.tp} + ${c.fp}} = ${n(P)}` },
        { label: "Recall", tex: `\\frac{${c.tp}}{${c.tp} + ${c.fn}} = ${n(R)}` },
        { label: "F1 (harmonic mean)", tex: `\\frac{2 \\times ${n(P)} \\times ${n(R)}}{${n(P)} + ${n(R)}} = ${n(F)}` },
        { label: "Accuracy, for comparison", tex: `\\frac{${c.tp} + ${c.tn}}{${c.tp + c.fp + c.fn + c.tn}} = ${n(ml.accuracy(c))}` },
      ],
      result: `P = ${n(P)},\\; R = ${n(R)},\\; F_1 = ${n(F)}`,
      meaning: `Of ${c.tp + c.fp} alarms, ${c.tp} were real; of ${c.tp + c.fn} real positives, ${c.tp} were caught. F1 is low whenever either precision or recall is low.`,
    };
  },
  mistakes: ["Swapping FP and FN in the formulas.", "Reporting accuracy on imbalanced data instead of precision and recall."],
};

const attentionCalc: Calculator = {
  id: "attention",
  title: "Scaled dot-product attention",
  group: "Transformers",
  lesson: "self-attention",
  formula: "\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V",
  symbols: [["q", "one query vector"], ["K", "keys, one row per token"], ["V", "values, one row per token"], ["d_k", "key dimension"]],
  fields: [
    { key: "q", label: "query q", kind: "vector", default: "1, 0", hint: "Course example" },
    { key: "k", label: "keys K (rows separated by ;)", kind: "matrix", default: "1, 0; 0, 1" },
    { key: "v", label: "values V (rows separated by ;)", kind: "matrix", default: "10, 0; 0, 6" },
  ],
  compute(v) {
    const q = vec(v.q, "q"), K = mat(v.k, "K"), V = mat(v.v, "V");
    if (K[0].length !== q.length) throw new Error(`Keys have dimension ${K[0].length} but the query has ${q.length}.`);
    if (K.length !== V.length) throw new Error(`There are ${K.length} keys but ${V.length} values; they must match one-to-one.`);
    const r = ml.attention([q], K, V);
    const ex = r.scaled[0].map((s) => Math.exp(s));
    return {
      steps: [
        { label: "Scores: query dotted with each key", tex: `${texRow(r.scores[0])}` },
        { label: `Scale by √d_k = √${K[0].length} = ${n(Math.sqrt(K[0].length))}`, tex: texRow(r.scaled[0]) },
        { label: "Exponentiate", tex: `${texRow(ex)},\\; \\text{sum} = ${n(ml.sum(ex))}` },
        { label: "Normalize into weights", tex: `${texRow(r.weights[0])}` },
        { label: "Weighted sum of values", tex: `${r.weights[0].map((w, j) => `${n(w)}\\,${texRow(V[j])}`).join(" + ")} = ${texRow(r.output[0])}` },
      ],
      result: `\\text{output} = ${texRow(r.output[0])}`,
      meaning: `The token draws most from value ${r.weights[0].indexOf(Math.max(...r.weights[0])) + 1}, whose key best matches the query. The output is always a blend of value vectors, never a hard lookup.`,
    };
  },
  mistakes: ["Dividing by d_k instead of √d_k.", "Applying softmax down columns instead of across each query's row."],
};

// Remaining Math Lab calculators. Each follows the same contract as those in index.ts.
export const MORE_CALCULATORS: Calculator[] = [
  matmul, distance, meanVar, standardize, mseCalc, maeCalc, bceCalc, cceCalc, gdCalc,
  cosineCalc, tfidfCalc, giniCalc, entropyCalc, prfCalc, attentionCalc,
];
