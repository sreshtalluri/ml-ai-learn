// Math Lab calculator contract. Each calculator is plain data + a pure compute function,
// so one component renders all of them and tests can call compute() directly.

export interface CalcField {
  key: string;
  label: string;
  /** number: "0.5"; vector: "1, 2, 3"; matrix: "1, 2; 3, 4" (rows split by ;); text: free text (one doc per line) */
  kind: "number" | "word" | "vector" | "matrix" | "text";
  default: string;
  hint?: string;
}

export interface CalcStep {
  /** Plain-language label for the step, e.g. "Multiply element-wise". */
  label: string;
  /** The arithmetic with real numbers substituted, as LaTeX. */
  tex: string;
}

export interface CalcResult {
  steps: CalcStep[];
  /** Final answer as LaTeX. */
  result: string;
  /** What the number means, in one or two sentences. */
  meaning: string;
}

export interface Calculator {
  id: string;
  title: string;
  group: "Linear algebra" | "Statistics" | "Activations" | "Losses" | "Optimization" | "NLP" | "Trees" | "Classification metrics" | "Transformers";
  lesson: string;                 // lesson slug
  formula: string;                // LaTeX
  symbols: [string, string][];    // [LaTeX symbol, meaning]
  fields: CalcField[];
  compute: (v: Record<string, string>) => CalcResult;
  mistakes: string[];
}

// ---------- input parsing (throws readable errors shown inline) ----------

export function num(s: string, name: string): number {
  const v = Number(s.trim());
  if (s.trim() === "" || !Number.isFinite(v)) throw new Error(`${name} must be a number.`);
  return v;
}

export function vec(s: string, name: string): number[] {
  const parts = s.split(/[,\s]+/).filter(Boolean);
  if (!parts.length) throw new Error(`${name} needs at least one number.`);
  return parts.map((p, i) => num(p, `${name}[${i + 1}]`));
}

export function mat(s: string, name: string): number[][] {
  const rows = s.split(";").map((r) => vec(r, name));
  if (rows.some((r) => r.length !== rows[0].length)) throw new Error(`Every row of ${name} must have the same length.`);
  return rows;
}

/** Number formatting for LaTeX: up to 4 decimals, trailing zeros trimmed, negatives parenthesized when asked. */
export function n(v: number, paren = false): string {
  if (!Number.isFinite(v)) return v > 0 ? "\\infty" : "-\\infty";
  const s = String(+v.toFixed(4));
  return paren && v < 0 ? `(${s})` : s;
}
