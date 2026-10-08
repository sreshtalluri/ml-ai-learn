// Tiny byte-pair encoding for teaching. Words are split on spaces; "▁" marks a word start (as in SentencePiece).

export type Merge = [string, string];

const words = (text: string) => text.toLowerCase().split(/\s+/).filter(Boolean).map((w) => ["▁", ...w]);

/** Learn up to `n` merges: repeatedly join the most frequent adjacent pair (ties: first seen). */
export function trainBpe(corpus: string, n: number): Merge[] {
  let seqs = words(corpus);
  const merges: Merge[] = [];
  for (let k = 0; k < n; k++) {
    const counts = new Map<string, number>();
    for (const s of seqs) for (let i = 0; i < s.length - 1; i++) {
      const key = s[i] + "\u0000" + s[i + 1];
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    let best = "", bestC = 1;
    for (const [key, c] of counts) if (c > bestC) { best = key; bestC = c; }
    if (!best) break;
    const [a, b] = best.split("\u0000") as Merge;
    merges.push([a, b]);
    seqs = seqs.map((s) => applyMerge(s, a, b));
  }
  return merges;
}

function applyMerge(s: string[], a: string, b: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < s.length; i++) {
    if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { out.push(a + b); i++; } else out.push(s[i]);
  }
  return out;
}

/** Tokenize text by applying the first `k` merges in training order. */
export function encode(text: string, merges: Merge[], k = merges.length): string[] {
  return words(text).flatMap((w) => merges.slice(0, k).reduce((s, [a, b]) => applyMerge(s, a, b), w));
}
