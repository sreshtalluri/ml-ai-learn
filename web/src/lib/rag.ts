// Offline, deterministic retrieval for the RAG lab. Same SYNTHETIC documents as guide/code/16-rag/rag_eval.py.
import { cosineSimilarity, tokenize } from "./ml";

export const DOCS: Record<string, string> = {
  refunds: "Refund policy. Customers may request a refund within 30 days of purchase. Refunds are issued to the original payment method within 5 business days. Digital downloads are refundable only if they have not been accessed. Shipping fees are not refundable unless the item arrived damaged.",
  shipping: "Shipping. Standard shipping takes 3 to 7 business days. Express shipping takes 1 to 2 business days and costs 15 dollars. International orders may take up to 21 days and can incur customs fees paid by the customer.",
  accounts: "Accounts. Passwords must be at least 12 characters. Two-factor authentication can be enabled in settings. Accounts inactive for 24 months are closed after an email warning. Deleted accounts cannot be recovered.",
  warranty: "Warranty. Hardware products include a 1-year limited warranty covering manufacturing defects. The warranty does not cover accidental damage, water damage, or unauthorized repairs. Extended warranties add 2 years.",
};

export const QUERIES = [
  { q: "How many days do I have to ask for my money back?", evidence: "within 30 days", answer: "You can request a refund within 30 days of purchase." },
  { q: "How long does express delivery take?", evidence: "1 to 2 business days", answer: "Express shipping takes 1 to 2 business days." },
  { q: "Does the warranty cover water damage?", evidence: "does not cover", answer: "No. The warranty does not cover water damage." },
  { q: "Are shipping costs refunded if my item was broken on arrival?", evidence: "arrived damaged", answer: "Yes: shipping fees are refundable when the item arrived damaged." },
  { q: "Can I get reimbursed for a game I never opened?", evidence: "not been accessed", answer: "Yes: digital downloads are refundable if they have not been accessed." },
];

const STOP = new Set("a an the and or of to in on for is are be can if my i do does how what which with it its that this have has may must not was were by at from as".split(" "));
// Simulated semantic matching: a hand-written synonym map stands in for an embedding model.
const SYNONYMS: Record<string, string[]> = {
  reimbursed: ["refund", "refundable"], money: ["refund"], back: ["refund"], game: ["digital", "downloads"],
  opened: ["accessed"], never: ["not"], broken: ["damaged"], delivery: ["shipping"], costs: ["fees"], password: ["passwords"],
};

export interface Chunk { id: number; doc: string; text: string }

export function chunkDocs(size: number, overlap: number): Chunk[] {
  const out: Chunk[] = [];
  const step = Math.max(1, size - overlap);
  for (const [doc, text] of Object.entries(DOCS)) {
    const w = text.split(/\s+/);
    for (let i = 0; i < Math.max(1, w.length - overlap); i += step) out.push({ id: out.length, doc, text: w.slice(i, i + size).join(" ") });
  }
  return out;
}

export const terms = (s: string) => tokenize(s).filter((t) => !STOP.has(t));

export function expand(query: string): string[] {
  return terms(query).flatMap((t) => [t, ...(SYNONYMS[t] ?? [])]);
}

/** TF-IDF (ln N/df) vectors fitted on the chunks, cosine scores against the query. */
export function retrieve(chunks: Chunk[], query: string, semantic: boolean) {
  const docs = chunks.map((c) => terms(c.text));
  const vocab = [...new Set(docs.flat())];
  const df = vocab.map((v) => docs.filter((d) => d.includes(v)).length);
  const idf = df.map((d) => Math.log(chunks.length / d));
  const vec = (toks: string[]) => vocab.map((v, j) => toks.filter((t) => t === v).length * idf[j]);
  const q = vec(semantic ? expand(query) : terms(query));
  return chunks.map((c, i) => ({ chunk: c, score: cosineSimilarity(q, vec(docs[i])) })).filter((r) => r.score > 0).sort((a, b) => b.score - a.score || a.chunk.id - b.chunk.id);
}

/** Simulated cross-encoder: rewards query terms (and synonyms) appearing close together in the chunk. */
export function rerankScore(chunk: Chunk, query: string): number {
  const toks = terms(chunk.text);
  const q = new Set(expand(query));
  const pos = toks.map((t, i) => (q.has(t) ? i : -1)).filter((i) => i >= 0);
  if (!pos.length) return 0;
  const coverage = new Set(pos.map((i) => toks[i])).size / Math.max(1, terms(query).length);
  const span = pos.length > 1 ? pos[pos.length - 1] - pos[0] + 1 : toks.length;
  return +(coverage * (1 + pos.length / span)).toFixed(4);
}
