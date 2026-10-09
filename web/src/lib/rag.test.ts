import { describe, expect, it } from "vitest";
import { chunkDocs, QUERIES, retrieve } from "./rag";

const recallAt = (k: number, size: number, overlap: number, semantic: boolean) => {
  const chunks = chunkDocs(size, overlap);
  return QUERIES.filter(({ q, evidence }) => retrieve(chunks, q, semantic).slice(0, k).some((r) => r.chunk.text.includes(evidence))).length / QUERIES.length;
};

describe("RAG lab retrieval", () => {
  it("chunks with overlap into the expected count", () => {
    expect(chunkDocs(20, 5).length).toBe(10); // matches the Python script
  });
  it("the vocabulary-mismatch query misses with lexical retrieval and hits with semantic expansion", () => {
    const chunks = chunkDocs(20, 5);
    const miss = QUERIES[4];
    expect(retrieve(chunks, miss.q, false).slice(0, 3).some((r) => r.chunk.text.includes(miss.evidence))).toBe(false);
    expect(retrieve(chunks, miss.q, true).slice(0, 3).some((r) => r.chunk.text.includes(miss.evidence))).toBe(true);
  });
  it("reproduces the lesson's worked example (and guide/code/16-rag/rag_eval.py)", () => {
    const q = QUERIES[3];
    const top = (size: number) => retrieve(chunkDocs(size, 5), q.q, false).slice(0, 2).map((r) => [r.chunk.doc, +r.score.toFixed(3)]);
    expect(top(20)).toEqual([["refunds", 0.408], ["shipping", 0.347]]); // evidence first at 20 words
    expect(top(30)).toEqual([["shipping", 0.303], ["refunds", 0.283]]); // and second at 30 words
  });
  it("bigger chunks raise evidence recall, as in the lesson", () => {
    expect(recallAt(3, 50, 5, false)).toBeGreaterThanOrEqual(recallAt(3, 12, 4, false));
  });
});
