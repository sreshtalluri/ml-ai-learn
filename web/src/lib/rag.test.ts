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
  it("bigger chunks raise evidence recall, as in the lesson", () => {
    expect(recallAt(3, 50, 5, false)).toBeGreaterThanOrEqual(recallAt(3, 12, 4, false));
  });
});
