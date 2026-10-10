import { describe, expect, it } from "vitest";
import { effectiveMerges, encode, trainBpe } from "./bpe";

describe("byte-pair encoding", () => {
  // Course example: low x5, lower x2, newest x6, widest x3.
  const corpus = [...Array(5).fill("low"), ...Array(2).fill("lower"), ...Array(6).fill("newest"), ...Array(3).fill("widest")].join(" ");
  it("merges the most frequent pair first", () => {
    const m = trainBpe(corpus, 3);
    expect(m[0]).toEqual(["e", "s"]);   // 6 + 3 = 9
    expect(m[1]).toEqual(["es", "t"]);
  });
  it("encodes with merges and round-trips", () => {
    const m = trainBpe(corpus, 20);
    const t = encode("newest lowest", m);
    expect(t.join("").replace(/▁/g, " ").trim()).toBe("newest lowest");
    expect(t.length).toBeLessThan("newestlowest".length);
    expect(encode("newest", m, 0)).toEqual(["▁", "n", "e", "w", "e", "s", "t"]);
  });
  it("splits punctuation off without a stray word marker", () => {
    expect(encode("low.", [], 0)).toEqual(["▁", "l", "o", "w", "."]);
  });
  it("lists exactly the merges that change a text", () => {
    const m = trainBpe(corpus, 20);
    const eff = effectiveMerges("lowest", m);
    for (let k = 1; k <= m.length; k++) {
      const changed = encode("lowest", m, k).length !== encode("lowest", m, k - 1).length;
      expect(eff.includes(k), `merge ${k}`).toBe(changed);
    }
  });
});
