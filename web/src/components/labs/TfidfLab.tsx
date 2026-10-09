"use client";
import { useMemo, useState } from "react";
import { cosineSimilarity, fmt, tfidf } from "@/lib/ml";
import { LabFrame, Segmented, Stat, Tex, type TourStep } from "./ui";

const DUPLICATE = "the model learns from the data";
const DEFAULT = ["the model learns from data", "gradient descent trains the model", "the cat sat on the mat", "data quality matters more than the model"].join("\n");

export default function TfidfLab() {
  const [text, setText] = useState(DEFAULT);
  const [view, setView] = useState<"tf" | "tfidf">("tfidf");
  const docs = useMemo(() => text.split("\n").map((d) => d.trim()).filter(Boolean).slice(0, 8), [text]);
  const r = useMemo(() => (docs.length ? tfidf(docs) : null), [docs]);
  const [term, setTerm] = useState("model");
  const [docIdx, setDocIdx] = useState(1);

  // Guided tour: each step sets corpus, matrix view and the selected cell.
  const setup = (v: "tf" | "tfidf", t: string, di: number, corpus = DEFAULT) => { setText(corpus); setView(v); setTerm(t); setDocIdx(di); };
  const tour: TourStep[] = [
    { id: "counts", caption: "Four short documents, one per row, and one column per word. In counts view each cell is how often the word appears: “the” shows up twice in document 3.", apply: () => setup("tf", "the", 2) },
    { id: "idf-zero", caption: "Switch to TF-IDF. “the” is in all four documents, so its IDF is ln(4/4) = 0 and its whole column goes blank. A word everyone uses tells documents apart not at all.", apply: () => setup("tfidf", "the", 2) },
    { id: "rare", caption: "“gradient” appears only in document 2, so it gets the largest IDF, ln 4 = 1.39. Rare words carry the most weight.", apply: () => setup("tfidf", "gradient", 1) },
    { id: "sweep", caption: "Walk the orange outline along document 4. “model” is in three documents and scores only 0.29; “data” is in two and scores 0.69; words unique to this document score 1.39.", apply: () => setup("tfidf", "data", 3),
      animate: (t) => { if (!r) return; const own = r.vocab.filter((_, jj) => (r.tf[3]?.[jj] ?? 0) > 0); setTerm(own[Math.min(own.length - 1, Math.floor(t * own.length))]); }, animMs: 3400 },
    { id: "similarity", caption: "Each row is now a vector, and the purple table compares them by cosine. Documents 1 and 4 share “data” and “model” and score 0.09; document 3 shares only “the” and scores 0.", apply: () => setup("tfidf", "model", 0) },
    { id: "duplicate", caption: "Type a near-copy of document 1 as document 5. Once every word is in, its cosine with document 1 hits 1.00: the extra “the” weighs nothing.", apply: () => setup("tfidf", "model", 4, DEFAULT),
      animate: (t) => { const words = DUPLICATE.split(" "); const typed = words.slice(0, Math.round(t * words.length)).join(" "); setText(typed ? `${DEFAULT}\n${typed}` : DEFAULT); }, animMs: 3200 },
  ];

  if (!r) {
    return (
      <LabFrame id="tfidf" title="TF-IDF lab" subtitle="One document per line.">
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={5} className="w-full rounded-lg border border-line bg-bg p-3 font-mono text-sm" aria-label="Corpus" />
        <p className="text-sm text-muted mt-2">Add at least one document.</p>
      </LabFrame>
    );
  }
  const j = Math.max(0, r.vocab.indexOf(term));
  const w = r.vocab[j];
  const d = Math.min(docIdx, docs.length - 1);
  const N = docs.length;
  const m = view === "tf" ? r.tf : r.tfidf;
  const max = Math.max(1e-9, ...m.flat());
  const sims = docs.map((_, a) => docs.map((__, b) => cosineSimilarity(r.tfidf[a], r.tfidf[b])));

  return (
    <LabFrame
      id="tfidf"
      title="TF-IDF lab"
      subtitle="Edit the corpus (one document per line). Click any cell to see its calculation. Formula: TF × ln(N / DF), with raw counts for TF."
      onReset={() => { setText(DEFAULT); setTerm("model"); setDocIdx(1); setView("tfidf"); }}
      presets={[{ label: "Add a near-duplicate", apply: () => setText(DEFAULT + "\n" + DUPLICATE) }]}
      tour={tour}
      controls={
        <>
          <label className="block">
            <span className="text-[0.8rem] text-muted">Corpus</span>
            <textarea value={text} onChange={(e) => setText(e.target.value)} rows={6} className="mt-1 w-full rounded-lg border border-line bg-bg p-2 font-mono text-xs" />
          </label>
          <Segmented label="Matrix shows" value={view} onChange={setView} options={[{ value: "tf", label: "Counts (BoW)" }, { value: "tfidf", label: "TF-IDF" }]} />
        </>
      }
      readout={
        <>
          <Stat label="documents N" value={N} />
          <Stat label="vocabulary" value={r.vocab.length} />
          <Stat label={`TF("${w}", d${d + 1})`} value={r.tf[d][j]} />
          <Stat label={`DF("${w}")`} value={r.df[j]} />
          <Stat label="IDF" value={fmt(r.idf[j], 3)} />
          <Stat label="TF-IDF" value={fmt(r.tfidf[d][j], 3)} color="var(--c-teal)" />
        </>
      }
      interpretation={
        <>
          <p className="text-ink">Worked calculation for “{w}” in document {d + 1}:{" "}
            <Tex>{`\\text{TF} = ${r.tf[d][j]},\\; \\text{DF} = ${r.df[j]},\\; \\text{IDF} = \\ln(${N}/${r.df[j]}) = ${fmt(r.idf[j], 3)},\\; \\text{TF-IDF} = ${r.tf[d][j]} \\times ${fmt(r.idf[j], 3)} = ${fmt(r.tfidf[d][j], 3)}`}</Tex>
          </p>
          <p className="mt-1">{r.df[j] === N ? `“${w}” appears in every document, so IDF = ln(1) = 0: it carries no information for telling documents apart.` : r.df[j] === 1 ? `“${w}” appears in only one document, so it gets the largest possible IDF, ln(${N}).` : "Words that are frequent in one document but rare across the corpus get the highest weights."}</p>
        </>
      }
    >
      <div className="overflow-x-auto">
        <table className="font-mono text-[0.7rem] border-separate border-spacing-0.5">
          <thead>
            <tr>
              <th />
              {r.vocab.map((v, jj) => (
                <th key={v} className="font-normal align-bottom h-20">
                  <button type="button" onClick={() => setTerm(v)} className={`[writing-mode:vertical-rl] rotate-180 px-0.5 ${jj === j ? "text-accent font-semibold" : "text-muted"}`}>{v}</button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {m.map((row, di) => (
              <tr key={di}>
                <th className="pr-2 text-right font-normal text-faint">d{di + 1}</th>
                {row.map((v, jj) => (
                  <td key={jj}>
                    <button type="button" onClick={() => { setTerm(r.vocab[jj]); setDocIdx(di); }} aria-label={`${r.vocab[jj]} in document ${di + 1}: ${fmt(v, 2)}`}
                      className={`w-9 rounded py-1 ${jj === j && di === d ? "outline outline-2 outline-[var(--c-orange)]" : ""}`}
                      style={{ background: `color-mix(in srgb, var(--c-teal) ${Math.round((v / max) * 75)}%, var(--surface-2))` }}>
                      {v === 0 ? "·" : fmt(v, view === "tf" ? 0 : 2)}
                    </button>
                  </td>
                ))}
              </tr>
            ))}
            <tr>
              <th className="pr-2 text-right font-normal text-faint">IDF</th>
              {r.idf.map((v, jj) => <td key={jj} className={`text-center ${jj === j ? "text-accent" : "text-faint"}`}>{fmt(v, 2)}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="text-sm font-medium mt-6 mb-2">Cosine similarity between documents (TF-IDF vectors)</p>
      <table className="font-mono text-[0.72rem] border-separate border-spacing-0.5">
        <thead><tr><th />{docs.map((_, b) => <th key={b} className="font-normal text-faint px-1">d{b + 1}</th>)}</tr></thead>
        <tbody>
          {sims.map((row, a) => (
            <tr key={a}>
              <th className="pr-2 font-normal text-faint text-right">d{a + 1}</th>
              {row.map((v, b) => (
                <td key={b} className="px-2 py-1 text-center rounded" style={{ background: `color-mix(in srgb, var(--c-purple) ${Math.round(v * 70)}%, var(--surface-2))`, color: v > 0.6 ? "white" : undefined }}>{fmt(v, 2)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-muted mt-2">Documents sharing no rare words score 0, even if they mean similar things. Embeddings fix that; see the next lesson.</p>
    </LabFrame>
  );
}
