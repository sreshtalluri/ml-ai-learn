"use client";
import { useMemo, useState } from "react";
import { fmt } from "@/lib/ml";
import { chunkDocs, QUERIES, rerankScore, retrieve } from "@/lib/rag";
import { LabFrame, Slider, Stat, Toggle, type TourStep } from "./ui";

const DOC_COLOR: Record<string, string> = { refunds: "var(--c-blue)", shipping: "var(--c-teal)", accounts: "var(--c-purple)", warranty: "var(--c-orange)" };

export default function RagLab() {
  const [qi, setQi] = useState(3);
  const [custom, setCustom] = useState("");
  const [size, setSize] = useState(20);
  const [overlap, setOverlap] = useState(5);
  const [k, setK] = useState(3);
  const [semantic, setSemantic] = useState(false);
  const [rerank, setRerank] = useState(false);

  const preset = QUERIES[qi];
  const query = custom.trim() || preset.q;
  const evidence = custom.trim() ? null : preset.evidence;
  const chunks = useMemo(() => chunkDocs(size, Math.min(overlap, size - 1)), [size, overlap]);
  const ranked = useMemo(() => {
    const r: { chunk: (typeof chunks)[number]; score: number; rr?: number }[] = retrieve(chunks, query, semantic);
    if (!rerank) return r;
    return [...r.slice(0, 10).map((x) => ({ ...x, rr: rerankScore(x.chunk, query) })).sort((a, b) => b.rr - a.rr), ...r.slice(10)];
  }, [chunks, query, semantic, rerank]);
  const top = ranked.slice(0, k);
  const hit = evidence ? top.some((r) => r.chunk.text.includes(evidence)) : null;
  const evidenceRank = evidence ? ranked.findIndex((r) => r.chunk.text.includes(evidence)) + 1 : 0;
  const contextWords = top.reduce((s, r) => s + r.chunk.text.split(/\s+/).length, 0);
  const citation = evidence ? top.findIndex((r) => r.chunk.text.includes(evidence)) + 1 : 0;

  // Guided tour (Watch mode + explainers). Each step sets the full state it needs, then animates.
  const setup = (o: { qi: number; size?: number; k?: number; semantic?: boolean; rerank?: boolean }) => {
    setQi(o.qi); setCustom(""); setSize(o.size ?? 20); setOverlap(5); setK(o.k ?? 3); setSemantic(!!o.semantic); setRerank(!!o.rerank);
  };
  const tour: TourStep[] = [
    { id: "chunks", caption: "Four policy documents are cut into overlapping chunks. Shrink the chunk size from 60 words to 20 and the index grows from 4 chunks to 10; each card below is one chunk.", apply: () => setup({ qi: 3, size: 60 }), animate: (t) => { const v = Math.round(60 - 40 * t); if (v !== size) setSize(v); }, animMs: 2600 },
    { id: "question", caption: "The question: are shipping costs refunded if my item was broken on arrival? Common words are dropped, and the remaining query terms are scored against every chunk.", apply: () => setup({ qi: 3 }) },
    { id: "lexical", caption: "Lexical retrieval rewards shared words, weighted by how rare they are. The refunds chunk with the underlined evidence ranks first; shipping chunks follow because they also say shipping.", apply: () => setup({ qi: 3 }) },
    { id: "miss", caption: "Now ask: can I get reimbursed for a game I never opened? The answer is in the refunds document, but no chunk shares a single word with the question, so nothing comes back.", apply: () => setup({ qi: 4 }) },
    { id: "semantic", caption: "Semantic matching knows reimbursed means refund, game means digital download, and opened means accessed. The same evidence now ranks first.", apply: () => setup({ qi: 4, semantic: true }) },
    { id: "topk", caption: "Ask about getting money back and the refund chunk only ranks fourth. Grow top-k and it enters the context at k = 4, but the words sent to the model keep climbing.", apply: () => setup({ qi: 0, k: 1 }), animate: (t) => { const v = Math.round(1 + 5 * t); if (v !== k) setK(v); }, animMs: 2600 },
    { id: "rerank", caption: "Back to k = 1 with the reranker on. It rescores the top 10 by how closely the query terms sit together and lifts the refund chunk from fourth to first.", apply: () => setup({ qi: 0, k: 1, rerank: true }) },
    { id: "no-prompt-fix", caption: "If the evidence never reaches the context, no prompt can fix it: the model can only say it doesn't know, or make something up. Fix retrieval first.", apply: () => setup({ qi: 4 }) },
  ];

  return (
    <LabFrame
      id="rag"
      tour={tour}
      title="RAG pipeline lab"
      subtitle="Four synthetic policy documents. Chunk them, retrieve for a question, optionally rerank, and see whether the evidence reaches the model."
      onReset={() => { setQi(3); setCustom(""); setSize(20); setOverlap(5); setK(3); setSemantic(false); setRerank(false); }}
      presets={[
        { label: "Retrieval miss", apply: () => { setQi(4); setCustom(""); setSemantic(false); } },
        { label: "Tiny chunks", apply: () => { setSize(8); setOverlap(0); } },
        { label: "k = 1", apply: () => setK(1) },
      ]}
      controls={
        <>
          <label className="block text-[0.8rem] text-muted">Question
            <select value={qi} onChange={(e) => { setQi(+e.target.value); setCustom(""); }} className="mt-1 w-full rounded-lg border border-line bg-bg px-2 py-1.5 text-sm text-ink">
              {QUERIES.map((q, i) => <option key={i} value={i}>{q.q}</option>)}
            </select>
          </label>
          <label className="block text-[0.8rem] text-muted">…or type your own
            <input value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="e.g. how long is the warranty?" className="mt-1 w-full rounded-lg border border-line bg-bg px-2 py-1.5 text-sm text-ink" />
          </label>
          <Slider label="chunk size (words)" value={size} min={6} max={60} onChange={setSize} />
          <Slider label="overlap (words)" value={Math.min(overlap, size - 1)} min={0} max={Math.min(20, size - 1)} onChange={setOverlap} />
          <Slider label="top-k chunks sent to the model" value={k} min={1} max={6} onChange={setK} />
          <Toggle label="Semantic matching (simulated embeddings)" checked={semantic} onChange={setSemantic} />
          <Toggle label="Rerank top 10 (simulated cross-encoder)" checked={rerank} onChange={setRerank} />
        </>
      }
      readout={
        <>
          <Stat label="chunks indexed" value={chunks.length} />
          <Stat label="chunks with any match" value={ranked.length} />
          {evidence && <Stat label="evidence rank" value={evidenceRank ? `#${evidenceRank}` : "not retrieved"} color={hit ? "var(--c-teal)" : "var(--c-red)"} />}
          <Stat label="context sent" value={`${contextWords} words`} />
        </>
      }
      interpretation={
        evidence === null
          ? "Custom question: no answer key, so judge for yourself whether the top chunks contain what's needed."
          : hit
            ? <><p className="text-ink">Grounded answer: “{preset.answer}” [{citation}]</p><p className="mt-1">The evidence made it into the context at position {citation}, so the model can answer and cite it.{evidenceRank > 1 && !rerank ? " It wasn't ranked first, though; try the reranker." : ""}</p></>
            : <><p className="text-ink">The evidence is not in the context. A well-instructed model should answer “I don&apos;t know”; a poorly instructed one will make something up.</p><p className="mt-1">{evidenceRank === 0 ? "Lexical retrieval found no chunk containing the evidence: the question uses different words (vocabulary mismatch). Turn on semantic matching." : `The evidence is at rank ${evidenceRank}, outside the top ${k}. Raise k, enable reranking, or change the chunk size.`} No prompt change fixes a retrieval miss.</p></>
      }
    >
      <p className="text-sm"><span className="text-muted">Query terms:</span> <span className="font-mono">{query}</span></p>
      <ol className="mt-3 space-y-2">
        {ranked.slice(0, 6).map((r, i) => {
          const inCtx = i < k;
          const isEv = evidence && r.chunk.text.includes(evidence);
          return (
            <li key={r.chunk.id} className={`rounded-lg border px-3 py-2 text-sm ${inCtx ? "border-line bg-surface" : "border-dashed border-line opacity-60"}`}>
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="font-mono text-faint">#{i + 1}</span>
                <span className="rounded-full px-2 py-0.5 text-ink" style={{ background: `color-mix(in srgb, ${DOC_COLOR[r.chunk.doc]} 18%, var(--surface))` }}>{r.chunk.doc}</span>
                <span className="font-mono">cosine {fmt(r.score, 3)}</span>
                {r.rr !== undefined && <span className="font-mono">rerank {fmt(r.rr, 3)}</span>}
                {inCtx && <span className="text-teal">in context [{i + 1}]</span>}
                {isEv && <span className="font-medium text-ink">contains the evidence</span>}
              </div>
              <p className="mt-1 text-muted">
                {isEv && evidence ? <>{r.chunk.text.split(evidence)[0]}<mark className="bg-transparent text-ink underline decoration-2 decoration-[var(--c-teal)]">{evidence}</mark>{r.chunk.text.split(evidence).slice(1).join(evidence)}</> : r.chunk.text}
              </p>
            </li>
          );
        })}
        {ranked.length === 0 && <li className="rounded-lg border border-dashed border-line px-3 py-6 text-center text-sm text-muted">No chunk shares a single word with the query.</li>}
      </ol>
    </LabFrame>
  );
}
