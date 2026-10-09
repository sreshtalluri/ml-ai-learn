"use client";
import { useMemo, useState } from "react";
import { encode, trainBpe } from "@/lib/bpe";
import { fmt } from "@/lib/ml";
import { LabFrame, Slider, Stat, type TourStep } from "./ui";

// Training corpus: a short paragraph about ML, repeated with variations. SYNTHETIC, for illustration only.
const CORPUS = `the model learns to predict the next token from the previous tokens
training the model on more data makes the predictions better
a transformer model uses attention to weigh the tokens in its context
the learning rate controls how fast the model learns during training
tokenization splits text into tokens and the model predicts tokens
the model is trained to predict the next token in the training data`.repeat(3);

const HUES = ["var(--c-blue)", "var(--c-teal)", "var(--c-purple)", "var(--c-orange)"];
const MAX_MERGES = 150;

export default function TokenizerLab() {
  const merges = useMemo(() => trainBpe(CORPUS, MAX_MERGES), []);
  const [text, setText] = useState("The transformer model predicts tokens. Unbelievable hyperparameters!");
  const [k, setK] = useState(40);
  const [ctx, setCtx] = useState(64);
  const [system, setSystem] = useState(20);
  const toks = useMemo(() => encode(text.replace(/[^\p{L}\p{N}\s]/gu, " $& "), merges, k), [text, merges, k]);
  const chars = text.length, wordCount = text.split(/\s+/).filter(Boolean).length;
  const used = system + toks.length;
  const over = used > ctx;
  const lastMerge = merges[k - 1];

  // Guided tour (Watch mode + explainers). Each step sets every control, then slides one.
  const DEFAULT = "The transformer model predicts tokens. Unbelievable hyperparameters!";
  const setup = (t: string, m: number, window = 64, sys = 20) => { setText(t); setK(m); setCtx(window); setSystem(sys); };
  const slide = (set: (v: number) => void, a: number, b: number, step = 1) => (t: number) => set(a + Math.round((t * (b - a)) / step) * step);
  const tour: TourStep[] = [
    { id: "characters", caption: "With zero merges every character is its own token, and ▁ marks the start of a word. This one sentence costs 71 tokens.", apply: () => setup(DEFAULT, 0) },
    { id: "merge", caption: "Each merge fuses the most frequent neighbouring pair in the training text: ▁ with t, then h with e, then ▁t with he. Watch the token count fall from 71 to 44.", apply: () => setup(DEFAULT, 0), animate: slide(setK, 0, 40), animMs: 3000 },
    { id: "whole-words", caption: "Keep merging and words common in the corpus, like transformer and predicts, become single tokens. All 129 merges bring the sentence down to 35.", apply: () => setup(DEFAULT, 40), animate: slide(setK, 40, merges.length), animMs: 2600 },
    { id: "unseen", caption: "Words the tokenizer never saw split into many small pieces. Nothing is ever unknown, but rare words cost far more tokens than common ones.", apply: () => setup("Photosynthesis quantization xylophone", merges.length) },
    { id: "window", caption: "The bar is the context window. The purple system prompt and your teal text share it, and as the window shrinks the text turns red once it no longer fits.", apply: () => setup(Array(6).fill("the model learns to predict the next token").join(" "), merges.length, 128), animate: slide(setCtx, 128, 48, 16), animMs: 2600 },
    { id: "budget", caption: "A longer system prompt eats the same budget, leaving less room for the answer. Cost and latency follow tokens, not characters.", apply: () => setup(DEFAULT, merges.length, 64, 0), animate: slide(setSystem, 0, 44, 4), animMs: 2600 },
  ];

  return (
    <LabFrame
      id="tokenizer"
      title="Tokenization explorer"
      tour={tour}
      subtitle={`A byte-pair encoder trained in your browser on a tiny ML corpus (${merges.length} merges learned). Real tokenizers learn ~50k–200k merges from huge corpora.`}
      onReset={() => { setText("The transformer model predicts tokens. Unbelievable hyperparameters!"); setK(40); setCtx(64); setSystem(20); }}
      presets={[
        { label: "Characters only (0 merges)", apply: () => setK(0) },
        { label: "All merges", apply: () => setK(merges.length) },
        { label: "Unseen words", apply: () => setText("Photosynthesis quantization xylophone") },
        { label: "Fill the context", apply: () => setText(Array(6).fill("the model learns to predict the next token").join(" ")) },
      ]}
      controls={
        <>
          <label className="block">
            <span className="text-[0.8rem] text-muted">Your text</span>
            <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} className="mt-1 w-full rounded-lg border border-line bg-bg px-2.5 py-1.5 text-sm" />
          </label>
          <Slider label="merges applied" value={k} min={0} max={merges.length} onChange={setK} hint={lastMerge ? `Merge ${k}: "${lastMerge[0]}" + "${lastMerge[1]}" → "${lastMerge.join("")}"` : "Start: every character is its own token."} />
          <Slider label="context window (tokens)" value={ctx} min={16} max={256} step={16} onChange={setCtx} />
          <Slider label="system prompt (tokens)" value={system} min={0} max={128} step={4} onChange={setSystem} />
        </>
      }
      readout={
        <>
          <Stat label="characters" value={chars} />
          <Stat label="words" value={wordCount} />
          <Stat label="tokens" value={toks.length} color="var(--c-teal)" />
          <Stat label="chars per token" value={fmt(chars / Math.max(1, toks.length), 2)} />
          <Stat label="context used" value={`${used} / ${ctx}`} color={over ? "var(--c-red)" : undefined} />
        </>
      }
      interpretation={
        over
          ? `Over the limit by ${used - ctx} tokens: everything past the window is invisible to the model unless you truncate, summarize, or retrieve only what's relevant. Cost and latency also scale with tokens, not characters.`
          : k === 0
            ? "With no merges every character is a token: no word is ever unknown, but sequences are long. Each merge trades a slightly bigger vocabulary for shorter sequences."
            : "Words common in the training corpus (\"the\", \"model\", \"token\") become single tokens; rare or unseen words split into pieces. The ▁ marks the start of a word."
      }
    >
      <div className="flex flex-wrap gap-1 leading-none" aria-label="Tokens">
        {toks.map((t, i) => (
          <span key={i} className="rounded px-1.5 py-1 font-mono text-[0.78rem] text-ink"
            style={{ background: `color-mix(in srgb, ${HUES[i % 4]} 20%, var(--surface))`, borderBottom: `2px solid ${HUES[i % 4]}` }}>
            {t}
          </span>
        ))}
      </div>
      <div className="mt-6">
        <p className="text-sm font-medium mb-1">Context window</p>
        <div className="flex h-6 w-full overflow-hidden rounded-lg border border-line bg-surface-2" role="img"
          aria-label={`${system} system tokens and ${toks.length} text tokens in a ${ctx}-token window`}>
          <div style={{ width: `${Math.min(100, (system / ctx) * 100)}%`, background: "var(--c-purple)" }} />
          <div style={{ width: `${Math.max(0, Math.min(100 - (system / ctx) * 100, (toks.length / ctx) * 100))}%`, background: over ? "var(--c-red)" : "var(--c-teal)" }} />
        </div>
        <div className="mt-1 flex gap-4 text-xs text-muted">
          <span className="inline-flex items-center gap-1"><span className="h-2 w-3 rounded-sm" style={{ background: "var(--c-purple)" }} /> system prompt</span>
          <span className="inline-flex items-center gap-1"><span className="h-2 w-3 rounded-sm" style={{ background: over ? "var(--c-red)" : "var(--c-teal)" }} /> your text</span>
          <span>{Math.max(0, ctx - used)} tokens left for the answer</span>
        </div>
      </div>
    </LabFrame>
  );
}
