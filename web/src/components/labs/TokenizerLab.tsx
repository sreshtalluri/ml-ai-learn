"use client";
import { useMemo, useState } from "react";
import { CaretLeft, CaretRight } from "@phosphor-icons/react";
import { effectiveMerges, encodeWords, trainBpe, type Merge } from "@/lib/bpe";
import { fmt } from "@/lib/ml";
import { Button, LabFrame, Slider, Stat, type TourStep } from "./ui";

// The tokenizer learns its merges from this text. SYNTHETIC, for illustration only.
const SENTENCES = `the model learns to predict the next token from the previous tokens
training the model on more data makes the predictions better
a transformer model uses attention to weigh the tokens in its context
the learning rate controls how fast the model learns during training
tokenization splits text into tokens and the model predicts tokens
the model is trained to predict the next token in the training data`;
const CORPUS = Array(3).fill(SENTENCES).join("\n"); // 3 copies so frequent pairs stand out (joined by newlines, so no fake words like "datathe")
const MAX_MERGES = 150;
const DEFAULT = "The transformer model predicts tokens. Unbelievable hyperparameters!";
const UNSEEN = "Photosynthesis quantization xylophone";
const LONG = Array(6).fill("the model learns to predict the next token").join(" ");

// One hue per word (not per token), so merging a word never recolours its neighbours.
const HUES = ["var(--c-blue)", "var(--c-teal)", "var(--c-purple)", "var(--c-orange)"];
const mergeLabel = ([a, b]: Merge) => `"${a}" + "${b}" → "${a + b}"`;

export default function TokenizerLab() {
  const merges = useMemo(() => trainBpe(CORPUS, MAX_MERGES), []);
  const [text, setText] = useState(DEFAULT);
  const [k, setK] = useState(40);
  const [ctx, setCtx] = useState(64);
  const [system, setSystem] = useState(20);

  const wordsTok = useMemo(() => encodeWords(text, merges, k), [text, merges, k]);
  const effective = useMemo(() => effectiveMerges(text, merges), [text, merges]);
  const count = (t: string, m: number) => encodeWords(t, merges, m).flat().length;
  const tokens = wordsTok.flat().length;
  const chars = text.length, wordCount = text.split(/\s+/).filter(Boolean).length;
  const used = system + tokens;
  const over = used > ctx;
  const lastEffective = [...effective].reverse().find((e) => e <= k) ?? 0; // merge that last changed the text
  const latest = lastEffective ? merges[lastEffective - 1].join("") : null;
  const current = merges[k - 1];
  const prevEff = [...effective].reverse().find((e) => e < k) ?? 0;
  const nextEff = effective.find((e) => e > k);

  // Guided tour. Numbers are computed from the real merges, so captions always match the screen.
  const effD = effectiveMerges(DEFAULT, merges);
  const firstThree = effD.slice(0, 3).map((e) => mergeLabel(merges[e - 1])).join(", then ");
  const midK = effD[Math.min(9, effD.length - 1)];
  const setup = (t: string, m: number, window = 128, sys = 20) => { setText(t); setK(m); setCtx(window); setSystem(sys); };
  // step through the merges that change the text, one at a time (skipping merges that don't touch it)
  const walk = (list: number[]) => (t: number) => setK(list[Math.min(list.length - 1, Math.floor(t * list.length))]);
  const slide = (set: (v: number) => void, a: number, b: number, step = 1) => (t: number) => set(a + Math.round((t * (b - a)) / step) * step);
  const tour: TourStep[] = [
    { id: "characters", caption: `With zero merges every character is its own token, and ▁ marks the start of a word. This sentence is ${count(DEFAULT, 0)} tokens.`, apply: () => setup(DEFAULT, 0) },
    { id: "corpus", caption: `The tokenizer learned ${merges.length} merges from a small training text (shown below the lab). Each merge joins the pair of symbols that appeared together most often there.`, apply: () => setup(DEFAULT, 0) },
    { id: "merge", caption: `Watch the merges that touch this sentence: ${firstThree}. Each one fuses two neighbouring pieces, and the count falls to ${count(DEFAULT, midK)}.`, apply: () => setup(DEFAULT, 0), animate: walk(effD.filter((e) => e <= midK)), animMs: 4000 },
    { id: "whole-words", caption: `Keep going and words common in the training text, like "transformer" and "predicts", become single tokens: ${count(DEFAULT, merges.length)} tokens after all merges. Only ${effD.length} of the ${merges.length} merges ever changed this sentence.`, apply: () => setup(DEFAULT, midK), animate: walk(effD.filter((e) => e > midK)), animMs: 3500 },
    { id: "unseen", caption: "Words the training text never contained split into many small pieces. Nothing is ever unknown, but rare words cost far more tokens than common ones.", apply: () => setup(UNSEEN, merges.length, 64) },
    { id: "window", caption: "The bar is the context window. The purple system prompt and your teal text share it; as the window shrinks, the text turns red once it no longer fits.", apply: () => setup(LONG, merges.length, 128), animate: slide(setCtx, 128, 48, 16), animMs: 2600 },
    { id: "budget", caption: "A longer system prompt eats the same budget, leaving less room for the answer. Cost and latency follow tokens, not characters.", apply: () => setup(DEFAULT, merges.length, 64, 0), animate: slide(setSystem, 0, 44, 4), animMs: 2600 },
  ];

  return (
    <LabFrame
      id="tokenizer"
      title="Tokenization explorer"
      tour={tour}
      subtitle={`A byte-pair encoder trained in your browser on a six-sentence ML text (${merges.length} merges learned). Real tokenizers learn 50k–200k merges from huge corpora.`}
      onReset={() => setup(DEFAULT, 40, 64)}
      presets={[
        { label: "Characters only (0 merges)", apply: () => setK(0) },
        { label: "All merges", apply: () => setK(merges.length) },
        { label: "Unseen words", apply: () => setText(UNSEEN) },
        { label: "Fill the context", apply: () => setText(LONG) },
      ]}
      controls={
        <>
          <label className="block">
            <span className="text-[0.8rem] text-muted">Your text</span>
            <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} className="mt-1 w-full rounded-lg border border-line bg-bg px-2.5 py-1.5 text-sm" />
          </label>
          <Slider label="merges applied" value={k} min={0} max={merges.length} onChange={setK} />
          <div className="flex items-center gap-2">
            <Button onClick={() => setK(prevEff)} disabled={k === 0}><CaretLeft size={14} /> Prev</Button>
            <Button primary onClick={() => nextEff && setK(nextEff)} disabled={!nextEff}>Next useful merge <CaretRight size={14} /></Button>
          </div>
          <Slider label="context window (tokens)" value={ctx} min={16} max={256} step={16} onChange={setCtx} />
          <Slider label="system prompt (tokens)" value={system} min={0} max={128} step={4} onChange={setSystem} />
        </>
      }
      readout={
        <>
          <Stat label="characters" value={chars} />
          <Stat label="words" value={wordCount} />
          <Stat label="tokens" value={tokens} color="var(--c-teal)" />
          <Stat label="chars per token" value={fmt(chars / Math.max(1, tokens), 2)} />
          <Stat label="merges that change this text" value={`${effective.filter((e) => e <= k).length} of ${effective.length}`} />
          <Stat label="context used" value={`${used} / ${ctx}`} color={over ? "var(--c-red)" : undefined} />
        </>
      }
      interpretation={
        over
          ? `Over the limit by ${used - ctx} tokens: everything past the window is invisible to the model unless you truncate, summarize, or retrieve only what's relevant. Cost and latency also scale with tokens, not characters.`
          : k === 0
            ? "With no merges every character is a token: no word is ever unknown, but sequences are long. Each merge trades a slightly bigger vocabulary for shorter sequences."
            : "Words common in the training text (\"the\", \"model\", \"token\") become single tokens; rare or unseen words stay in pieces. Most merges never touch a given sentence, which is why the count only drops at some steps."
      }
    >
      <p className="mb-2 text-sm text-muted">
        {k === 0 ? "Merge 0: no merges yet, every character is its own token."
          : <>Merge {k} of {merges.length}: <span className="font-mono text-ink">{mergeLabel(current)}</span>{" "}
            {effective.includes(k) ? <span className="text-teal">changes your text (outlined)</span> : <span>doesn&apos;t appear in your text, so nothing changes</span>}</>}
      </p>
      <div className="flex flex-wrap gap-x-2.5 gap-y-2 leading-none" aria-label="Tokens, grouped by word">
        {wordsTok.map((w, wi) => (
          <span key={wi} className="inline-flex overflow-hidden rounded" style={{ boxShadow: `inset 0 -2px 0 ${HUES[wi % 4]}` }}>
            {w.map((t, ti) => (
              <span key={ti}
                className={`px-1.5 py-1 font-mono text-[0.78rem] text-ink ${ti ? "border-l border-[var(--bg)]" : ""} ${latest && t === latest && lastEffective === k ? "outline outline-2 -outline-offset-2 outline-[var(--c-red)]" : ""}`}
                style={{ background: `color-mix(in srgb, ${HUES[wi % 4]} ${ti % 2 ? 26 : 16}%, var(--surface))` }}>
                {t}
              </span>
            ))}
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-faint">Each coloured group is one word; each box inside it is one token. ▁ marks the start of a word (the space before it).</p>

      <div className="mt-6">
        <p className="text-sm font-medium mb-1">Context window</p>
        <div className="flex h-6 w-full overflow-hidden rounded-lg border border-line bg-surface-2" role="img"
          aria-label={`${system} system tokens and ${tokens} text tokens in a ${ctx}-token window`}>
          <div style={{ width: `${Math.min(100, (system / ctx) * 100)}%`, background: "var(--c-purple)" }} />
          <div style={{ width: `${Math.max(0, Math.min(100 - (system / ctx) * 100, (tokens / ctx) * 100))}%`, background: over ? "var(--c-red)" : "var(--c-teal)" }} />
        </div>
        <div className="mt-1 flex gap-4 text-xs text-muted">
          <span className="inline-flex items-center gap-1"><span className="h-2 w-3 rounded-sm" style={{ background: "var(--c-purple)" }} /> system prompt</span>
          <span className="inline-flex items-center gap-1"><span className="h-2 w-3 rounded-sm" style={{ background: over ? "var(--c-red)" : "var(--c-teal)" }} /> your text</span>
          <span>{Math.max(0, ctx - used)} tokens left for the answer</span>
        </div>
      </div>

      <details className="mt-5 rounded-lg border border-line px-3 py-2 text-sm">
        <summary className="cursor-pointer text-muted">Where the merges come from: the training text, and the merges that change your text</summary>
        <p className="mt-2 text-xs text-faint">The tokenizer counted every adjacent pair of symbols in this text (repeated 3 times) and merged the most frequent pair, {merges.length} times.</p>
        <pre className="mt-1 whitespace-pre-wrap font-mono text-[0.75rem] text-muted">{SENTENCES}</pre>
        <p className="mt-3 text-xs text-faint">Merges that change your text ({effective.length} of {merges.length}). Click one to jump to it:</p>
        <div className="mt-1 flex flex-wrap gap-1">
          {effective.map((e) => (
            <button key={e} type="button" onClick={() => setK(e)}
              className={`rounded border px-1.5 py-0.5 font-mono text-[0.72rem] ${e === k ? "border-accent bg-accent-soft text-ink" : e < k ? "border-line text-ink" : "border-dashed border-line text-faint"}`}>
              {e}: {merges[e - 1][0]}+{merges[e - 1][1]}
            </button>
          ))}
        </div>
      </details>
    </LabFrame>
  );
}
