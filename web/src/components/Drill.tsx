"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { CheckCircle, Timer, XCircle } from "@phosphor-icons/react";
import { grade, type Question } from "@/lib/quiz-types";
import { MiniMarkdown } from "./MiniMarkdown";
import { QuestionView } from "./quiz/QuizRunner";

export type DrillCard =
  | { id: string; kind: "interview"; lesson: string; lessonTitle: string; q: string; a: string }
  | { id: string; kind: "quiz"; lesson: string; lessonTitle: string; question: Question };
export interface DrillScope { id: string; label: string; lessons: string[]; hidden?: boolean }
type Mode = "interview" | "quiz" | "mixed";

const SHAKY_KEY = "ml-ai-learn:drill:shaky";
const loadShaky = (): string[] => { try { return JSON.parse(localStorage.getItem(SHAKY_KEY) ?? "[]"); } catch { return []; } };
function markShaky(id: string, shaky: boolean) {
  const s = new Set(loadShaky());
  if (shaky) s.add(id); else s.delete(id);
  localStorage.setItem(SHAKY_KEY, JSON.stringify([...s]));
}

const shuffle = <T,>(xs: T[]) => {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
};

const pill = (on: boolean) => `rounded-lg border px-3 py-1.5 text-sm ${on ? "border-accent bg-accent-soft text-ink" : "border-line text-muted hover:text-ink"}`;

export function Drill({ cards, scopes }: { cards: DrillCard[]; scopes: DrillScope[] }) {
  const params = useSearchParams();
  const initial = params.get("lesson") ? `lesson:${params.get("lesson")}` : params.get("sprint") ? `sprint:${params.get("sprint")}` : params.get("module") ? `module:${params.get("module")}` : "all";
  const [scopeId, setScopeId] = useState(scopes.some((s) => s.id === initial) ? initial : "all");
  const [mode, setMode] = useState<Mode>(params.get("lesson") ? "interview" : "mixed");
  const [count, setCount] = useState(10);
  const [seconds, setSeconds] = useState(60);
  // rendered client-only (useSearchParams under Suspense in a static export), so localStorage is available
  const [shaky, setShaky] = useState<string[]>(() => (typeof window === "undefined" ? [] : loadShaky()));
  const [deck, setDeck] = useState<DrillCard[] | null>(null);

  const scope = scopes.find((s) => s.id === scopeId);
  const pool = useMemo(() => {
    const inScope = scopeId === "shaky" ? cards.filter((c) => shaky.includes(c.id)) : cards.filter((c) => scope?.lessons.includes(c.lesson));
    return inScope.filter((c) => mode === "mixed" || c.kind === mode);
  }, [cards, scope, scopeId, mode, shaky]);

  if (deck) return <Session deck={deck} seconds={seconds} onExit={() => { setDeck(null); setShaky(loadShaky()); }} />;

  const visible = scopes.filter((s) => !s.hidden || s.id === scopeId);
  return (
    <div className="mt-8 rounded-xl border border-line bg-surface p-5 space-y-5">
      <label className="block text-sm text-muted">What to drill
        <select value={scopeId} onChange={(e) => setScopeId(e.target.value)} className="mt-1 block w-full rounded-lg border border-line bg-bg px-3 py-2 text-ink">
          {shaky.length > 0 && <option value="shaky">My shaky cards ({shaky.length})</option>}
          {visible.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
      </label>
      <fieldset>
        <legend className="text-sm text-muted mb-1.5">Card type</legend>
        <div className="flex flex-wrap gap-2">
          {([["mixed", "Mixed"], ["interview", "Interview flashcards"], ["quiz", "Quiz blitz"]] as const).map(([m, label]) => (
            <button key={m} type="button" aria-pressed={mode === m} onClick={() => setMode(m)} className={pill(mode === m)}>{label}</button>
          ))}
        </div>
      </fieldset>
      <div className="flex flex-wrap gap-6">
        <fieldset>
          <legend className="text-sm text-muted mb-1.5">Cards</legend>
          <div className="flex gap-2">
            {[10, 20, 0].map((n) => <button key={n} type="button" aria-pressed={count === n} onClick={() => setCount(n)} className={pill(count === n)}>{n || "All"}</button>)}
          </div>
        </fieldset>
        <fieldset>
          <legend className="text-sm text-muted mb-1.5">Time per card</legend>
          <div className="flex gap-2">
            {[30, 60, 90, 0].map((n) => <button key={n} type="button" aria-pressed={seconds === n} onClick={() => setSeconds(n)} className={pill(seconds === n)}>{n ? `${n}s` : "Untimed"}</button>)}
          </div>
        </fieldset>
      </div>
      <div className="flex items-center gap-3">
        <button type="button" disabled={!pool.length} onClick={() => setDeck(shuffle(pool).slice(0, count || pool.length))}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white dark:text-zinc-950 disabled:opacity-40">Start drill</button>
        <span className="text-sm text-muted">{pool.length} cards available</span>
      </div>
    </div>
  );
}

function Session({ deck, seconds, onExit }: { deck: DrillCard[]; seconds: number; onExit: () => void }) {
  const [i, setI] = useState(0);
  const [shown, setShown] = useState(false);
  const [response, setResponse] = useState<unknown>(undefined);
  const [results, setResults] = useState<Record<string, boolean>>({});
  const [left, setLeft] = useState(seconds);
  const card = deck[i];
  const revealed = shown || (seconds > 0 && left <= 0); // running out of time reveals the answer
  const quizOk = card?.kind === "quiz" && grade(card.question, response) === true;

  useEffect(() => {
    if (!seconds || revealed || !card) return;
    const t = setTimeout(() => setLeft((s) => s - 1), 1000);
    return () => clearTimeout(t);
  });

  const record = (ok: boolean) => { setResults((r) => ({ ...r, [card.id]: ok })); markShaky(card.id, !ok); };
  const next = () => {
    if (card.kind === "quiz") record(quizOk);
    setI(i + 1); setShown(false); setResponse(undefined); setLeft(seconds);
  };
  const rate = (ok: boolean) => { record(ok); setI(i + 1); setShown(false); setLeft(seconds); };

  if (!card) {
    const right = deck.filter((c) => results[c.id]).length;
    const missed = deck.filter((c) => results[c.id] === false);
    return (
      <section className="mt-8 rounded-xl border border-line bg-surface p-5" aria-live="polite">
        <h2 className="font-semibold">Drill complete</h2>
        <p className="mt-2 text-3xl font-semibold tracking-tight">{right} / {deck.length}</p>
        {missed.length > 0 && (
          <>
            <p className="mt-3 text-sm text-muted">Shaky or missed (saved to “My shaky cards”). Reread the lesson, then drill them again:</p>
            <ul className="mt-2 space-y-1.5 text-sm">
              {missed.map((c) => (
                <li key={c.id}>
                  <Link href={`/learn/${c.lesson}/`} className="text-accent hover:underline">{c.lessonTitle}</Link>
                  <span className="text-muted">: <MiniMarkdown inline>{c.kind === "interview" ? c.q : c.question.prompt.split("\n")[0]}</MiniMarkdown></span>
                </li>
              ))}
            </ul>
          </>
        )}
        <button type="button" onClick={onExit} className="mt-5 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white dark:text-zinc-950">New drill</button>
      </section>
    );
  }

  const ok = quizOk;
  return (
    <section className="mt-8 rounded-xl border border-line bg-surface" aria-label="Drill card">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-5 py-3 text-sm">
        <span className="text-muted">Card {i + 1} of {deck.length} · <Link href={`/learn/${card.lesson}/`} className="hover:text-ink underline underline-offset-4">{card.lessonTitle}</Link></span>
        <span className="flex items-center gap-3">
          {seconds > 0 && !revealed && (
            <span className={`inline-flex items-center gap-1 font-mono ${left <= 10 ? "text-bad" : "text-muted"}`} aria-label={`${left} seconds left`}><Timer size={14} /> {left}s</span>
          )}
          <button type="button" onClick={onExit} className="text-faint hover:text-ink">End</button>
        </span>
      </header>
      {seconds > 0 && <div className="h-1 bg-surface-2" aria-hidden><div className="h-full bg-accent transition-[width] duration-1000 ease-linear" style={{ width: `${revealed ? 0 : (left / seconds) * 100}%` }} /></div>}
      <div className="p-5 space-y-4">
        <p className="text-xs text-faint uppercase tracking-wide">{card.kind === "interview" ? "Answer out loud, then reveal" : "Quiz blitz"}</p>
        {card.kind === "interview" ? (
          <>
            <div className="prose text-lg font-medium"><MiniMarkdown>{card.q}</MiniMarkdown></div>
            {revealed ? (
              <>
                <div className="rounded-xl border border-line bg-bg p-4 prose text-sm"><MiniMarkdown>{card.a}</MiniMarkdown></div>
                <div className="flex flex-wrap gap-2">
                  <button type="button" onClick={() => rate(true)} className="rounded-lg border border-good/50 px-3 py-1.5 text-sm">I nailed it</button>
                  <button type="button" onClick={() => rate(false)} className="rounded-lg border border-bad/50 px-3 py-1.5 text-sm">Shaky, drill again</button>
                </div>
              </>
            ) : (
              <button type="button" onClick={() => setShown(true)} className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-white dark:text-zinc-950">Reveal model answer</button>
            )}
          </>
        ) : (
          <>
            <div className="prose text-[0.98rem]"><MiniMarkdown>{card.question.prompt}</MiniMarkdown></div>
            <QuestionView q={card.question} response={response} setResponse={setResponse} checked={revealed} />
            {revealed ? (
              <>
                <div className={`rounded-xl border p-4 text-sm ${ok ? "border-good/50 bg-good/5" : "border-bad/50 bg-bad/5"}`}>
                  <p className="flex items-center gap-1.5 font-semibold mb-2">
                    {ok ? <CheckCircle size={18} className="text-good" /> : <XCircle size={18} className="text-bad" />}
                    {ok ? "Correct" : !shown ? "Time's up." : "Not quite."}
                  </p>
                  <div className="prose text-sm"><MiniMarkdown>{card.question.explanation}</MiniMarkdown></div>
                </div>
                <button type="button" onClick={next} className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-white dark:text-zinc-950">Next</button>
              </>
            ) : (
              <button type="button" onClick={() => setShown(true)} className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-white dark:text-zinc-950">Check</button>
            )}
          </>
        )}
      </div>
    </section>
  );
}
