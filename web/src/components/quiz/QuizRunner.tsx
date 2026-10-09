"use client";
import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, CheckCircle, XCircle } from "@phosphor-icons/react";
import { grade, type Question, type Quiz } from "@/lib/quiz-types";
import { actions, useProgress } from "@/lib/progress";
import { MiniMarkdown } from "../MiniMarkdown";

/** Deterministic shuffle keyed by a string, so server and client render the same order. */
export function shuffled(n: number, key: string): number[] {
  let h = 2166136261;
  for (const c of key) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    const j = Math.abs(h) % (i + 1);
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  // never present an "order" question already solved
  if (n > 1 && idx.every((v, i) => v === i)) idx.push(idx.shift()!);
  return idx;
}

const DIFF: Record<Question["difficulty"], string> = {
  easy: "text-good border-good/40",
  medium: "text-orange border-orange/40",
  hard: "text-bad border-bad/40",
};

const TYPE_LABEL: Record<Question["type"], string> = {
  single: "Multiple choice", multi: "Select all that apply", numeric: "Calculation", fill: "Fill in",
  order: "Arrange the steps", match: "Match the concepts", short: "Reflection",
};

export function QuestionView({ q, response, setResponse, checked }: {
  q: Question; response: unknown; setResponse: (r: unknown) => void; checked: boolean;
}) {
  const opt = "w-full text-left rounded-lg border px-3 py-2 text-sm transition-colors";
  switch (q.type) {
    case "single":
    case "multi": {
      const sel = q.type === "single" ? [response] : ((response as number[]) ?? []);
      const correct = q.type === "single" ? [q.answer] : (q.answer as number[]);
      return (
        <div className="space-y-2" role={q.type === "single" ? "radiogroup" : "group"}>
          {q.options!.map((o, i) => {
            const chosen = sel.includes(i);
            const isRight = correct.includes(i);
            const tone = checked ? (isRight ? "border-good bg-good/10" : chosen ? "border-bad bg-bad/10" : "border-line") : chosen ? "border-accent bg-accent-soft" : "border-line hover:border-faint";
            return (
              <div key={i}>
                <button
                  type="button"
                  role={q.type === "single" ? "radio" : "checkbox"}
                  aria-checked={chosen}
                  disabled={checked}
                  onClick={() => setResponse(q.type === "single" ? i : chosen ? sel.filter((s) => s !== i) : [...(sel as number[]), i])}
                  className={`${opt} ${tone}`}
                >
                  <MiniMarkdown inline>{o}</MiniMarkdown>
                </button>
                {checked && q.why_not?.[i] && (chosen || isRight) && (
                  <p className="text-xs text-muted mt-1 ml-3"><MiniMarkdown inline>{q.why_not[i]}</MiniMarkdown></p>
                )}
              </div>
            );
          })}
        </div>
      );
    }
    case "numeric":
    case "fill":
      return (
        <input
          type="text"
          inputMode={q.type === "numeric" ? "decimal" : "text"}
          aria-label="Your answer"
          disabled={checked}
          value={(response as string) ?? ""}
          onChange={(e) => setResponse(e.target.value)}
          placeholder={q.type === "numeric" ? "e.g. 0.75" : "Type your answer"}
          className="w-full max-w-xs rounded-lg border border-line bg-bg px-3 py-2 font-mono text-sm"
        />
      );
    case "order": {
      const order = (response as number[]) ?? shuffled(q.options!.length, q.id);
      const move = (i: number, d: number) => {
        const next = [...order];
        [next[i], next[i + d]] = [next[i + d], next[i]];
        setResponse(next);
      };
      return (
        <ol className="space-y-2">
          {order.map((o, i) => (
            <li key={o} className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm ${checked ? (o === i ? "border-good bg-good/10" : "border-bad bg-bad/10") : "border-line"}`}>
              <span className="font-mono text-faint w-5">{i + 1}.</span>
              <span className="flex-1"><MiniMarkdown inline>{q.options![o]}</MiniMarkdown></span>
              {!checked && (
                <span className="flex gap-1">
                  <button type="button" aria-label="Move up" disabled={i === 0} onClick={() => move(i, -1)} className="rounded p-1 hover:bg-surface-2 disabled:opacity-30"><ArrowUp size={14} /></button>
                  <button type="button" aria-label="Move down" disabled={i === order.length - 1} onClick={() => move(i, 1)} className="rounded p-1 hover:bg-surface-2 disabled:opacity-30"><ArrowDown size={14} /></button>
                </span>
              )}
            </li>
          ))}
        </ol>
      );
    }
    case "match": {
      const rights = shuffled(q.pairs!.length, q.id + "r");
      const sel = (response as number[]) ?? q.pairs!.map(() => -1);
      return (
        <div className="space-y-2">
          {q.pairs!.map(([left], i) => (
            <label key={i} className={`grid sm:grid-cols-2 gap-2 items-center rounded-lg border px-3 py-2 text-sm ${checked ? (sel[i] === i ? "border-good bg-good/10" : "border-bad bg-bad/10") : "border-line"}`}>
              <span><MiniMarkdown inline>{left}</MiniMarkdown></span>
              <select
                disabled={checked}
                value={sel[i]}
                onChange={(e) => setResponse(sel.map((v, j) => (j === i ? Number(e.target.value) : v)))}
                className="rounded-lg border border-line bg-bg px-2 py-1.5"
              >
                <option value={-1}>Choose…</option>
                {rights.map((r) => <option key={r} value={r}>{q.pairs![r][1]}</option>)}
              </select>
            </label>
          ))}
          {checked && (
            <p className="text-xs text-muted">Correct pairs: {q.pairs!.map(([l, r]) => `${l} → ${r}`).join("; ")}</p>
          )}
        </div>
      );
    }
    case "short":
      return (
        <textarea
          aria-label="Your answer"
          rows={4}
          value={(response as string) ?? ""}
          onChange={(e) => setResponse(e.target.value)}
          placeholder="Write your answer in your own words, then compare with the model answer."
          className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm"
        />
      );
  }
}

export function QuizRunner({ quiz, onlyIds }: { quiz: Quiz; onlyIds?: string[] }) {
  const progress = useProgress();
  const [subset, setSubset] = useState<string[] | null>(onlyIds ?? null);
  const questions = useMemo(() => (subset ? quiz.questions.filter((q) => subset.includes(q.id)) : quiz.questions), [quiz, subset]);
  const [i, setI] = useState(0);
  const [responses, setResponses] = useState<Record<string, unknown>>({});
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const [results, setResults] = useState<Record<string, boolean>>({});
  const [done, setDone] = useState(false);

  const q = questions[i];
  const record = progress.quizzes[quiz.id];

  const restart = (ids: string[] | null) => {
    setSubset(ids); setI(0); setResponses({}); setChecked({}); setResults({}); setDone(false);
  };

  const check = () => {
    const r = grade(q, q.type === "order" && responses[q.id] === undefined ? shuffled(q.options!.length, q.id) : responses[q.id]);
    setChecked((c) => ({ ...c, [q.id]: true }));
    if (r !== null) setResults((s) => ({ ...s, [q.id]: r }));
  };

  const finish = (final: Record<string, boolean>) => {
    const graded = questions.filter((x) => final[x.id] !== undefined);
    const correct = graded.filter((x) => final[x.id]).length;
    const wrong = graded.filter((x) => !final[x.id]).map((x) => x.id);
    if (!subset) actions.recordQuiz(quiz.id, correct, graded.length, wrong);
    // review rounds only shrink the review queue; they don't count as a quiz attempt
    else actions.setReviewQueue(quiz.id, (record?.wrong ?? []).filter((id) => final[id] !== true));
    setDone(true);
  };

  const selfGrade = (ok: boolean) => {
    const nr = { ...results, [q.id]: ok };
    setResults(nr);
    if (i + 1 < questions.length) setI(i + 1);
    else finish(nr);
  };

  if (!questions.length) return <p className="text-muted">No questions to review. Nice work.</p>;

  if (done) {
    const graded = Object.keys(results).length;
    const correct = Object.values(results).filter(Boolean).length;
    const wrongIds = Object.keys(results).filter((k) => !results[k]);
    return (
      <section className="not-prose my-8 rounded-xl border border-line bg-surface p-5" aria-live="polite">
        <h3 className="font-semibold">{quiz.title}: results</h3>
        <p className="mt-2 text-3xl font-semibold tracking-tight">{correct} / {graded}</p>
        <p className="text-sm text-muted mt-1">
          {graded === 0 ? "Only reflection questions this round." : correct === graded ? "Everything correct. Come back in a few days to check it stuck." : "Wrong answers are saved to your review queue on the Quizzes page."}
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          {wrongIds.length > 0 && <button type="button" onClick={() => restart(wrongIds)} className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white dark:text-zinc-950">Retry incorrect ({wrongIds.length})</button>}
          <button type="button" onClick={() => restart(null)} className="rounded-lg border border-line px-3 py-1.5 text-sm">Retake full quiz</button>
        </div>
      </section>
    );
  }

  const isChecked = !!checked[q.id];
  const result = results[q.id];
  const hasResponse = q.type === "order" || q.type === "short" || (responses[q.id] !== undefined && responses[q.id] !== "" && !(Array.isArray(responses[q.id]) && (responses[q.id] as unknown[]).length === 0));

  return (
    <section aria-label={`Quiz: ${quiz.title}`} className="not-prose my-8 rounded-xl border border-line bg-surface">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-5 py-3">
        <div>
          <h3 className="font-semibold">{subset ? "Review: " : "Quiz: "}{quiz.title}</h3>
          {record && !subset && <p className="text-xs text-faint">Best so far {Math.round(record.best * 100)}% over {record.attempts} attempt{record.attempts > 1 ? "s" : ""}</p>}
        </div>
        <div className="flex items-center gap-3 text-xs text-muted">
          <span>Question {i + 1} of {questions.length}</span>
          <div className="h-1.5 w-24 rounded-full bg-surface-2 overflow-hidden" aria-hidden>
            <div className="h-full bg-accent" style={{ width: `${((i + (isChecked ? 1 : 0)) / questions.length) * 100}%` }} />
          </div>
        </div>
      </header>
      <div className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded-full border border-line px-2 py-0.5 text-muted">{TYPE_LABEL[q.type]}</span>
          <span className={`rounded-full border px-2 py-0.5 ${DIFF[q.difficulty]}`}>{q.difficulty}</span>
        </div>
        <div className="prose text-[0.98rem]"><MiniMarkdown>{q.prompt}</MiniMarkdown></div>
        <QuestionView q={q} response={responses[q.id]} setResponse={(r) => setResponses((s) => ({ ...s, [q.id]: r }))} checked={isChecked} />

        {isChecked && (
          <div className={`rounded-xl border p-4 text-sm ${q.type === "short" ? "border-line" : result ? "border-good/50 bg-good/5" : "border-bad/50 bg-bad/5"}`}>
            {q.type !== "short" && (
              <p className="flex items-center gap-1.5 font-semibold mb-2">
                {result ? <CheckCircle size={18} className="text-good" /> : <XCircle size={18} className="text-bad" />}
                {result ? "Correct" : q.type === "numeric" ? `Not quite. The answer is ${q.answer}${q.tolerance ? ` (±${q.tolerance})` : ""}.` : q.type === "fill" ? `Not quite. Accepted: ${(q.answer as string[]).join(", ")}` : "Not quite."}
              </p>
            )}
            {q.type === "short" && q.sample_answer && (
              <div className="mb-3"><p className="font-semibold mb-1">Model answer</p><div className="prose text-sm"><MiniMarkdown>{q.sample_answer}</MiniMarkdown></div></div>
            )}
            <div className="prose text-sm"><MiniMarkdown>{q.explanation}</MiniMarkdown></div>
            {q.type === "short" && (
              <div className="mt-3 flex gap-2">
                <button type="button" onClick={() => selfGrade(true)} className="rounded-lg border border-good/50 px-3 py-1.5 text-sm">My answer covered this</button>
                <button type="button" onClick={() => selfGrade(false)} className="rounded-lg border border-bad/50 px-3 py-1.5 text-sm">Add to review</button>
              </div>
            )}
          </div>
        )}

        <div className="flex gap-2">
          {!isChecked && (
            <button type="button" disabled={!hasResponse} onClick={check} className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-white dark:text-zinc-950 disabled:opacity-40">
              {q.type === "short" ? "Show model answer" : "Check"}
            </button>
          )}
          {isChecked && q.type !== "short" && (
            <button type="button" onClick={() => (i + 1 < questions.length ? setI(i + 1) : finish(results))} className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-white dark:text-zinc-950">
              {i + 1 < questions.length ? "Next question" : "See results"}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
