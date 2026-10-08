"use client";
import Link from "next/link";
import { useHydrated, useProgress } from "@/lib/progress";
import { MiniMarkdown } from "../MiniMarkdown";

type Item = { id: string; title: string; count: number; lessonTitle?: string; questions: { id: string; prompt: string }[] };

export function QuizIndex({ quizzes }: { quizzes: Item[] }) {
  const p = useProgress();
  const hydrated = useHydrated();
  const review = quizzes.flatMap((q) => (p.quizzes[q.id]?.wrong ?? []).map((wid) => ({ quiz: q, question: q.questions.find((x) => x.id === wid) })).filter((r) => r.question));
  const attempted = quizzes.filter((q) => p.quizzes[q.id]);
  const mastery = attempted.length ? attempted.reduce((s, q) => s + p.quizzes[q.id].best, 0) / attempted.length : 0;

  return (
    <>
      <section className="mt-10 grid gap-4 sm:grid-cols-3" aria-label="Quiz stats">
        <div className="rounded-xl border border-line bg-surface p-4"><p className="text-sm text-muted">Quizzes attempted</p><p className="text-2xl font-semibold mt-1">{hydrated ? attempted.length : "–"} / {quizzes.length}</p></div>
        <div className="rounded-xl border border-line bg-surface p-4"><p className="text-sm text-muted">Average best score</p><p className="text-2xl font-semibold mt-1">{hydrated && attempted.length ? `${Math.round(mastery * 100)}%` : "–"}</p></div>
        <div className="rounded-xl border border-line bg-surface p-4"><p className="text-sm text-muted">In review queue</p><p className="text-2xl font-semibold mt-1">{hydrated ? review.length : "–"}</p></div>
      </section>

      <section className="mt-10" aria-labelledby="review">
        <h2 id="review" className="text-xl font-semibold tracking-tight">Review queue</h2>
        {!hydrated ? null : review.length === 0 ? (
          <p className="mt-2 text-sm text-muted">Empty. Questions you miss show up here so you can retry just those.</p>
        ) : (
          <ul className="mt-3 rounded-xl border border-line bg-surface divide-y divide-line">
            {review.slice(0, 12).map(({ quiz, question }) => (
              <li key={quiz.id + question!.id} className="px-4 py-3 text-sm flex gap-3 items-start">
                <span className="flex-1 min-w-0 line-clamp-2"><MiniMarkdown inline>{question!.prompt}</MiniMarkdown></span>
                <Link href={`/quizzes/${quiz.id}/?review=1`} className="shrink-0 text-accent hover:underline">Retry in {quiz.title}</Link>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="mt-10" aria-labelledby="all">
        <h2 id="all" className="text-xl font-semibold tracking-tight">All quizzes</h2>
        <div className="mt-3 grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {quizzes.map((q) => {
            const r = p.quizzes[q.id];
            return (
              <Link key={q.id} href={`/quizzes/${q.id}/`} className="group rounded-xl border border-line bg-surface p-4 hover:border-faint">
                <span className="font-semibold group-hover:text-accent">{q.title}</span>
                <span className="block text-xs text-faint mt-1">{q.count} questions{q.lessonTitle ? ` · ${q.lessonTitle}` : ""}</span>
                {hydrated && r && <span className="block text-xs mt-2 text-muted">Best {Math.round(r.best * 100)}% · {r.attempts} attempt{r.attempts > 1 ? "s" : ""}</span>}
              </Link>
            );
          })}
        </div>
      </section>
    </>
  );
}
