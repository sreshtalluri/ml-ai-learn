"use client";
import Link from "next/link";
import { useRef, useState } from "react";
import { DownloadSimple, Trash, UploadSimple } from "@phosphor-icons/react";
import type { LessonIndex } from "@/lib/content";
import { actions, streak, useHydrated, useProgress } from "@/lib/progress";
import { PROJECTS } from "@/lib/projects";
import { SkillBars } from "./ProgressWidgets";

type Mod = { id: string; number: number; title: string; slugs: string[] };

export function ProgressDashboard({ lessons, modules, labCount, quizzes }: {
  lessons: LessonIndex; modules: Mod[]; labCount: number; quizzes: { id: string; title: string; lesson: string }[];
}) {
  const p = useProgress();
  const hydrated = useHydrated();
  const fileRef = useRef<HTMLInputElement>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);

  if (!hydrated) return <div className="mt-10 h-64 rounded-xl border border-line bg-surface animate-pulse" aria-hidden />;

  const done = lessons.filter((l) => p.lessons[l.slug]).length;
  const attempted = quizzes.filter((q) => p.quizzes[q.id]);
  const avg = attempted.length ? attempted.reduce((s, q) => s + p.quizzes[q.id].best, 0) / attempted.length : 0;
  const modulesDone = modules.filter((m) => m.slugs.length && m.slugs.every((s) => p.lessons[s])).length;
  const needsReview = quizzes.filter((q) => p.quizzes[q.id] && (p.quizzes[q.id].best < 0.7 || p.quizzes[q.id].wrong.length > 0));
  const quizLesson = Object.fromEntries(quizzes.map((q) => [q.id, q.lesson]));
  const title = (slug: string) => lessons.find((l) => l.slug === slug)?.title ?? slug;

  const download = () => {
    const blob = new Blob([actions.exportJSON()], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `ml-ai-learn-progress-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    setMsg("Progress exported.");
  };

  const upload = async (f: File) => {
    try {
      actions.importJSON(await f.text());
      setMsg("Progress imported.");
    } catch (e) {
      setMsg(`Import failed: ${(e as Error).message}`);
    }
  };

  const stats = [
    { label: "Lessons completed", value: `${done} / ${lessons.length}` },
    { label: "Modules completed", value: `${modulesDone} / ${modules.length}` },
    { label: "Labs explored", value: `${Object.keys(p.labs).length} / ${labCount}` },
    { label: "Average best quiz score", value: attempted.length ? `${Math.round(avg * 100)}%` : "No quizzes yet" },
    { label: "Current streak", value: `${streak(p.days)} day${streak(p.days) === 1 ? "" : "s"}` },
  ];

  return (
    <div className="mt-10 space-y-12">
      <section aria-label="Summary" className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        {stats.map((s) => (
          <div key={s.label} className="rounded-xl border border-line bg-surface p-4">
            <p className="text-xs text-muted">{s.label}</p>
            <p className="text-xl font-semibold mt-1 tracking-tight">{s.value}</p>
          </div>
        ))}
      </section>

      <div className="grid gap-10 lg:grid-cols-2">
        <section aria-labelledby="mastery">
          <h2 id="mastery" className="text-xl font-semibold tracking-tight">Mastery by skill area</h2>
          <p className="text-sm text-muted mt-1 mb-4">70% lessons completed, 30% best quiz scores.</p>
          <SkillBars lessons={lessons} quizLesson={quizLesson} />
        </section>
        <section aria-labelledby="review">
          <h2 id="review" className="text-xl font-semibold tracking-tight">Topics needing review</h2>
          {needsReview.length === 0 ? (
            <p className="mt-2 text-sm text-muted">Nothing flagged. A topic appears here when its best quiz score is under 70% or it has questions in the review queue.</p>
          ) : (
            <ul className="mt-3 space-y-2">
              {needsReview.map((q) => (
                <li key={q.id} className="flex items-center justify-between gap-3 rounded-lg border border-line bg-surface px-3 py-2 text-sm">
                  <Link href={`/learn/${q.lesson}/`} className="hover:text-accent">{q.title}</Link>
                  <span className="flex items-center gap-3 text-xs text-faint">
                    best {Math.round(p.quizzes[q.id].best * 100)}%
                    {p.quizzes[q.id].wrong.length > 0 && <Link href={`/quizzes/${q.id}/?review=1`} className="text-accent hover:underline">review {p.quizzes[q.id].wrong.length}</Link>}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <section aria-labelledby="modules">
        <h2 id="modules" className="text-xl font-semibold tracking-tight">Modules</h2>
        <div className="mt-4 grid sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {modules.map((m) => {
            const d = m.slugs.filter((s) => p.lessons[s]).length;
            return (
              <Link key={m.id} href={`/modules/${m.id}/`} className="rounded-lg border border-line bg-surface px-3 py-2 hover:border-faint">
                <span className="flex justify-between text-sm"><span className="truncate pr-2">{m.number}. {m.title}</span><span className="font-mono text-xs text-faint">{d}/{m.slugs.length}</span></span>
                <span className="mt-1.5 block h-1 rounded-full bg-surface-2 overflow-hidden"><span className="block h-full bg-accent" style={{ width: `${m.slugs.length ? (d / m.slugs.length) * 100 : 0}%` }} /></span>
              </Link>
            );
          })}
        </div>
      </section>

      <section aria-labelledby="projects">
        <h2 id="projects" className="text-xl font-semibold tracking-tight">Project milestones</h2>
        <p className="text-sm text-muted mt-1">The portfolio ladder from <Link href="/learn/project-ladder/" className="text-accent hover:underline">Module 23</Link>. Tick milestones as you ship them.</p>
        <div className="mt-4 grid md:grid-cols-2 gap-3">
          {PROJECTS.map((pr, i) => (
            <fieldset key={pr.id} className="rounded-xl border border-line bg-surface p-4">
              <legend className="sr-only">{pr.title}</legend>
              <p className="font-medium text-sm"><span className="font-mono text-faint mr-1.5">{i + 1}</span>{pr.title}</p>
              <ul className="mt-2 space-y-1.5">
                {pr.milestones.map((ms, j) => {
                  const key = `${pr.id}/${j}`;
                  return (
                    <li key={key}>
                      <label className="flex items-center gap-2 text-sm text-muted cursor-pointer">
                        <input type="checkbox" checked={!!p.projects[key]} onChange={() => actions.toggleProject(key)} className="accent-[var(--accent)]" />
                        <span className={p.projects[key] ? "line-through" : ""}>{ms}</span>
                      </label>
                    </li>
                  );
                })}
              </ul>
            </fieldset>
          ))}
        </div>
      </section>

      <div className="grid gap-10 lg:grid-cols-2">
        <section aria-labelledby="bookmarks">
          <h2 id="bookmarks" className="text-xl font-semibold tracking-tight">Bookmarks</h2>
          {p.bookmarks.length === 0 ? <p className="mt-2 text-sm text-muted">No bookmarks yet. Use the Bookmark button on any lesson, model card, or cheat sheet.</p> : (
            <ul className="mt-3 space-y-1.5">{p.bookmarks.map((b) => <li key={b.path}><Link href={b.path} className="text-sm text-accent hover:underline">{b.title}</Link></li>)}</ul>
          )}
        </section>
        <section aria-labelledby="notes">
          <h2 id="notes" className="text-xl font-semibold tracking-tight">Notes</h2>
          {Object.keys(p.notes).length === 0 ? <p className="mt-2 text-sm text-muted">No notes yet. Each lesson has a personal note box at the bottom.</p> : (
            <ul className="mt-3 space-y-3">
              {Object.entries(p.notes).map(([path, note]) => (
                <li key={path} className="rounded-lg border border-line bg-surface px-3 py-2">
                  <Link href={path} className="text-sm font-medium hover:text-accent">{path.startsWith("/learn/") ? title(path.split("/")[2]) : path}</Link>
                  <p className="text-sm text-muted whitespace-pre-wrap mt-1">{note}</p>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <section aria-labelledby="data" className="rounded-xl border border-line bg-surface p-5">
        <h2 id="data" className="text-lg font-semibold tracking-tight">Your data</h2>
        <p className="text-sm text-muted mt-1">Progress lives only in this browser&apos;s local storage. Export it to move between devices or keep a backup.</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <button type="button" onClick={download} className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-sm hover:bg-surface-2"><DownloadSimple size={16} /> Export JSON</button>
          <button type="button" onClick={() => fileRef.current?.click()} className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-sm hover:bg-surface-2"><UploadSimple size={16} /> Import JSON</button>
          <input ref={fileRef} type="file" accept="application/json,.json" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f); e.target.value = ""; }} />
          {!confirmReset ? (
            <button type="button" onClick={() => setConfirmReset(true)} className="inline-flex items-center gap-1.5 rounded-lg border border-bad/40 px-3 py-1.5 text-sm text-bad hover:bg-bad/5"><Trash size={16} /> Reset progress</button>
          ) : (
            <span className="inline-flex items-center gap-2 text-sm">
              Erase everything?
              <button type="button" onClick={() => { actions.reset(); setConfirmReset(false); setMsg("Progress reset."); }} className="rounded-lg bg-bad px-3 py-1.5 text-white">Yes, reset</button>
              <button type="button" onClick={() => setConfirmReset(false)} className="rounded-lg border border-line px-3 py-1.5">Cancel</button>
            </span>
          )}
        </div>
        {msg && <p className="mt-3 text-sm" role="status">{msg}</p>}
      </section>
    </div>
  );
}
