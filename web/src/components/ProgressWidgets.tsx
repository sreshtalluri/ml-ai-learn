"use client";
import Link from "next/link";
import { useHydrated, useProgress } from "@/lib/progress";
import { SKILL_KEYS, SKILLS } from "@/lib/skills";
import type { LessonIndex } from "@/lib/content";

export function ModuleProgress({ slugs }: { slugs: string[] }) {
  const p = useProgress();
  const done = slugs.filter((s) => p.lessons[s]).length;
  return <span className="text-xs text-faint font-mono">{done}/{slugs.length} done</span>;
}

/** Mastery per skill area: 70% lessons completed + 30% best quiz scores in that area. */
export function skillMastery(lessons: LessonIndex, p: ReturnType<typeof useProgress>, quizLesson: Record<string, string>) {
  return SKILL_KEYS.map((k) => {
    const mine = lessons.filter((l) => l.skill === k);
    const done = mine.filter((l) => p.lessons[l.slug]).length;
    const scores = Object.entries(p.quizzes).filter(([id]) => mine.some((l) => l.slug === (quizLesson[id] ?? id))).map(([, r]) => r.best);
    const quiz = mine.length ? scores.reduce((s, v) => s + v, 0) / mine.length : 0;
    const mastery = mine.length ? 0.7 * (done / mine.length) + 0.3 * quiz : 0;
    return { key: k, label: SKILLS[k].label, color: SKILLS[k].color, total: mine.length, done, mastery };
  });
}

export function SkillBars({ lessons, quizLesson }: { lessons: LessonIndex; quizLesson: Record<string, string> }) {
  const p = useProgress();
  const rows = skillMastery(lessons, p, quizLesson);
  return (
    <ul className="space-y-3">
      {rows.map((r) => (
        <li key={r.key}>
          <div className="flex justify-between text-sm">
            <span>{r.label}</span>
            <span className="font-mono text-xs text-faint">{r.done}/{r.total} lessons · {Math.round(r.mastery * 100)}%</span>
          </div>
          <div className="mt-1 h-1.5 rounded-full bg-surface-2 overflow-hidden">
            <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${r.mastery * 100}%`, background: r.color }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

export function ContinueCard({ lessons }: { lessons: LessonIndex }) {
  const p = useProgress();
  const hydrated = useHydrated();
  const lastLesson = p.recent.find((r) => r.path.startsWith("/learn/"));
  const next = lessons.find((l) => !p.lessons[l.slug]);
  const doneCount = lessons.filter((l) => p.lessons[l.slug]).length;
  const target = lastLesson && !p.lessons[lastLesson.path.split("/")[2]] ? { href: lastLesson.path, title: lastLesson.title, label: "Continue where you left off" }
    : next ? { href: `/learn/${next.slug}/`, title: next.title, label: doneCount ? "Recommended next lesson" : "Start here" } : null;
  if (!hydrated) return <div className="h-36 rounded-xl border border-line bg-surface animate-pulse" aria-hidden />;
  return (
    <div className="rounded-xl border border-line bg-surface p-5">
      <p className="text-sm text-muted">{target?.label ?? "Course complete"}</p>
      {target ? (
        <Link href={target.href} className="mt-1 block text-xl font-semibold tracking-tight hover:text-accent">{target.title}</Link>
      ) : (
        <p className="mt-1 text-xl font-semibold">Every lesson is marked complete. Time for the projects.</p>
      )}
      <div className="mt-4 h-1.5 rounded-full bg-surface-2 overflow-hidden" aria-label={`${doneCount} of ${lessons.length} lessons complete`}>
        <div className="h-full bg-accent rounded-full" style={{ width: `${(doneCount / Math.max(1, lessons.length)) * 100}%` }} />
      </div>
      <p className="mt-2 text-xs text-faint font-mono">{doneCount} of {lessons.length} lessons complete</p>
    </div>
  );
}

export function RecentList() {
  const p = useProgress();
  const hydrated = useHydrated();
  if (!hydrated) return null;
  if (!p.recent.length) return <p className="text-sm text-muted">Nothing yet. Pages you open show up here.</p>;
  return (
    <ul className="space-y-1.5">
      {p.recent.slice(0, 6).map((r) => (
        <li key={r.path}><Link href={r.path} className="text-sm text-muted hover:text-ink">{r.title}</Link></li>
      ))}
    </ul>
  );
}
