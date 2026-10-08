"use client";
import { useEffect, useState } from "react";
import { BookmarkSimple, CheckCircle, Circle, NotePencil } from "@phosphor-icons/react";
import { actions, useHydrated, useProgress } from "@/lib/progress";

/** Records the visit for "recently viewed" and streaks. Renders nothing. */
export function VisitTracker({ path, title }: { path: string; title: string }) {
  useEffect(() => actions.visit(path, title), [path, title]);
  return null;
}

export function BookmarkButton({ path, title }: { path: string; title: string }) {
  const p = useProgress();
  const on = p.bookmarks.some((b) => b.path === path);
  return (
    <button type="button" onClick={() => actions.toggleBookmark(path, title)} aria-pressed={on}
      className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm ${on ? "border-accent text-accent bg-accent-soft" : "border-line text-muted hover:text-ink"}`}>
      <BookmarkSimple size={16} weight={on ? "fill" : "regular"} /> {on ? "Bookmarked" : "Bookmark"}
    </button>
  );
}

export function CompleteButton({ slug }: { slug: string }) {
  const p = useProgress();
  const done = !!p.lessons[slug];
  return (
    <button type="button" onClick={() => actions.setLessonComplete(slug, !done)} aria-pressed={done}
      className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium active:scale-[0.98] ${done ? "bg-good/15 text-good border border-good/40" : "bg-accent text-white dark:text-zinc-950"}`}>
      {done ? <CheckCircle size={16} weight="fill" /> : <Circle size={16} />} {done ? "Completed" : "Mark complete"}
    </button>
  );
}

export function NoteBox({ path }: { path: string }) {
  const p = useProgress();
  const hydrated = useHydrated();
  const saved = p.notes[path] ?? "";
  const [draft, setDraft] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const value = draft ?? saved;
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="flex items-center gap-1.5 text-sm font-medium">
        <NotePencil size={16} /> Personal note {hydrated && saved && !open && <span className="text-faint font-normal">(saved)</span>}
      </button>
      {open && (
        <div className="mt-3 space-y-2">
          <label className="sr-only" htmlFor={`note-${path}`}>Note for this page</label>
          <textarea id={`note-${path}`} rows={4} value={value} onChange={(e) => setDraft(e.target.value)}
            placeholder="What clicked, what didn't, what to revisit."
            className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm" />
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => { actions.setNote(path, value); setDraft(null); }} className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white dark:text-zinc-950">Save note</button>
            <span className="text-xs text-faint">Stored only in this browser.</span>
          </div>
        </div>
      )}
    </div>
  );
}

/** Small check mark for lesson lists. */
export function DoneMark({ slug }: { slug: string }) {
  const p = useProgress();
  return p.lessons[slug]
    ? <CheckCircle size={18} weight="fill" className="text-good shrink-0" aria-label="Completed" />
    : <Circle size={18} className="text-line shrink-0" aria-label="Not completed" />;
}
