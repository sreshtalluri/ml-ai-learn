"use client";
// Local-first progress store. One JSON blob in localStorage; components subscribe
// via useSyncExternalStore. Swap `storage` for a synced backend later without touching callers.
import { useSyncExternalStore } from "react";

export interface QuizRecord { best: number; last: number; total: number; attempts: number; wrong: string[] }

export interface Progress {
  version: 1;
  lessons: Record<string, number>;           // slug -> completedAt (ms)
  labs: Record<string, number>;              // lab id -> first interaction (ms)
  quizzes: Record<string, QuizRecord>;       // quiz id -> record
  bookmarks: { path: string; title: string }[];
  notes: Record<string, string>;             // path -> note
  recent: { path: string; title: string }[]; // most recent first, max 8
  days: string[];                            // YYYY-MM-DD with activity (for streaks)
  projects: Record<string, boolean>;         // "project-id/milestone-index" -> done
}

export const STORAGE_KEY = "ml-ai-learn:progress:v1";

export const emptyProgress = (): Progress => ({
  version: 1, lessons: {}, labs: {}, quizzes: {}, bookmarks: [], notes: {}, recent: [], days: [], projects: {},
});

const listeners = new Set<() => void>();
let cache: Progress | null = null;
let cacheRaw: string | null = null;

function load(): Progress {
  if (typeof window === "undefined") return SERVER_SNAPSHOT;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (cache && raw === cacheRaw) return cache;
  cacheRaw = raw;
  try {
    cache = raw ? parseProgress(raw) : emptyProgress();
  } catch {
    cache = emptyProgress();
  }
  return cache;
}

function save(p: Progress) {
  const raw = JSON.stringify(p);
  window.localStorage.setItem(STORAGE_KEY, raw);
  cache = p;
  cacheRaw = raw;
  listeners.forEach((l) => l());
}

function update(fn: (p: Progress) => Progress) {
  const today = new Date().toISOString().slice(0, 10);
  const next = fn(structuredClone(load()));
  if (!next.days.includes(today)) next.days = [...next.days, today].slice(-400);
  save(next);
}

/** Validate an imported/stored blob; throws on anything that isn't a progress file. */
export function parseProgress(raw: string): Progress {
  const data = JSON.parse(raw);
  if (!data || typeof data !== "object" || data.version !== 1) throw new Error("Not an ml-ai-learn progress file (version 1).");
  const base = emptyProgress();
  for (const key of Object.keys(base) as (keyof Progress)[]) {
    if (key === "version") continue;
    const want = Array.isArray(base[key]) ? "array" : "object";
    const got = data[key] === undefined ? want : Array.isArray(data[key]) ? "array" : typeof data[key];
    if (got !== want) throw new Error(`Invalid progress file: "${key}" should be an ${want}.`);
  }
  return { ...base, ...data };
}

const SERVER_SNAPSHOT = emptyProgress();

function subscribe(cb: () => void) {
  listeners.add(cb);
  const onStorage = (e: StorageEvent) => e.key === STORAGE_KEY && cb();
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", onStorage);
  };
}

export function useProgress(): Progress {
  return useSyncExternalStore(subscribe, load, () => SERVER_SNAPSHOT);
}

/** True once mounted on the client, so pages can avoid flashing empty server state. */
export function useHydrated(): boolean {
  return useSyncExternalStore(subscribe, () => true, () => false);
}

export const actions = {
  setLessonComplete(slug: string, done: boolean) {
    update((p) => {
      if (done) p.lessons[slug] = Date.now();
      else delete p.lessons[slug];
      return p;
    });
  },
  touchLab(id: string) {
    if (load().labs[id]) return;
    update((p) => ({ ...p, labs: { ...p.labs, [id]: Date.now() } }));
  },
  recordQuiz(id: string, correct: number, total: number, wrong: string[]) {
    update((p) => {
      const prev = p.quizzes[id];
      const score = total ? correct / total : 0;
      p.quizzes[id] = {
        best: Math.max(prev?.best ?? 0, score),
        last: score,
        total,
        attempts: (prev?.attempts ?? 0) + 1,
        wrong,
      };
      return p;
    });
  },
  setReviewQueue(id: string, wrong: string[]) {
    update((p) => {
      if (p.quizzes[id]) p.quizzes[id].wrong = wrong;
      return p;
    });
  },
  toggleBookmark(path: string, title: string) {
    update((p) => {
      const has = p.bookmarks.some((b) => b.path === path);
      p.bookmarks = has ? p.bookmarks.filter((b) => b.path !== path) : [{ path, title }, ...p.bookmarks];
      return p;
    });
  },
  setNote(path: string, note: string) {
    update((p) => {
      if (note.trim()) p.notes[path] = note;
      else delete p.notes[path];
      return p;
    });
  },
  visit(path: string, title: string) {
    const cur = load().recent;
    if (cur[0]?.path === path) return;
    update((p) => ({ ...p, recent: [{ path, title }, ...p.recent.filter((r) => r.path !== path)].slice(0, 8) }));
  },
  toggleProject(key: string) {
    update((p) => ({ ...p, projects: { ...p.projects, [key]: !p.projects[key] } }));
  },
  exportJSON(): string {
    return JSON.stringify(load(), null, 2);
  },
  importJSON(raw: string) {
    save(parseProgress(raw));
  },
  reset() {
    save(emptyProgress());
  },
};

/** Consecutive days with activity ending today (or yesterday, so the streak survives until you study). */
export function streak(days: string[], today = new Date()): number {
  const set = new Set(days);
  const d = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate()));
  if (!set.has(d.toISOString().slice(0, 10))) d.setUTCDate(d.getUTCDate() - 1);
  let n = 0;
  while (set.has(d.toISOString().slice(0, 10))) {
    n++;
    d.setUTCDate(d.getUTCDate() - 1);
  }
  return n;
}
