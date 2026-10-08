"use client";
import Link from "next/link";
import { useState } from "react";
import { MagnifyingGlass } from "@phosphor-icons/react";
import { MiniMarkdown } from "./MiniMarkdown";

type Card = { id: string; name: string; tags: string[]; summary: string };

export function ModelExplorer({ models, tags }: { models: Card[]; tags: string[] }) {
  const [q, setQ] = useState("");
  const [active, setActive] = useState<string[]>([]);
  const shown = models.filter(
    (m) =>
      active.every((t) => m.tags.includes(t)) &&
      (m.name + " " + m.summary + " " + m.tags.join(" ")).toLowerCase().includes(q.toLowerCase().trim()),
  );
  return (
    <div className="mt-8">
      <label className="relative block max-w-md">
        <span className="sr-only">Search models</span>
        <MagnifyingGlass size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-faint" />
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search, e.g. “margin” or “tabular”"
          className="w-full rounded-lg border border-line bg-surface pl-9 pr-3 py-2 text-sm" />
      </label>
      <div className="mt-4 flex flex-wrap gap-1.5" role="group" aria-label="Filter by tag">
        {tags.map((t) => {
          const on = active.includes(t);
          return (
            <button key={t} type="button" aria-pressed={on} onClick={() => setActive(on ? active.filter((x) => x !== t) : [...active, t])}
              className={`rounded-full border px-2.5 py-1 text-xs ${on ? "border-accent bg-accent-soft text-accent" : "border-line text-muted hover:text-ink"}`}>
              {t.replace(/-/g, " ")}
            </button>
          );
        })}
        {active.length > 0 && <button type="button" onClick={() => setActive([])} className="px-2 text-xs text-faint underline">clear</button>}
      </div>
      <p className="mt-4 text-sm text-faint" aria-live="polite">{shown.length} of {models.length} models</p>
      {shown.length === 0 ? (
        <p className="mt-6 rounded-xl border border-dashed border-line p-8 text-center text-muted">No model has all of those tags. Remove a filter to widen the search.</p>
      ) : (
        <div className="mt-3 grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {shown.map((m) => (
            <Link key={m.id} href={`/models/${m.id}/`} className="group rounded-xl border border-line bg-surface p-4 hover:border-faint">
              <span className="font-semibold group-hover:text-accent">{m.name}</span>
              <span className="block text-sm text-muted mt-1 leading-snug line-clamp-3"><MiniMarkdown inline>{m.summary}</MiniMarkdown></span>
              <span className="mt-3 flex flex-wrap gap-1">
                {m.tags.map((t) => <span key={t} className="rounded-full bg-surface-2 px-2 py-0.5 text-[11px] text-faint">{t.replace(/-/g, " ")}</span>)}
              </span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
