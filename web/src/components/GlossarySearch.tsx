"use client";
import { useState } from "react";
import { MagnifyingGlass } from "@phosphor-icons/react";
import { MiniMarkdown } from "./MiniMarkdown";

type Entry = { id: string; term: string; fields: Record<string, string> };
const ORDER = ["Plain English", "Formal", "Example", "Related", "Lesson"];

export function GlossarySearch({ entries }: { entries: Entry[] }) {
  const [q, setQ] = useState("");
  const needle = q.toLowerCase().trim();
  const shown = entries.filter((e) => !needle || e.term.toLowerCase().includes(needle) || Object.values(e.fields).join(" ").toLowerCase().includes(needle));
  const letters = [...new Set(entries.map((e) => e.term[0].toUpperCase()))].sort();
  return (
    <div className="mt-8">
      <label className="relative block max-w-md">
        <span className="sr-only">Search the glossary</span>
        <MagnifyingGlass size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-faint" />
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder={`Search ${entries.length} terms`} className="w-full rounded-lg border border-line bg-surface pl-9 pr-3 py-2 text-sm" />
      </label>
      {!needle && (
        <nav aria-label="Jump to letter" className="mt-4 flex flex-wrap gap-1 text-sm">
          {letters.map((l) => <a key={l} href={`#letter-${l}`} className="rounded px-2 py-0.5 text-muted hover:text-ink hover:bg-surface-2">{l}</a>)}
        </nav>
      )}
      {shown.length === 0 && <p className="mt-8 text-muted">No term matches “{q}”. Try a shorter word.</p>}
      <dl className="mt-8 space-y-4">
        {shown.map((e, i) => {
          const first = !needle && (i === 0 || shown[i - 1].term[0].toUpperCase() !== e.term[0].toUpperCase());
          return (
            <div key={e.id} id={e.id} className="scroll-mt-24 rounded-xl border border-line bg-surface p-5 target:border-accent">
              {first && <span id={`letter-${e.term[0].toUpperCase()}`} className="block scroll-mt-24" />}
              <dt className="text-lg font-semibold tracking-tight">{e.term}</dt>
              <dd className="mt-2 space-y-1.5 text-sm">
                {ORDER.filter((k) => e.fields[k]).map((k) => (
                  <p key={k} className={k === "Plain English" ? "text-ink text-[0.95rem]" : "text-muted"}>
                    {k !== "Plain English" && <span className="font-medium text-ink">{k}: </span>}
                    <span className="[&_a]:text-accent [&_a]:underline [&_a]:underline-offset-2"><MiniMarkdown inline>{e.fields[k]}</MiniMarkdown></span>
                  </p>
                ))}
              </dd>
            </div>
          );
        })}
      </dl>
    </div>
  );
}
