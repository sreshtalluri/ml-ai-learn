import type { Metadata } from "next";
import Link from "next/link";
import { getCheatSheets } from "@/lib/content";

export const metadata: Metadata = { title: "Cheat sheets", description: "Concise references for metrics, losses, optimizers, model selection, RAG, and more." };

export default function CheatSheetsPage() {
  const sheets = getCheatSheets();
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Cheat sheets</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">One page per topic, for review and interviews. Each sheet links back to the full lessons.</p>
      <div className="mt-10 grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {sheets.map((s) => (
          <Link key={s.id} href={`/cheatsheets/${s.id}/`} className="group rounded-xl border border-line bg-surface p-4 hover:border-faint">
            <span className="font-semibold group-hover:text-accent">{s.title}</span>
            <span className="block text-sm text-muted mt-1 leading-snug">{s.summary}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
