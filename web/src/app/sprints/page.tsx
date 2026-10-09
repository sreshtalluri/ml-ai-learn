import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, Lightning } from "@phosphor-icons/react/dist/ssr";
import { Markdown } from "@/components/Markdown";
import { ModuleProgress } from "@/components/ProgressWidgets";
import { getDoc, getSprints } from "@/lib/content";
import { stripH1 } from "@/lib/toc";

export const metadata: Metadata = {
  title: "Interview prep",
  description: "Seven-day interview sprints for ML engineers, AI engineers, applied scientists, and data scientists, plus a rapid-fire drill.",
};

export default function SprintsPage() {
  const sprints = getSprints();
  const doc = getDoc("sprints/README.md");
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Interview prep</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">{doc?.data.summary}</p>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {sprints.map((s) => (
          <div key={s.id} className="rounded-xl border border-line bg-surface p-5 flex flex-col">
            <div className="flex items-baseline justify-between gap-2">
              <h2 className="text-lg font-semibold">{s.role}</h2>
              <ModuleProgress slugs={s.lessons} />
            </div>
            <p className="mt-1 text-sm text-muted flex-1">{s.summary}</p>
            <p className="mt-2 text-xs text-faint">7 days · {s.lessons.length} lessons</p>
            <div className="mt-4 flex flex-wrap gap-2">
              <Link href={`/sprints/${s.id}/`} className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white dark:text-zinc-950">Open sprint <ArrowRight size={14} /></Link>
              <Link href={`/drill/?sprint=${s.id}`} className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-sm text-muted hover:text-ink"><Lightning size={14} /> Rapid-fire</Link>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 rounded-xl border border-line p-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-semibold">Rapid-fire drill</h2>
          <p className="text-sm text-muted">Interview flashcards and timed quiz questions from any sprint, module, or the whole course.</p>
        </div>
        <div className="flex gap-2">
          <Link href="/drill/" className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white dark:text-zinc-950">Start drilling</Link>
          <Link href="/quizzes/" className="rounded-lg border border-line px-3 py-1.5 text-sm text-muted hover:text-ink">All quizzes and review queue</Link>
        </div>
      </div>

      {/* the README's role table duplicates the cards above */}
      {doc && <div className="mt-12"><Markdown source={stripH1(doc.body).replace(/^\|.*\n/gm, "")} file={doc.file} /></div>}
    </div>
  );
}
