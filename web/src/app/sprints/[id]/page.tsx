import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Lightning } from "@phosphor-icons/react/dist/ssr";
import { Markdown } from "@/components/Markdown";
import { BookmarkButton, VisitTracker } from "@/components/PageControls";
import { ModuleProgress } from "@/components/ProgressWidgets";
import { QuickToggle } from "@/components/QuickToggle";
import { getSprint, getSprints } from "@/lib/content";
import { stripH1 } from "@/lib/toc";

export function generateStaticParams() {
  return getSprints().map((s) => ({ id: s.id }));
}

export async function generateMetadata({ params }: PageProps<"/sprints/[id]">): Promise<Metadata> {
  const s = getSprint((await params).id);
  return s ? { title: s.title, description: s.summary } : {};
}

export default async function SprintPage({ params }: PageProps<"/sprints/[id]">) {
  const s = getSprint((await params).id);
  if (!s) notFound();
  const path = `/sprints/${s.id}/`;
  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 pt-10">
      <VisitTracker path={path} title={s.title} />
      <Link href="/sprints/" className="text-sm text-faint hover:text-ink">Interview prep</Link>
      <h1 className="mt-2 text-3xl md:text-4xl font-semibold tracking-tight">{s.role}: 7-day sprint</h1>
      <p className="mt-2 text-muted max-w-[65ch]">{s.summary}</p>
      <div className="mt-5 flex flex-wrap items-center gap-2">
        <Link href={`/drill/?sprint=${s.id}`} className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white dark:text-zinc-950"><Lightning size={14} /> Rapid-fire this sprint</Link>
        <BookmarkButton path={path} title={s.title} />
        <span className="ml-1"><ModuleProgress slugs={s.lessons} /></span>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3 text-sm text-muted">
        <span>Reading mode for lessons:</span> <QuickToggle />
      </div>
      <div className="mt-8"><Markdown source={stripH1(s.body)} file={s.file} /></div>
    </div>
  );
}
