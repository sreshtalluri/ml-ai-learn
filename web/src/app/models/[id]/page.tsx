import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Markdown } from "@/components/Markdown";
import { BookmarkButton, VisitTracker } from "@/components/PageControls";
import { getLesson, getModel, getModels } from "@/lib/content";
import { getLab } from "@/lib/labs";
import { stripH1, toc } from "@/lib/toc";

export function generateStaticParams() {
  return getModels().map((m) => ({ id: m.id }));
}

export async function generateMetadata({ params }: PageProps<"/models/[id]">): Promise<Metadata> {
  const m = getModel((await params).id);
  return m ? { title: `${m.name} model card` } : {};
}

export default async function ModelPage({ params }: PageProps<"/models/[id]">) {
  const m = getModel((await params).id);
  if (!m) notFound();
  const body = stripH1(m.body);
  const path = `/models/${m.id}/`;
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 pt-10 lg:grid lg:grid-cols-[minmax(0,1fr)_13rem] lg:gap-12">
      <VisitTracker path={path} title={`${m.name} (model)`} />
      <article className="min-w-0 max-w-3xl">
        <Link href="/models/" className="text-sm text-faint hover:text-ink">Model explorer</Link>
        <h1 className="mt-2 text-3xl md:text-4xl font-semibold tracking-tight">{m.name}</h1>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {m.tags.map((t) => <span key={t} className="rounded-full bg-surface-2 px-2.5 py-0.5 text-xs text-muted">{t.replace(/-/g, " ")}</span>)}
        </div>
        <div className="mt-4 flex flex-wrap gap-2 items-center text-sm">
          <BookmarkButton path={path} title={`${m.name} (model)`} />
          {m.lessons.map((s) => getLesson(s)).filter((l) => l !== undefined).map((l) => (
            <Link key={l.slug} href={`/learn/${l.slug}/`} className="rounded-lg border border-line px-3 py-1.5 text-muted hover:text-ink">Lesson: {l.title}</Link>
          ))}
          {m.labs.map((id) => getLab(id)).filter((l) => l !== undefined).map((l) => (
            <Link key={l.id} href={`/labs/${l.id}/`} className="rounded-lg border border-line px-3 py-1.5 text-muted hover:text-ink">Lab: {l.title}</Link>
          ))}
        </div>
        <div className="mt-8"><Markdown source={body} file={m.file} /></div>
      </article>
      <aside className="hidden lg:block" aria-label="Card sections">
        <ul className="sticky top-24 space-y-1.5 text-sm">
          {toc(body).filter((t) => t.depth === 2).map((t) => <li key={t.id}><a href={`#${t.id}`} className="text-muted hover:text-ink">{t.text}</a></li>)}
        </ul>
      </aside>
    </div>
  );
}
