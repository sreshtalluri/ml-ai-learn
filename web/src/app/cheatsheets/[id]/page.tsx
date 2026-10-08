import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Markdown } from "@/components/Markdown";
import { BookmarkButton, VisitTracker } from "@/components/PageControls";
import { getCheatSheet, getCheatSheets } from "@/lib/content";
import { stripH1 } from "@/lib/toc";

export function generateStaticParams() {
  return getCheatSheets().map((c) => ({ id: c.id }));
}

export async function generateMetadata({ params }: PageProps<"/cheatsheets/[id]">): Promise<Metadata> {
  const c = getCheatSheet((await params).id);
  return c ? { title: `${c.title} cheat sheet`, description: c.summary } : {};
}

export default async function CheatSheetPage({ params }: PageProps<"/cheatsheets/[id]">) {
  const c = getCheatSheet((await params).id);
  if (!c) notFound();
  const path = `/cheatsheets/${c.id}/`;
  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 pt-10">
      <VisitTracker path={path} title={`${c.title} (cheat sheet)`} />
      <Link href="/cheatsheets/" className="text-sm text-faint hover:text-ink">Cheat sheets</Link>
      <h1 className="mt-2 text-3xl md:text-4xl font-semibold tracking-tight">{c.title}</h1>
      <p className="mt-2 text-muted">{c.summary}</p>
      <div className="mt-4"><BookmarkButton path={path} title={`${c.title} (cheat sheet)`} /></div>
      <div className="mt-8"><Markdown source={stripH1(c.body)} file={c.file} /></div>
    </div>
  );
}
