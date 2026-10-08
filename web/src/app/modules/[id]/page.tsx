import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Markdown } from "@/components/Markdown";
import { DoneMark } from "@/components/PageControls";
import { getModule, getModules } from "@/lib/content";
import { stripH1 } from "@/lib/toc";

export function generateStaticParams() {
  return getModules().map((m) => ({ id: m.id }));
}

export async function generateMetadata({ params }: PageProps<"/modules/[id]">): Promise<Metadata> {
  const m = getModule((await params).id);
  return m ? { title: `Module ${m.number}: ${m.title}`, description: m.summary } : {};
}

export default async function ModulePage({ params }: PageProps<"/modules/[id]">) {
  const m = getModule((await params).id);
  if (!m) notFound();
  const all = getModules();
  const nextMod = all.find((x) => x.number === m.number + 1);
  return (
    <div className="mx-auto max-w-3xl px-4 sm:px-6 pt-12">
      <Link href="/path/" className="text-sm text-faint hover:text-ink">Learning path</Link>
      <p className="mt-4 font-mono text-sm text-faint">Module {m.number}</p>
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">{m.title}</h1>
      <p className="mt-3 text-lg text-muted">{m.summary}</p>

      <ol className="mt-8 rounded-xl border border-line bg-surface divide-y divide-line">
        {m.lessons.map((l, i) => (
          <li key={l.slug}>
            <Link href={`/learn/${l.slug}/`} className="flex gap-4 px-5 py-4 hover:bg-surface-2/60">
              <DoneMark slug={l.slug} />
              <span className="flex-1">
                <span className="font-medium">{i + 1}. {l.title}</span>
                <span className="block text-sm text-muted mt-0.5">{l.summary}</span>
              </span>
              <span className="text-xs text-faint whitespace-nowrap">{l.minutes} min</span>
            </Link>
          </li>
        ))}
      </ol>

      {m.body.trim() && <div className="mt-10"><Markdown source={stripH1(m.body)} file={m.file} /></div>}

      {nextMod && (
        <Link href={`/modules/${nextMod.id}/`} className="mt-12 block rounded-xl border border-line p-4 hover:border-faint">
          <span className="text-xs text-faint">Next module</span>
          <span className="block font-medium">Module {nextMod.number}: {nextMod.title}</span>
        </Link>
      )}
    </div>
  );
}
