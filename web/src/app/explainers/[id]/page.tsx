import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Markdown } from "@/components/Markdown";
import { VisitTracker } from "@/components/PageControls";
import { Scrolly } from "@/components/Scrolly";
import { getExplainer, getExplainers, getLesson } from "@/lib/content";

export function generateStaticParams() {
  return getExplainers().map((e) => ({ id: e.id }));
}

export async function generateMetadata({ params }: PageProps<"/explainers/[id]">): Promise<Metadata> {
  const e = getExplainer((await params).id);
  return e ? { title: `${e.title} (visual explainer)`, description: e.summary } : {};
}

export default async function ExplainerPage({ params }: PageProps<"/explainers/[id]">) {
  const e = getExplainer((await params).id);
  if (!e) notFound();
  const lesson = getLesson(e.lesson);
  const path = `/explainers/${e.id}/`;
  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 pt-10">
      <VisitTracker path={path} title={`${e.title} (explainer)`} />
      <Link href="/explainers/" className="text-sm text-faint hover:text-ink">Visual explainers</Link>
      <h1 className="mt-2 text-3xl md:text-4xl font-semibold tracking-tight">{e.title}</h1>
      <p className="mt-2 text-muted max-w-[65ch]">{e.summary}</p>
      <p className="mt-2 text-sm text-faint">
        {e.minutes} min · Scroll, and the lab follows along. You can still drag every control.
        {lesson && <> Full lesson: <Link href={`/learn/${lesson.slug}/`} className="text-accent hover:underline">{lesson.title}</Link>.</>}
      </p>
      <div className="mt-10">
        <Scrolly lab={e.lab} sections={e.sections.map((s) => ({ step: s.step, content: <Markdown source={s.body} file={e.file} /> }))} />
      </div>
    </div>
  );
}
