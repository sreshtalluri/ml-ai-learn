import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { LabEmbed } from "@/components/labs/LabEmbed";
import { VisitTracker } from "@/components/PageControls";
import { getLesson } from "@/lib/content";
import { getLab, LAB_AREAS, LABS } from "@/lib/labs";

export function generateStaticParams() {
  return LABS.map((l) => ({ id: l.id }));
}

export async function generateMetadata({ params }: PageProps<"/labs/[id]">): Promise<Metadata> {
  const l = getLab((await params).id);
  return l ? { title: `${l.title} lab`, description: l.description } : {};
}

export default async function LabPage({ params }: PageProps<"/labs/[id]">) {
  const lab = getLab((await params).id);
  if (!lab) notFound();
  const lesson = getLesson(lab.lesson);
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 pt-10">
      <VisitTracker path={`/labs/${lab.id}/`} title={`${lab.title} lab`} />
      <Link href="/labs/" className="text-sm text-faint hover:text-ink">Labs / {LAB_AREAS[lab.area].title}</Link>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">{lab.title}</h1>
      <p className="mt-2 text-muted max-w-[65ch]">{lab.description}</p>
      {lesson && (
        <p className="mt-2 text-sm text-muted">
          Theory and worked examples: <Link href={`/learn/${lesson.slug}/`} className="text-accent hover:underline">{lesson.title}</Link>
        </p>
      )}
      <LabEmbed id={lab.id} />
    </div>
  );
}
