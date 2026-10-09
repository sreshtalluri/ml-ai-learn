import type { Metadata } from "next";
import Link from "next/link";
import { LabVisited } from "@/components/LabVisited";
import { getLesson } from "@/lib/content";
import { LAB_AREAS, LABS, type LabArea } from "@/lib/labs";

export const metadata: Metadata = { title: "Interactive labs", description: "Every interactive visualization in the course, grouped by area." };

export default function LabsPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Interactive labs</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">Each lab also appears inside its lesson. Open one here when you want to experiment without the surrounding text. All data is synthetic and generated with fixed seeds. Press <strong>Watch</strong> on any lab for a guided, captioned tour, or read a <Link href="/explainers/" className="text-accent hover:underline">visual explainer</Link>, where the lab follows along as you scroll.</p>
      <div className="mt-12 space-y-12">
        {(Object.keys(LAB_AREAS) as LabArea[]).map((area) => {
          const labs = LABS.filter((l) => l.area === area);
          return (
            <section key={area} aria-labelledby={area}>
              <h2 id={area} className="text-xl font-semibold tracking-tight">{LAB_AREAS[area].title}</h2>
              <p className="text-sm text-muted mt-1">{LAB_AREAS[area].blurb}</p>
              <div className="mt-4 grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {labs.map((l) => (
                  <Link key={l.id} href={`/labs/${l.id}/`} className="group rounded-xl border border-line bg-surface p-4 hover:border-faint">
                    <span className="flex items-start justify-between gap-2">
                      <span className="font-semibold group-hover:text-accent">{l.title}</span>
                      <LabVisited id={l.id} />
                    </span>
                    <span className="block text-sm text-muted mt-1 leading-snug">{l.description}</span>
                    <span className="block text-xs text-faint mt-3">Lesson: {getLesson(l.lesson)?.title ?? l.lesson}</span>
                  </Link>
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
