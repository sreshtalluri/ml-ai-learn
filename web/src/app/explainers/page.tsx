import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight } from "@phosphor-icons/react/dist/ssr";
import { getExplainers, getLesson } from "@/lib/content";

export const metadata: Metadata = {
  title: "Visual explainers",
  description: "Scroll-driven walkthroughs: the lab stays on screen and changes as you read.",
};

export default function ExplainersPage() {
  const explainers = getExplainers();
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Visual explainers</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">
        One idea, told in pictures. The interactive lab stays pinned on screen and moves as you scroll, so you watch the mechanism change one step at a time. Prefer to be shown? Every lab also has a <strong>Watch</strong> button that plays its tour with captions and optional narration.
      </p>
      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {explainers.map((e) => (
          <Link key={e.id} href={`/explainers/${e.id}/`} className="group rounded-xl border border-line bg-surface p-5 hover:border-faint flex flex-col">
            <span className="font-semibold group-hover:text-accent">{e.title}</span>
            <span className="mt-1 text-sm text-muted flex-1">{e.summary}</span>
            <span className="mt-3 flex items-center justify-between text-xs text-faint">
              <span>{e.minutes} min · from {getLesson(e.lesson)?.title}</span>
              <ArrowRight size={14} className="text-accent" />
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
