import type { Metadata } from "next";
import { ProgressDashboard } from "@/components/ProgressDashboard";
import { getLessonIndex, getModules, getQuizzes } from "@/lib/content";
import { LABS } from "@/lib/labs";

export const metadata: Metadata = { title: "Progress", description: "Lessons, labs, quiz scores, streaks, notes, and bookmarks, stored locally." };

export default function ProgressPage() {
  const modules = getModules().map((m) => ({ id: m.id, number: m.number, title: m.title, slugs: m.lessons.map((l) => l.slug) }));
  const quizzes = getQuizzes().map((q) => ({ id: q.id, title: q.title, lesson: q.lesson ?? q.id }));
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Progress</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">Everything you have completed, explored, and flagged for review.</p>
      <ProgressDashboard lessons={getLessonIndex()} modules={modules} labCount={LABS.length} quizzes={quizzes} />
    </div>
  );
}
