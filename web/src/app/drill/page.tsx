import type { Metadata } from "next";
import path from "node:path";
import { Suspense } from "react";
import { Drill, type DrillCard, type DrillScope } from "@/components/Drill";
import { getLessons, getModules, getQuizzes, getSprints, routeForFile } from "@/lib/content";
import { BASE_PATH } from "@/lib/site";

export const metadata: Metadata = {
  title: "Rapid-fire drill",
  description: "Timed interview flashcards and quiz blitzes across the whole course, a role sprint, a module, or one lesson.",
};

const AUTO = new Set(["single", "multi", "numeric", "fill"]);

/** Guide-relative markdown links -> site URLs, so answers rendered on the client link correctly. */
const siteLinks = (md: string, file: string) =>
  md.replace(/\]\((?!https?:|#)([^)\s]+)\)/g, (all, href: string) => {
    const [p, hash] = href.split("#");
    const route = routeForFile(path.posix.normalize(path.posix.join(path.posix.dirname(file), p)));
    return route ? `](${BASE_PATH}${route}${hash ? `#${hash}` : ""})` : all;
  });

export default function DrillPage() {
  const lessons = getLessons();
  const byQuiz = new Map(lessons.filter((l) => l.quiz).map((l) => [l.quiz!, l]));
  const cards: DrillCard[] = [
    ...lessons.flatMap((l) =>
      l.interview.map((x, i) => ({ id: `${l.slug}#${i}`, kind: "interview" as const, lesson: l.slug, lessonTitle: l.title, q: x.q, a: siteLinks(x.a, l.file) }))),
    ...getQuizzes().flatMap((quiz) => {
      const l = byQuiz.get(quiz.id);
      if (!l) return [];
      return quiz.questions.filter((q) => AUTO.has(q.type))
        .map((q) => ({ id: `${quiz.id}/${q.id}`, kind: "quiz" as const, lesson: l.slug, lessonTitle: l.title, question: q }));
    }),
  ];
  const scopes: DrillScope[] = [
    { id: "all", label: "Whole course", lessons: lessons.map((l) => l.slug) },
    ...getSprints().map((s) => ({ id: `sprint:${s.id}`, label: `${s.role} sprint`, lessons: s.lessons })),
    ...getModules().filter((m) => m.lessons.length).map((m) => ({ id: `module:${m.id}`, label: `Module ${m.number}: ${m.title}`, lessons: m.lessons.map((l) => l.slug) })),
    ...lessons.map((l) => ({ id: `lesson:${l.slug}`, label: `Lesson: ${l.title}`, lessons: [l.slug], hidden: true })),
  ];

  return (
    <div className="mx-auto max-w-3xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Rapid-fire drill</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">
        Interview flashcards: read the question, answer out loud before the timer runs out, then compare with a model answer. Quiz blitz: timed auto-graded questions. Cards you mark shaky come back in “My shaky cards”.
      </p>
      <Suspense>
        <Drill cards={cards} scopes={scopes} />
      </Suspense>
    </div>
  );
}
