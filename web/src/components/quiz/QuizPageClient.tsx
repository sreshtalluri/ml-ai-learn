"use client";
import { useSearchParams } from "next/navigation";
import type { Quiz } from "@/lib/quiz-types";
import { useHydrated, useProgress } from "@/lib/progress";
import { QuizRunner } from "./QuizRunner";

/** ?review=1 runs only the questions in this quiz's review queue. */
export function QuizPageClient({ quiz }: { quiz: Quiz }) {
  const review = useSearchParams().get("review") === "1";
  const p = useProgress();
  const hydrated = useHydrated();
  if (!review) return <QuizRunner quiz={quiz} />;
  if (!hydrated) return null;
  return <QuizRunner key="review" quiz={quiz} onlyIds={p.quizzes[quiz.id]?.wrong ?? []} />;
}
