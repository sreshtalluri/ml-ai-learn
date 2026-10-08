import type { Metadata } from "next";
import { QuizIndex } from "@/components/quiz/QuizIndex";
import { getLesson, getQuizzes } from "@/lib/content";

export const metadata: Metadata = { title: "Quizzes", description: "Active recall for every lesson: calculations, diagnosis, ordering, matching, and design questions." };

export default function QuizzesPage() {
  const quizzes = getQuizzes().map((q) => {
    const lesson = q.lesson ? getLesson(q.lesson) : undefined;
    return {
      id: q.id, title: q.title, count: q.questions.length,
      lessonTitle: lesson?.title, moduleId: lesson?.moduleId ?? "",
      questions: q.questions.map((x) => ({ id: x.id, prompt: x.prompt })),
    };
  }).sort((a, b) => a.moduleId.localeCompare(b.moduleId));
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Quizzes</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">
        Recognition is not understanding. These quizzes ask you to calculate, diagnose bugs, order processes, and make design calls. Missed questions go into your review queue.
      </p>
      <QuizIndex quizzes={quizzes} />
    </div>
  );
}
