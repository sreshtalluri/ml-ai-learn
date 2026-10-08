import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Suspense } from "react";
import { QuizPageClient } from "@/components/quiz/QuizPageClient";
import { QuizRunner } from "@/components/quiz/QuizRunner";
import { getLesson, getQuiz, getQuizzes } from "@/lib/content";

export function generateStaticParams() {
  return getQuizzes().map((q) => ({ id: q.id }));
}

export async function generateMetadata({ params }: PageProps<"/quizzes/[id]">): Promise<Metadata> {
  const q = getQuiz((await params).id);
  return q ? { title: `${q.title} quiz` } : {};
}

export default async function QuizPage({ params }: PageProps<"/quizzes/[id]">) {
  const quiz = getQuiz((await params).id);
  if (!quiz) notFound();
  const lesson = quiz.lesson ? getLesson(quiz.lesson) : undefined;
  return (
    <div className="mx-auto max-w-3xl px-4 sm:px-6 pt-10">
      <Link href="/quizzes/" className="text-sm text-faint hover:text-ink">Quizzes</Link>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">{quiz.title}</h1>
      {lesson && <p className="mt-2 text-muted">Covers the lesson <Link href={`/learn/${lesson.slug}/`} className="text-accent hover:underline">{lesson.title}</Link>.</p>}
      <Suspense fallback={<QuizRunner quiz={quiz} />}>
        <QuizPageClient quiz={quiz} />
      </Suspense>
    </div>
  );
}
