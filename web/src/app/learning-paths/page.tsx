import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Markdown } from "@/components/Markdown";
import { getDoc } from "@/lib/content";
import { stripH1 } from "@/lib/toc";

export const metadata: Metadata = { title: "Learning paths", description: "Three routes through the course: full, AI-engineering fast track, and interview prep." };

export default function LearningPathsPage() {
  const doc = getDoc("learning-paths.md");
  if (!doc) notFound();
  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">{doc.data.title}</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">{doc.data.summary}</p>
      <div className="mt-8"><Markdown source={stripH1(doc.body)} file={doc.file} /></div>
    </div>
  );
}
