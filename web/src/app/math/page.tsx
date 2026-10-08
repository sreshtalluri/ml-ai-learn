import type { Metadata } from "next";
import { MathLab } from "@/components/MathLab";
import { getLessons } from "@/lib/content";

export const metadata: Metadata = { title: "Math Lab", description: "Editable calculators that show every formula, symbol, and arithmetic step." };

export default function MathPage() {
  const lessonTitles = Object.fromEntries(getLessons().map((l) => [l.slug, l.title]));
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Math Lab</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">
        Change the inputs and every intermediate step recomputes. Use it to check your hand calculations from the worked examples.
      </p>
      <MathLab lessonTitles={lessonTitles} />
    </div>
  );
}
