import type { Metadata } from "next";
import { ModelExplorer } from "@/components/ModelExplorer";
import { getModels, MODEL_TAGS } from "@/lib/content";

export const metadata: Metadata = { title: "Model explorer", description: "Searchable model cards: what each model optimizes, assumes, and how it fails." };

export default function ModelsPage() {
  const models = getModels().map(({ id, name, tags, summary }) => ({ id, name, tags, summary }));
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Model explorer</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">
        One card per model: inputs, outputs, objective, assumptions, hyperparameters, failure modes, and how it compares with its neighbors. Filter by what you need.
      </p>
      <ModelExplorer models={models} tags={[...MODEL_TAGS]} />
    </div>
  );
}
