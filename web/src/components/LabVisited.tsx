"use client";
import { useProgress } from "@/lib/progress";

export function LabVisited({ id }: { id: string }) {
  const p = useProgress();
  return p.labs[id] ? <span className="text-[11px] rounded-full bg-good/15 text-good px-2 py-0.5 whitespace-nowrap">explored</span> : null;
}
