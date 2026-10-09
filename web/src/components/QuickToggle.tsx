"use client";
// Quick mode hides each lesson's math, implementation, and engineering sections.
// The class lives on <html> (set before paint in layout.tsx), so hiding is pure CSS.
import { useSyncExternalStore } from "react";
import { Lightning } from "@phosphor-icons/react";

const subscribe = (cb: () => void) => {
  window.addEventListener("quickmode", cb);
  return () => window.removeEventListener("quickmode", cb);
};
const isQuick = () => document.documentElement.classList.contains("quick");

export function setQuick(on: boolean) {
  document.documentElement.classList.toggle("quick", on);
  localStorage.setItem("quick", on ? "1" : "0");
  window.dispatchEvent(new Event("quickmode"));
}

export function QuickToggle() {
  const on = useSyncExternalStore(subscribe, isQuick, () => false);
  const btn = (active: boolean) =>
    `inline-flex items-center gap-1.5 rounded-md px-3 py-1 ${active ? "bg-surface-2 text-ink font-medium" : "text-muted hover:text-ink"}`;
  return (
    <div role="group" aria-label="Reading mode" className="inline-flex rounded-lg border border-line p-0.5 text-sm">
      <button type="button" aria-pressed={!on} onClick={() => setQuick(false)} className={btn(!on)}>Full lesson</button>
      <button type="button" aria-pressed={on} onClick={() => setQuick(true)} className={btn(on)}><Lightning size={14} /> Quick read</button>
    </div>
  );
}

export function ShowFullButton() {
  return <button type="button" onClick={() => setQuick(false)} className="underline underline-offset-4 hover:text-ink">Show the full lesson</button>;
}
