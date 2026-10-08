"use client";
import { useState } from "react";
import { Check, Copy } from "@phosphor-icons/react";

export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          setTimeout(() => setDone(false), 1400);
        });
      }}
      className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-faint hover:text-ink hover:bg-surface-2"
      aria-label={label}
    >
      {done ? <Check size={14} /> : <Copy size={14} />}
      {done ? "Copied" : label}
    </button>
  );
}

export function CodeBlock({ code, lang }: { code: string; lang?: string }) {
  return (
    <div className="not-prose group relative rounded-xl border border-line bg-surface-2/60 overflow-hidden">
      <div className="flex items-center justify-between border-b border-line px-3 py-1.5">
        <span className="font-mono text-xs text-faint">{lang ?? "text"}</span>
        <CopyButton text={code} />
      </div>
      <pre className="overflow-x-auto p-4 text-[0.84rem] leading-relaxed font-mono">
        <code>{code}</code>
      </pre>
    </div>
  );
}
