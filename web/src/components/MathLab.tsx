"use client";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { WarningCircle } from "@phosphor-icons/react";
import { CALCULATORS, type Calculator } from "@/lib/calculators";
import { CopyButton } from "./CodeBlock";
import { Tex } from "./labs/Tex";

function CalculatorView({ calc, lessonTitle }: { calc: Calculator; lessonTitle?: string }) {
  const defaults = useMemo(() => Object.fromEntries(calc.fields.map((f) => [f.key, f.default])), [calc]);
  const [values, setValues] = useState<Record<string, string>>(defaults);
  let out: ReturnType<Calculator["compute"]> | null = null;
  let error: string | null = null;
  try {
    out = calc.compute(values);
  } catch (e) {
    error = (e as Error).message;
  }

  return (
    <article className="rounded-xl border border-line bg-surface" aria-labelledby={`calc-${calc.id}`}>
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-5 py-3">
        <div>
          <p className="text-xs text-faint">{calc.group}</p>
          <h2 id={`calc-${calc.id}`} className="text-xl font-semibold tracking-tight">{calc.title}</h2>
        </div>
        <button type="button" onClick={() => setValues(defaults)} className="rounded-full border border-line px-3 py-1 text-xs text-muted hover:text-ink">Reset example</button>
      </header>

      <div className="p-5 space-y-6">
        <div className="flex items-start justify-between gap-2 rounded-lg bg-surface-2/60 px-4 py-3">
          <Tex display>{calc.formula}</Tex>
          <CopyButton text={calc.formula} label="LaTeX" />
        </div>

        <table className="w-full text-sm">
          <caption className="text-left text-xs text-faint mb-1">Symbols</caption>
          <tbody>
            {calc.symbols.map(([s, m]) => (
              <tr key={s} className="border-t border-line first:border-0">
                <td className="py-1.5 pr-4 whitespace-nowrap align-top"><Tex>{s}</Tex></td>
                <td className="py-1.5 text-muted">{m}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <fieldset className="grid sm:grid-cols-2 gap-4">
          <legend className="text-sm font-medium mb-2">Inputs</legend>
          {calc.fields.map((f) => (
            <label key={f.key} className={`block ${f.kind === "text" || f.kind === "matrix" ? "sm:col-span-2" : ""}`}>
              <span className="text-sm text-muted">{f.label}</span>
              {f.kind === "text" ? (
                <textarea rows={4} value={values[f.key] ?? ""} onChange={(e) => setValues({ ...values, [f.key]: e.target.value })}
                  className="mt-1 w-full rounded-lg border border-line bg-bg px-3 py-2 font-mono text-sm" />
              ) : (
                <input value={values[f.key] ?? ""} onChange={(e) => setValues({ ...values, [f.key]: e.target.value })}
                  inputMode={f.kind === "number" ? "decimal" : "text"}
                  className="mt-1 w-full rounded-lg border border-line bg-bg px-3 py-2 font-mono text-sm" />
              )}
              {f.hint && <span className="block text-xs text-faint mt-1">{f.hint}</span>}
            </label>
          ))}
        </fieldset>

        {error ? (
          <p role="alert" className="flex items-center gap-2 rounded-lg border border-bad/40 bg-bad/5 px-4 py-3 text-sm"><WarningCircle size={18} className="text-bad" /> {error}</p>
        ) : out && (
          <>
            <ol className="space-y-3" aria-label="Worked steps">
              {out.steps.map((s, i) => (
                <li key={i} className="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-2">
                  <span className="font-mono text-xs text-faint pt-1">{i + 1}</span>
                  <div className="min-w-0">
                    <p className="text-sm text-muted">{s.label}</p>
                    <div className="overflow-x-auto"><Tex display>{s.tex}</Tex></div>
                  </div>
                </li>
              ))}
            </ol>
            <div className="rounded-lg border border-accent/40 bg-accent-soft px-4 py-3">
              <p className="text-xs text-accent font-medium">Result</p>
              <div className="overflow-x-auto text-lg"><Tex display>{out.result}</Tex></div>
              <p className="text-sm mt-1">{out.meaning}</p>
            </div>
          </>
        )}

        <div>
          <p className="text-sm font-medium mb-1">Common mistakes</p>
          <ul className="list-disc pl-5 space-y-1 text-sm text-muted">{calc.mistakes.map((m) => <li key={m}>{m}</li>)}</ul>
        </div>
        <p className="text-sm"><Link href={`/learn/${calc.lesson}/`} className="text-accent hover:underline">Related lesson: {lessonTitle ?? calc.lesson}</Link></p>
      </div>
    </article>
  );
}

export function MathLab({ lessonTitles }: { lessonTitles: Record<string, string> }) {
  const [id, setId] = useState(CALCULATORS[0].id);
  useEffect(() => {
    const fromHash = () => {
      const h = window.location.hash.slice(1);
      if (CALCULATORS.some((c) => c.id === h)) setId(h);
    };
    fromHash();
    window.addEventListener("hashchange", fromHash);
    return () => window.removeEventListener("hashchange", fromHash);
  }, []);
  const groups = [...new Set(CALCULATORS.map((c) => c.group))];
  const calc = CALCULATORS.find((c) => c.id === id)!;
  return (
    <div className="mt-10 grid gap-8 lg:grid-cols-[14rem_minmax(0,1fr)]">
      <nav aria-label="Calculators" className="lg:sticky lg:top-24 lg:self-start">
        <label className="lg:hidden block">
          <span className="sr-only">Choose a calculator</span>
          <select value={id} onChange={(e) => { setId(e.target.value); history.replaceState(null, "", `#${e.target.value}`); }} className="w-full rounded-lg border border-line bg-surface px-3 py-2">
            {CALCULATORS.map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}
          </select>
        </label>
        <div className="hidden lg:block space-y-4 text-sm">
          {groups.map((g) => (
            <div key={g}>
              <p className="text-xs text-faint mb-1">{g}</p>
              <ul>
                {CALCULATORS.filter((c) => c.group === g).map((c) => (
                  <li key={c.id}>
                    <a href={`#${c.id}`} aria-current={c.id === id ? "true" : undefined}
                      className={`block rounded-lg px-2 py-1 ${c.id === id ? "bg-surface-2 text-ink font-medium" : "text-muted hover:text-ink"}`}>{c.title}</a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </nav>
      <CalculatorView key={calc.id} calc={calc} lessonTitle={lessonTitles[calc.lesson]} />
    </div>
  );
}
