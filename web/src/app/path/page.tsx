import type { Metadata } from "next";
import Link from "next/link";
import { DoneMark } from "@/components/PageControls";
import { ModuleProgress } from "@/components/ProgressWidgets";
import { getModules } from "@/lib/content";
import { SKILLS } from "@/lib/skills";

export const metadata: Metadata = { title: "Learning path", description: "The guided sequence from math foundations to production AI engineering." };

const PHASES = [
  { title: "Foundations", range: [0, 2], blurb: "Math, vocabulary, and how to run a valid experiment." },
  { title: "Classical machine learning", range: [3, 8], blurb: "Regression, classification, neighbors, trees, clustering, and PCA." },
  { title: "NLP and deep learning", range: [9, 13], blurb: "Text as vectors, neural networks, gradients, and architectures." },
  { title: "Transformers and LLMs", range: [14, 17], blurb: "Attention, pretraining, decoding, adaptation, RAG, and evaluation." },
  { title: "AI engineering", range: [18, 20], blurb: "Production systems, security, and a portfolio of projects." },
];

export default function PathPage() {
  const modules = getModules();
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Learning path</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">
        Guided mode: work top to bottom. Each lesson moves from intuition to visualization, math, code, and engineering, then ends with a knowledge check.
        Reference mode: jump straight to any lesson, <Link href="/models/" className="text-accent hover:underline">model</Link>, or <Link href="/cheatsheets/" className="text-accent hover:underline">cheat sheet</Link>.
      </p>
      <p className="mt-2 text-sm text-faint">Suggested pace: 12 weeks at 5 to 7 hours per week. See the <Link href="/learn/twelve-week-plan/" className="underline">12-week plan</Link>.</p>

      <div className="mt-12 space-y-14">
        {PHASES.map((ph) => (
          <section key={ph.title} aria-labelledby={ph.title}>
            <h2 id={ph.title} className="text-xl font-semibold tracking-tight">{ph.title}</h2>
            <p className="text-sm text-muted mt-1">{ph.blurb}</p>
            <div className="mt-5 space-y-4">
              {modules.filter((m) => m.number >= ph.range[0] && m.number <= ph.range[1]).map((m) => (
                <div key={m.id} className="rounded-xl border border-line bg-surface">
                  <div className="flex flex-wrap items-baseline justify-between gap-2 px-5 pt-4">
                    <Link href={`/modules/${m.id}/`} className="group">
                      <span className="text-xs font-mono text-faint">Module {m.number}</span>
                      <h3 className="font-semibold group-hover:text-accent">{m.title}</h3>
                    </Link>
                    <ModuleProgress slugs={m.lessons.map((l) => l.slug)} />
                  </div>
                  <p className="px-5 mt-1 text-sm text-muted">{m.summary}</p>
                  <ul className="mt-3 border-t border-line">
                    {m.lessons.map((l) => (
                      <li key={l.slug}>
                        <Link href={`/learn/${l.slug}/`} className="flex items-center gap-3 px-5 py-2.5 hover:bg-surface-2/60">
                          <DoneMark slug={l.slug} />
                          <span className="flex-1">{l.title}</span>
                          {l.labs.length > 0 && <span className="text-xs rounded-full bg-accent-soft text-accent px-2 py-0.5">lab</span>}
                          <span className="text-xs text-faint hidden sm:inline" style={{ color: SKILLS[l.skill].color }}>{SKILLS[l.skill].label}</span>
                          <span className="text-xs text-faint w-14 text-right">{l.minutes} min</span>
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
