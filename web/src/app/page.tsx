import Link from "next/link";
import { ArrowRight, GithubLogo } from "@phosphor-icons/react/dist/ssr";
import { LifecycleMap } from "@/components/LifecycleMap";
import { ContinueCard, RecentList, SkillBars } from "@/components/ProgressWidgets";
import { getLessonIndex, getModules, getQuizzes, getSprints, REPO_URL } from "@/lib/content";
import { LAB_AREAS, LABS } from "@/lib/labs";

export default function Home() {
  const lessons = getLessonIndex();
  const modules = getModules();
  const quizLesson = Object.fromEntries(getQuizzes().map((q) => [q.id, q.lesson ?? q.id]));
  const featured = ["fit-explorer", "gradient-descent", "attention", "rag", "kv-cache", "agent-loop"].map((id) => LABS.find((l) => l.id === id)!);

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6">
      <section className="pt-14 pb-12 md:pt-20 grid gap-10 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)] items-start">
        <div>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-semibold tracking-tighter leading-[1.05]">
            From linear regression to LLM agents in production.
          </h1>
          <p className="mt-5 text-lg text-muted max-w-[52ch]">
            A visual, math-first course on how ML and AI systems work. Go deep, skim in quick-read mode, or prepare for a specific interview loop.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link href="/path/" className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 font-medium text-white dark:text-zinc-950 active:scale-[0.98]">
              Start learning <ArrowRight size={16} />
            </Link>
            <Link href="/labs/" className="inline-flex items-center gap-2 rounded-lg border border-line px-4 py-2.5 font-medium hover:bg-surface-2">
              Open the labs
            </Link>
          </div>
        </div>
        <div className="space-y-4">
          <ContinueCard lessons={lessons} />
          <div className="rounded-xl border border-line bg-surface p-5">
            <p className="text-sm font-medium mb-3">Skill progress</p>
            <SkillBars lessons={lessons} quizLesson={quizLesson} />
          </div>
        </div>
      </section>

      <section className="py-12 border-t border-line" aria-labelledby="ways">
        <h2 id="ways" className="text-2xl font-semibold tracking-tight">Four ways to use it</h2>
        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { title: "Deep dive", href: "/path/", cta: "Full course", body: `${lessons.length} lessons in ${modules.length} modules. Intuition, an interactive lab, the math with every step, code, and engineering trade-offs.` },
            { title: "Quick read", href: "/path/", cta: "Pick a lesson", body: `Flip any lesson to Quick read: intuition, the lab, the summary, and interview questions in about a quarter of the time.` },
            { title: "Interview prep", href: "/sprints/", cta: "Choose your role", body: "Seven-day sprints for ML engineer, AI engineer, applied scientist, and data scientist loops, plus a rapid-fire drill." },
            { title: "Quick reference", href: "/cheatsheets/", cta: "Cheat sheets", body: "Cheat sheets, model cards, the glossary, and the Math Lab when you need one fact fast." },
          ].map((w) => (
            <Link key={w.title} href={w.href} className="group rounded-xl border border-line bg-surface p-5 hover:border-faint flex flex-col">
              <span className="font-semibold group-hover:text-accent">{w.title}</span>
              <span className="mt-1 text-sm text-muted leading-snug flex-1">{w.body}</span>
              <span className="mt-3 inline-flex items-center gap-1 text-sm text-accent">{w.cta} <ArrowRight size={14} /></span>
            </Link>
          ))}
        </div>
        <p className="mt-4 text-sm text-muted">
          Interview sprints: {getSprints().map((s, i) => (
            <span key={s.id}>{i > 0 && " · "}<Link href={`/sprints/${s.id}/`} className="text-accent hover:underline">{s.role}</Link></span>
          ))} · <Link href="/drill/" className="text-accent hover:underline">Rapid-fire drill</Link>
        </p>
      </section>

      <section className="py-12 border-t border-line" aria-labelledby="connects">
        <h2 id="connects" className="text-2xl font-semibold tracking-tight">How everything connects</h2>
        <p className="mt-2 text-muted max-w-[65ch]">
          Every model in this course is a step in the same loop. Data becomes a representation, a model makes a prediction, a loss scores it, optimization improves it, and evaluation decides whether it ships. Click any stage or branch to open its lesson.
        </p>
        <div className="mt-8"><LifecycleMap /></div>
      </section>

      <section className="py-12 border-t border-line grid gap-10 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]" aria-labelledby="labs">
        <div>
          <h2 id="labs" className="text-2xl font-semibold tracking-tight">Interactive labs</h2>
          <p className="mt-2 text-muted">{LABS.length} labs across {Object.keys(LAB_AREAS).length} areas. Every number on screen is computed live, in your browser.</p>
          <div className="mt-6 grid sm:grid-cols-2 gap-3">
            {featured.map((l) => (
              <Link key={l.id} href={`/labs/${l.id}/`} className="group rounded-xl border border-line bg-surface p-4 hover:border-faint">
                <span className="text-xs text-faint">{LAB_AREAS[l.area].title}</span>
                <span className="block font-semibold group-hover:text-accent">{l.title}</span>
                <span className="block text-sm text-muted mt-1 leading-snug">{l.description}</span>
              </Link>
            ))}
          </div>
        </div>
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Recently viewed</h2>
          <div className="mt-3"><RecentList /></div>
          <h2 className="text-lg font-semibold tracking-tight mt-10">Prefer plain markdown?</h2>
          <p className="mt-2 text-sm text-muted">
            The full course, with the same lessons, figures, runnable Python, quizzes, and model cards, reads directly on GitHub. No build step, no JavaScript.
          </p>
          <a href={`${REPO_URL}/tree/main/guide`} className="mt-3 inline-flex items-center gap-1.5 text-sm text-accent hover:underline"><GithubLogo size={16} /> Open the GitHub edition</a>
        </div>
      </section>

      <section className="py-12 border-t border-line" aria-labelledby="overview">
        <h2 id="overview" className="text-2xl font-semibold tracking-tight">The full path</h2>
        <p className="mt-2 text-muted">{modules.length} modules, {lessons.length} lessons.</p>
        <ol className="mt-6 grid sm:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-1">
          {modules.map((m) => (
            <li key={m.id}>
              <Link href={`/modules/${m.id}/`} className="flex gap-3 py-2 group">
                <span className="font-mono text-xs text-faint pt-1 w-5">{m.number}</span>
                <span>
                  <span className="font-medium group-hover:text-accent">{m.title}</span>
                  <span className="block text-xs text-faint">{m.lessons.length} lesson{m.lessons.length === 1 ? "" : "s"}</span>
                </span>
              </Link>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
