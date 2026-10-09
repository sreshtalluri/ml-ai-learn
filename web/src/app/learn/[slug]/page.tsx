import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, ArrowRight, Clock, GithubLogo } from "@phosphor-icons/react/dist/ssr";
import { Markdown } from "@/components/Markdown";
import { BookmarkButton, CompleteButton, NoteBox, VisitTracker } from "@/components/PageControls";
import { QuickToggle, ShowFullButton } from "@/components/QuickToggle";
import { getAdjacent, getLesson, getLessons, getModule, REPO_URL } from "@/lib/content";
import { SKILLS } from "@/lib/skills";
import { isDeepSection, stripH1, toc } from "@/lib/toc";

export function generateStaticParams() {
  return getLessons().map((l) => ({ slug: l.slug }));
}

export async function generateMetadata({ params }: PageProps<"/learn/[slug]">): Promise<Metadata> {
  const l = getLesson((await params).slug);
  return l ? { title: l.title, description: l.summary } : {};
}

export default async function LessonPage({ params }: PageProps<"/learn/[slug]">) {
  const { slug } = await params;
  const lesson = getLesson(slug);
  if (!lesson) notFound();
  const mod = getModule(lesson.moduleId)!;
  const { prev, next } = getAdjacent(slug);
  const body = stripH1(lesson.body);
  const headings = toc(body);
  // a ### entry is hidden in quick mode when its parent ## section is
  const items = headings.map((t, i) => ({ ...t, deep: isDeepSection(headings.slice(0, i + 1).findLast((h) => h.depth === 2)?.text ?? "") }));
  const path = `/learn/${slug}/`;
  const prereqs = lesson.prerequisites.map((s) => getLesson(s)).filter((l) => l !== undefined);
  const related = lesson.related.map((s) => getLesson(s)).filter((l) => l !== undefined);

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 pt-8 lg:grid lg:grid-cols-[14rem_minmax(0,1fr)] xl:grid-cols-[14rem_minmax(0,1fr)_14rem] lg:gap-10">
      <VisitTracker path={path} title={lesson.title} />

      {/* Left: module navigation */}
      <nav aria-label="Module lessons" className="hidden lg:block">
        <div className="sticky top-24 text-sm">
          <Link href={`/modules/${mod.id}/`} className="text-faint hover:text-ink">Module {mod.number}</Link>
          <p className="font-medium mt-0.5 mb-3">{mod.title}</p>
          <ul className="space-y-1 border-l border-line">
            {mod.lessons.map((l) => (
              <li key={l.slug}>
                <Link href={`/learn/${l.slug}/`} aria-current={l.slug === slug ? "page" : undefined}
                  className={`block -ml-px border-l pl-3 py-1 ${l.slug === slug ? "border-accent text-ink font-medium" : "border-transparent text-muted hover:text-ink"}`}>
                  {l.title}
                </Link>
              </li>
            ))}
          </ul>
          <Link href="/path/" className="inline-block mt-6 text-faint hover:text-ink">All modules</Link>
        </div>
      </nav>

      <article className="min-w-0 max-w-3xl">
        <header className="mb-8">
          <p className="text-sm text-muted flex flex-wrap items-center gap-x-3 gap-y-1">
            <Link href={`/modules/${mod.id}/`} className="hover:text-ink">Module {mod.number}: {mod.title}</Link>
            <span className="inline-flex items-center gap-1"><Clock size={14} /> <span className="full-only">{lesson.minutes} min</span><span className="quick-only">{lesson.quickMinutes} min quick read</span></span>
            <span style={{ color: SKILLS[lesson.skill].color }}>{SKILLS[lesson.skill].label}</span>
          </p>
          <h1 className="mt-2 text-3xl md:text-4xl font-semibold tracking-tight">{lesson.title}</h1>
          <p className="mt-3 text-lg text-muted max-w-[65ch]">{lesson.summary}</p>
          {prereqs.length > 0 && (
            <p className="mt-4 text-sm text-muted">
              Before this: {prereqs.map((p, i) => (
                <span key={p.slug}>{i > 0 && ", "}<Link href={`/learn/${p.slug}/`} className="text-accent hover:underline">{p.title}</Link></span>
              ))}
            </p>
          )}
          <div className="mt-5 flex flex-wrap gap-2">
            <QuickToggle />
            <CompleteButton slug={slug} />
            <BookmarkButton path={path} title={lesson.title} />
            <a href={`${REPO_URL}/blob/main/guide/${lesson.file}`} className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-sm text-muted hover:text-ink">
              <GithubLogo size={16} /> Read on GitHub
            </a>
          </div>
        </header>

        <p className="quick-only mb-8 rounded-xl border border-line bg-surface px-4 py-3 text-sm text-muted">
          Quick read: the intuition, interactive visualization, summary, and interview questions. The math, implementation, engineering, and knowledge-check sections are hidden. <ShowFullButton />
        </p>
        <Markdown source={body} file={lesson.file} lesson />

        <footer className="mt-12 space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <CompleteButton slug={slug} />
            <span className="text-sm text-muted">Marking lessons complete feeds your progress dashboard.</span>
          </div>
          {lesson.interview.length > 0 && (
            <p className="text-sm text-muted">
              <Link href={`/drill/?lesson=${slug}`} className="text-accent hover:underline">Drill this lesson&apos;s {lesson.interview.length} interview questions</Link> as rapid-fire flashcards.
            </p>
          )}
          <NoteBox path={path} />
          {related.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-muted mb-2">Related lessons</h2>
              <div className="flex flex-wrap gap-2">
                {related.map((r) => <Link key={r.slug} href={`/learn/${r.slug}/`} className="rounded-full border border-line px-3 py-1 text-sm hover:border-faint">{r.title}</Link>)}
              </div>
            </div>
          )}
          <nav aria-label="Lesson navigation" className="grid sm:grid-cols-2 gap-3 pt-4">
            {prev ? (
              <Link href={`/learn/${prev.slug}/`} className="rounded-xl border border-line p-4 hover:border-faint">
                <span className="text-xs text-faint inline-flex items-center gap-1"><ArrowLeft size={12} /> Previous</span>
                <span className="block font-medium mt-1">{prev.title}</span>
              </Link>
            ) : <span />}
            {next && (
              <Link href={`/learn/${next.slug}/`} className="rounded-xl border border-line p-4 hover:border-faint sm:text-right">
                <span className="text-xs text-faint inline-flex items-center gap-1">Suggested next <ArrowRight size={12} /></span>
                <span className="block font-medium mt-1">{next.title}</span>
              </Link>
            )}
          </nav>
        </footer>
      </article>

      {/* Right: on this page */}
      <aside aria-label="On this page" className="hidden xl:block">
        <div className="sticky top-24 text-sm">
          <p className="font-medium mb-3">On this page</p>
          <ul className="space-y-1.5">
            {items.map((t) => (
              <li key={t.id} className={t.depth === 3 ? "pl-3" : ""} data-deep-toc={t.deep ? "" : undefined}>
                <a href={`#${t.id}`} className={`hover:text-ink ${t.depth === 3 ? "text-faint" : "text-muted"}`}>{t.text}</a>
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </div>
  );
}
