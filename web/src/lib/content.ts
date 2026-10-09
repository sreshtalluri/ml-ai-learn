// Server-only loader for ../guide. The guide is the single source of truth;
// this file turns its markdown/YAML into typed data for pages.
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import YAML from "yaml";
import { SKILLS, type Skill } from "./skills";
import type { Quiz } from "./quiz-types";
import { isDeepSection, sections } from "./toc";

export const GUIDE_DIR = path.join(process.cwd(), "..", "guide");
export const REPO_URL = "https://github.com/sreshtalluri/ml-ai-learn";

const read = (rel: string) => fs.readFileSync(path.join(GUIDE_DIR, rel), "utf8");
const exists = (rel: string) => fs.existsSync(path.join(GUIDE_DIR, rel));
const list = (rel: string) => (exists(rel) ? fs.readdirSync(path.join(GUIDE_DIR, rel)).sort() : []);

export interface Lesson {
  slug: string;
  title: string;
  summary: string;
  skill: Skill;
  minutes: number;
  prerequisites: string[];
  related: string[];
  moduleId: string;
  file: string;      // guide-relative path
  body: string;
  labs: string[];    // lab ids referenced by markers
  quiz?: string;     // quiz id referenced by marker
  quickMinutes: number; // estimate with math, implementation, and engineering hidden
  interview: InterviewQuestion[];
}

export interface InterviewQuestion { q: string; a: string }

const words = (s: string) => s.split(/\s+/).filter(Boolean).length;

/** Questions in the lesson's "## Interview angle" section: <details><summary><strong>Q</strong></summary> A </details>. */
export function parseInterview(body: string): InterviewQuestion[] {
  const sec = sections(body).find((x) => /^interview angle$/i.test(x.heading));
  if (!sec) return [];
  return [...sec.text.matchAll(/<summary>\s*<strong>([\s\S]+?)<\/strong>\s*<\/summary>([\s\S]*?)<\/details>/g)]
    .map((m) => ({ q: m[1].trim(), a: m[2].trim() }));
}

/** Quick-read estimate: the share of words still visible (deep sections hidden, collapsed answers unopened). */
function quickMinutes(body: string, minutes: number) {
  const total = words(body) || 1;
  const visible = sections(body)
    .filter((x) => !isDeepSection(x.heading))
    .reduce((n, x) => n + words(x.text.replace(/<details>[\s\S]*?<\/details>/g, "")), 0);
  return Math.max(5, Math.round((minutes * visible) / total));
}

export interface Module {
  id: string;        // folder name, e.g. "03-regression"
  number: number;
  title: string;
  summary: string;
  skill: Skill;
  file: string;
  body: string;
  lessons: Lesson[];
}

const markers = (body: string, kind: string) =>
  [...body.matchAll(new RegExp(`<!--\\s*${kind}:([\\w-]+)\\s*-->`, "g"))].map((m) => m[1]);

const asSkill = (s: unknown, where: string): Skill => {
  if (typeof s === "string" && s in SKILLS) return s as Skill;
  throw new Error(`${where}: unknown skill "${String(s)}"`);
};

let modulesCache: Module[] | null = null;

export function getModules(): Module[] {
  if (modulesCache) return modulesCache;
  modulesCache = list("lessons")
    .filter((d) => /^\d\d-/.test(d))
    .map((id) => {
      const file = `lessons/${id}/README.md`;
      const { data, content } = matter(exists(file) ? read(file) : "");
      const lessons = list(`lessons/${id}`)
        .filter((f) => /^\d\d-.+\.md$/.test(f))
        .map((f): Lesson => {
          const lf = `lessons/${id}/${f}`;
          const { data: d, content: body } = matter(read(lf));
          const quizzes = markers(body, "quiz");
          return {
            slug: f.replace(/^\d\d-/, "").replace(/\.md$/, ""),
            title: d.title ?? f,
            summary: d.summary ?? "",
            skill: asSkill(d.skill ?? data.skill, lf),
            minutes: Number(d.minutes ?? 20),
            prerequisites: d.prerequisites ?? [],
            related: d.related ?? [],
            moduleId: id,
            file: lf,
            body,
            labs: markers(body, "lab"),
            quiz: quizzes[0],
            quickMinutes: quickMinutes(body, Number(d.minutes ?? 20)),
            interview: parseInterview(body),
          };
        });
      return {
        id,
        number: Number(id.slice(0, 2)),
        title: data.title ?? id,
        summary: data.summary ?? "",
        skill: asSkill(data.skill ?? lessons[0]?.skill ?? "math", file),
        file,
        body: content,
        lessons,
      };
    });
  return modulesCache;
}

export const getModule = (id: string) => getModules().find((m) => m.id === id);
export const getLessons = () => getModules().flatMap((m) => m.lessons);
export const getLesson = (slug: string) => getLessons().find((l) => l.slug === slug);

export function getAdjacent(slug: string) {
  const all = getLessons();
  const i = all.findIndex((l) => l.slug === slug);
  return { prev: all[i - 1], next: all[i + 1] };
}

/** Compact lesson index for client components (no bodies). */
export const getLessonIndex = () =>
  getLessons().map(({ slug, title, skill, moduleId, minutes, summary }) => ({ slug, title, skill, moduleId, minutes, summary }));
export type LessonIndex = ReturnType<typeof getLessonIndex>;

// ---------- glossary ----------

export interface GlossaryEntry {
  id: string;
  term: string;
  fields: Record<string, string>; // "Plain English", "Formal", "Example", "Related", "Lesson"
}

let glossaryCache: GlossaryEntry[] | null = null;

export function getGlossary(): GlossaryEntry[] {
  if (glossaryCache) return glossaryCache;
  if (!exists("glossary.md")) return (glossaryCache = []);
  glossaryCache = read("glossary.md")
    .split(/^### /m)
    .slice(1)
    .map((chunk) => {
      const [termLine, ...rest] = chunk.split("\n");
      const term = termLine.trim();
      const fields: Record<string, string> = {};
      for (const m of rest.join("\n").matchAll(/^\*\*([^*:]+):\*\*\s*(.+)$/gm)) fields[m[1].trim()] = m[2].trim();
      return { id: slugify(term), term, fields };
    });
  return glossaryCache;
}

/** GitHub-compatible heading anchor. */
export function slugify(s: string): string {
  return s.toLowerCase().trim().replace(/[^\p{L}\p{N}\s-]/gu, "").replace(/\s/g, "-");
}

// ---------- models ----------

export const MODEL_TAGS = [
  "supervised", "unsupervised", "self-supervised", "regression", "classification", "clustering",
  "dimensionality-reduction", "nlp", "vision", "generative", "interpretable", "low-latency", "small-data",
] as const;

export interface ModelCard {
  id: string;
  name: string;
  tags: string[];
  lessons: string[];
  labs: string[];
  summary: string; // the "Mental model" section, for cards
  file: string;
  body: string;
}

let modelsCache: ModelCard[] | null = null;

export function getModels(): ModelCard[] {
  if (modelsCache) return modelsCache;
  modelsCache = list("models")
    .filter((f) => f.endsWith(".md") && f !== "README.md")
    .map((f) => {
      const file = `models/${f}`;
      const { data, content } = matter(read(file));
      const mental = content.match(/^## Mental model\s*\n+([\s\S]*?)(?=\n## |\n*$)/m)?.[1]?.trim() ?? "";
      return {
        id: f.replace(/\.md$/, ""),
        name: data.name ?? f,
        tags: data.tags ?? [],
        lessons: data.lessons ?? [],
        labs: data.labs ?? [],
        summary: mental,
        file,
        body: content,
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
  return modelsCache;
}

export const getModel = (id: string) => getModels().find((m) => m.id === id);

// ---------- cheat sheets ----------

export interface CheatSheet { id: string; title: string; summary: string; file: string; body: string }

export function getCheatSheets(): CheatSheet[] {
  return list("cheatsheets")
    .filter((f) => f.endsWith(".md") && f !== "README.md")
    .map((f) => {
      const file = `cheatsheets/${f}`;
      const { data, content } = matter(read(file));
      return { id: f.replace(/\.md$/, ""), title: data.title ?? f, summary: data.summary ?? "", file, body: content };
    });
}

export const getCheatSheet = (id: string) => getCheatSheets().find((c) => c.id === id);

// ---------- quizzes ----------

let quizCache: Quiz[] | null = null;

export function getQuizzes(): Quiz[] {
  if (quizCache) return quizCache;
  quizCache = list("quizzes")
    .filter((f) => f.endsWith(".yml"))
    .map((f) => ({ id: f.replace(/\.yml$/, ""), ...(YAML.parse(read(`quizzes/${f}`)) as Omit<Quiz, "id">) }));
  return quizCache;
}

export const getQuiz = (id: string) => getQuizzes().find((q) => q.id === id);

// ---------- interview sprints ----------

export interface Sprint { id: string; title: string; role: string; summary: string; order: number; file: string; body: string; lessons: string[] }

export function getSprints(): Sprint[] {
  return list("sprints")
    .filter((f) => f.endsWith(".md") && f !== "README.md")
    .map((f) => {
      const file = `sprints/${f}`;
      const { data, content } = matter(read(file));
      // lessons in the order the sprint links them
      const lessons = [...new Set([...content.matchAll(/\]\(\.\.\/lessons\/[^/]+\/\d\d-([\w-]+)\.md/g)].map((m) => m[1]))];
      return { id: f.replace(/\.md$/, ""), title: data.title ?? f, role: data.role ?? "", summary: data.summary ?? "", order: Number(data.order ?? 99), file, body: content, lessons };
    })
    .sort((a, b) => a.order - b.order);
}

export const getSprint = (id: string) => getSprints().find((s) => s.id === id);

// ---------- generic documents ----------

export function getDoc(rel: string) {
  if (!exists(rel)) return null;
  const { data, content } = matter(read(rel));
  return { data, body: content, file: rel };
}

// ---------- link resolution ----------

/**
 * Map a guide-relative file path (+ optional #hash) to a site route.
 * Returns null for files the site doesn't render (those link to GitHub).
 */
export function routeForFile(rel: string): string | null {
  const norm = path.posix.normalize(rel);
  let m: RegExpMatchArray | null;
  if (norm === "README.md") return "/path/";
  if (norm === "glossary.md") return "/glossary/";
  if (norm === "learning-paths.md") return "/learning-paths/";
  if ((m = norm.match(/^lessons\/([^/]+)\/README\.md$/))) return `/modules/${m[1]}/`;
  if ((m = norm.match(/^lessons\/[^/]+\/\d\d-(.+)\.md$/))) return `/learn/${m[1]}/`;
  if ((m = norm.match(/^models\/(.+)\.md$/)) && m[1] !== "README") return `/models/${m[1]}/`;
  if (norm === "models/README.md") return "/models/";
  if ((m = norm.match(/^cheatsheets\/(.+)\.md$/)) && m[1] !== "README") return `/cheatsheets/${m[1]}/`;
  if (norm === "cheatsheets/README.md") return "/cheatsheets/";
  if ((m = norm.match(/^quizzes\/(.+)\.md$/))) return `/quizzes/${m[1]}/`;
  if (norm === "sprints/README.md") return "/sprints/";
  if ((m = norm.match(/^sprints\/(.+)\.md$/))) return `/sprints/${m[1]}/`;
  return null;
}
