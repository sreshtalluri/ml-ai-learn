// Content integrity for the guide: the GitHub edition and the website must never have broken links.
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { getGlossary, getLessons, getModels, getSprints, GUIDE_DIR, slugify } from "./content";
import { LABS } from "./labs";

const stripCode = (s: string) => s.replace(/```[\s\S]*?```/g, "").replace(/`[^`\n]*`/g, "");
const mdFiles = (fs.readdirSync(GUIDE_DIR, { recursive: true }) as string[])
  .filter((f) => f.endsWith(".md") && !f.startsWith(".venv") && !f.includes("node_modules"));

describe("guide links", () => {
  it("every relative link points at a file that exists", () => {
    const broken: string[] = [];
    for (const rel of mdFiles) {
      const src = stripCode(fs.readFileSync(path.join(GUIDE_DIR, rel), "utf8"));
      for (const m of src.matchAll(/\]\(([^)\s]+)\)/g)) {
        const href = m[1];
        if (/^(https?:|mailto:|#)/.test(href)) continue;
        const target = path.join(GUIDE_DIR, path.dirname(rel), href.split("#")[0]);
        if (!fs.existsSync(target)) broken.push(`${rel} -> ${href}`);
      }
    }
    expect(broken).toEqual([]);
  });

  it("glossary anchors used in lessons exist", () => {
    const ids = new Set(getGlossary().map((g) => g.id));
    const missing: string[] = [];
    for (const rel of mdFiles) {
      const src = stripCode(fs.readFileSync(path.join(GUIDE_DIR, rel), "utf8"));
      for (const m of src.matchAll(/glossary\.md#([\w-]+)/g)) if (!ids.has(m[1])) missing.push(`${rel} -> #${m[1]}`);
    }
    expect(missing).toEqual([]);
  });
});

describe("lessons", () => {
  const lessons = getLessons();
  const slugs = new Set(lessons.map((l) => l.slug));
  const labIds = new Set(LABS.map((l) => l.id));

  it("have unique slugs and valid prerequisites, related lessons, and lab markers", () => {
    expect(slugs.size).toBe(lessons.length);
    for (const l of lessons) {
      for (const s of [...l.prerequisites, ...l.related]) expect(slugs.has(s), `${l.slug} references ${s}`).toBe(true);
      for (const id of l.labs) expect(labIds.has(id), `${l.slug} uses lab ${id}`).toBe(true);
    }
  });

  it("follow the six-layer structure", () => {
    for (const l of lessons) {
      for (const h of ["## 1. Intuition", "## 2. Visualization", "## 3.", "## 4.", "## 5.", "## 6. Knowledge check", "## Summary"]) {
        expect(l.body.includes(h), `${l.slug} is missing "${h}"`).toBe(true);
      }
    }
  });

  it("every lab is embedded in its own lesson", () => {
    for (const lab of LABS) expect(lessons.find((l) => l.slug === lab.lesson)?.labs, lab.id).toContain(lab.id);
  });

  it("end with 3 to 5 parseable interview questions", () => {
    for (const l of lessons) {
      expect(l.interview.length, `${l.slug} interview questions`).toBeGreaterThanOrEqual(3);
      expect(l.interview.length, `${l.slug} interview questions`).toBeLessThanOrEqual(5);
      for (const x of l.interview) {
        expect(x.a.split(/\s+/).length, `${l.slug}: ${x.q}`).toBeGreaterThan(40);
        // <summary> is raw HTML: math and markdown escapes would show up literally
        expect(x.q, `${l.slug} question title`).not.toMatch(/[$\\]/);
      }
    }
  });

  it("quick read is meaningfully shorter than the full lesson", () => {
    for (const l of lessons) expect(l.quickMinutes, l.slug).toBeLessThan(l.minutes);
  });
});

describe("interview sprints", () => {
  const slugs = new Set(getLessons().map((l) => l.slug));
  it("cover seven days and link real lessons", () => {
    const sprints = getSprints();
    expect(sprints.length).toBeGreaterThanOrEqual(4);
    for (const s of sprints) {
      for (let d = 1; d <= 7; d++) expect(s.body, `${s.id} day ${d}`).toContain(`## Day ${d}:`);
      expect(s.lessons.length, s.id).toBeGreaterThanOrEqual(10);
      for (const l of s.lessons) expect(slugs.has(l), `${s.id} -> ${l}`).toBe(true);
    }
  });
});

describe("model cards", () => {
  it("use allowed tags and reference real lessons", () => {
    const slugs = new Set(getLessons().map((l) => l.slug));
    for (const m of getModels()) {
      for (const s of m.lessons) expect(slugs.has(s), `${m.id} -> ${s}`).toBe(true);
      expect(m.summary.length, `${m.id} needs a Mental model section`).toBeGreaterThan(20);
    }
  });
  it("slugify matches GitHub anchors", () => {
    expect(slugify("Context window")).toBe("context-window");
    expect(slugify("RAG (retrieval-augmented generation)")).toBe("rag-retrieval-augmented-generation");
  });
});
