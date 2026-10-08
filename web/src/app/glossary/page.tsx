import type { Metadata } from "next";
import { GlossarySearch } from "@/components/GlossarySearch";
import path from "node:path";
import { getGlossary, routeForFile } from "@/lib/content";
import { BASE_PATH } from "@/lib/site";

export const metadata: Metadata = { title: "Glossary and formulas", description: "Plain-English and formal definitions for every key term, with examples and links to lessons." };

export default function GlossaryPage() {
  // Resolve markdown links inside fields ([x](#y) and [x](lessons/...md)) to site routes.
  const link = (s: string) =>
    s.replace(/\]\(([^)]+)\)/g, (_m, href: string) => {
      if (href.startsWith("#") || /^https?:/.test(href)) return `](${href})`;
      const [p, hash] = href.split("#");
      const route = routeForFile(path.posix.normalize(p));
      return `](${route ? BASE_PATH + route : href}${hash ? `#${hash}` : ""})`;
    });
  const entries = getGlossary().map((g) => ({
    id: g.id,
    term: g.term,
    fields: Object.fromEntries(Object.entries(g.fields).map(([k, v]) => [k, link(v)])),
  }));
  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 pt-12">
      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">Glossary and formula reference</h1>
      <p className="mt-3 text-lg text-muted max-w-[65ch]">Every term in plain English first, then formally, with a small example and the lesson that teaches it.</p>
      <GlossarySearch entries={entries} />
    </div>
  );
}
