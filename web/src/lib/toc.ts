import GithubSlugger from "github-slugger";

export interface TocItem { depth: 2 | 3; text: string; id: string }

/** Headings in document order, slugged exactly like rehype-slug (shared slugger across all levels). */
export function toc(markdown: string): TocItem[] {
  const slugger = new GithubSlugger();
  const out: TocItem[] = [];
  let fenced = false;
  for (const line of markdown.split("\n")) {
    if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
    if (fenced) continue;
    const m = line.match(/^(#{1,6})\s+(.+?)\s*#*$/);
    if (!m) continue;
    const text = m[2].replace(/\[([^\]]+)\]\([^)]*\)/g, "$1").replace(/[*_`]/g, "").replace(/\$([^$]+)\$/g, "$1");
    const id = slugger.slug(text);
    if (m[1].length === 2 || m[1].length === 3) out.push({ depth: m[1].length as 2 | 3, text, id });
  }
  return out;
}

/** Remove the leading "# Title" line; pages render the title from frontmatter. */
export const stripH1 = (md: string) => md.replace(/^\s*# .+\n/, "");

/** Lesson sections hidden in quick mode: math, implementation, engineering, and the knowledge check (sections 3 to 6). */
export const isDeepSection = (heading: string) => /^[3-6]\.\s/.test(heading.trim());

/** Split a lesson body into `##` sections (the preamble before the first one has heading ""). */
export function sections(markdown: string): { heading: string; text: string }[] {
  const out = [{ heading: "", text: "" }];
  let fenced = false;
  for (const line of markdown.split("\n")) {
    if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
    const m = !fenced && line.match(/^##\s+(.+?)\s*$/);
    if (m) out.push({ heading: m[1], text: "" });
    else out[out.length - 1].text += line + "\n";
  }
  return out;
}

