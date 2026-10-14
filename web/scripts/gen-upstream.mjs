// Writes the "Go deeper" / credit block into each lesson from guide/upstream/map.json, so the
// GitHub edition and the website show the same links. Run with --check to fail if any lesson is out of date.
//
// A lesson gets a block when:
//   - upstream lessons in map.json have mode "enrich" or "link" with target = this lesson's slug, or
//   - its frontmatter lists `adapted_from: [phases/...]` (imported or enriched content: credit required by the MIT license).
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";

const GUIDE = path.resolve(import.meta.dirname, "../../guide");
const check = process.argv.includes("--check");
const source = JSON.parse(fs.readFileSync(path.join(GUIDE, "upstream/source.json"), "utf8"));
const map = JSON.parse(fs.readFileSync(path.join(GUIDE, "upstream/map.json"), "utf8"));
const url = (p) => `${source.site}/lesson?path=${p}`;
const byPath = new Map(map.map((e) => [e.upstream, e]));
const START = "<!-- deeper:", END = "<!-- /deeper -->";

const lessonFiles = fs.readdirSync(path.join(GUIDE, "lessons"))
  .filter((m) => /^\d\d-/.test(m))
  .flatMap((m) => fs.readdirSync(path.join(GUIDE, "lessons", m)).filter((f) => /^\d\d-.+\.md$/.test(f)).map((f) => path.join("lessons", m, f)));

export function block(slug, adaptedFrom) {
  const deeper = map.filter((e) => (e.mode === "link" || e.mode === "enrich") && e.target === slug && !adaptedFrom.includes(e.upstream));
  const credit = adaptedFrom.map((p) => byPath.get(p) ?? { upstream: p, title: p.split("/").pop() });
  if (!deeper.length && !credit.length) return "";
  const lines = [`${START} generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->`];
  if (deeper.length) lines.push(`**Go deeper, code-first** ([AI Engineering from Scratch](${source.site})): ${deeper.map((e) => `[${e.title}](${url(e.upstream)})`).join(" · ")}`);
  if (credit.length) {
    if (deeper.length) lines.push("");
    lines.push(`*Parts of this lesson are adapted from [AI Engineering from Scratch](${source.repo}) by Rohit Ghumare (MIT license, [notice](${"../".repeat(2)}upstream/LICENSE-ai-engineering-from-scratch.txt)): ${credit.map((e) => `[${e.title}](${url(e.upstream)})`).join(", ")}.*`);
  }
  lines.push(END);
  return lines.join("\n");
}

let stale = 0;
for (const rel of lessonFiles) {
  const file = path.join(GUIDE, rel);
  const raw = fs.readFileSync(file, "utf8");
  const { data } = matter(raw);
  const slug = path.basename(rel, ".md").replace(/^\d\d-/, "");
  const b = block(slug, data.adapted_from ?? []);
  // the block sits just before "## Interview angle" (or at the end)
  let body = raw.replace(new RegExp(`\\n*${START}[\\s\\S]*?${END.replace(/[/]/g, "\\/")}\\n*`), "\n\n");
  if (b) {
    const at = body.search(/\n## Interview angle/);
    body = at >= 0 ? `${body.slice(0, at).trimEnd()}\n\n${b}\n${body.slice(at)}` : `${body.trimEnd()}\n\n${b}\n`;
  }
  body = body.replace(/\n{3,}/g, "\n\n");
  if (body !== raw) {
    stale++;
    if (check) console.error(`out of date: ${rel}`);
    else fs.writeFileSync(file, body);
  }
}
if (check && stale) { console.error(`${stale} lesson(s) need: npm run gen:upstream`); process.exit(1); }
console.log(check ? "upstream links: up to date" : `upstream links: ${stale} lesson(s) updated`);
