// Generate guide/quizzes/<id>.md (GitHub-readable, answers hidden in <details>) from the canonical .yml.
// Run with --check to fail if any generated file is out of date (used by tests/CI).
import fs from "node:fs";
import path from "node:path";
import YAML from "yaml";

const dir = path.resolve(import.meta.dirname, "../../guide/quizzes");
const check = process.argv.includes("--check");
const LETTERS = "ABCDEFGHIJ";
const TYPE = { single: "Multiple choice", multi: "Select all that apply", numeric: "Calculation", fill: "Fill in", order: "Arrange in order", match: "Match", short: "Reflection" };

export function render(quiz, id) {
  const out = [
    `<!-- GENERATED from ${id}.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->`,
    `# Quiz: ${quiz.title}`,
    "",
    quiz.lesson ? `Covers the lesson [${quiz.title}](${lessonPath(quiz.lesson)}). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/${id}/) grades these interactively and tracks a review queue.` : "",
    "",
  ];
  quiz.questions.forEach((q, i) => {
    out.push(`## ${i + 1}. ${TYPE[q.type]} (${q.difficulty})`, "", q.prompt.trim(), "");
    if (q.type === "order") {
      // show in a fixed scrambled order: reverse-rotate so it is never already solved
      const idx = q.options.map((_, j) => j).reverse();
      idx.forEach((j) => out.push(`- ${q.options[j]}`));
      out.push("");
    } else if (q.options) {
      q.options.forEach((o, j) => out.push(`- **${LETTERS[j]}.** ${o}`));
      out.push("");
    }
    if (q.type === "match") {
      const rights = q.pairs.map((p) => p[1]).reverse();
      out.push("| Concept | Options |", "|---|---|");
      q.pairs.forEach((p, j) => out.push(`| ${p[0]} | ${rights[j]} |`));
      out.push("");
    }
    out.push("<details>", "<summary>Answer</summary>", "");
    if (q.type === "single") out.push(`**${LETTERS[q.answer]}.** ${q.options[q.answer]}`, "");
    if (q.type === "multi") out.push(`**${q.answer.map((a) => LETTERS[a]).join(", ")}**`, "");
    if (q.type === "numeric") out.push(`**${q.answer}**${q.tolerance ? ` (within ±${q.tolerance})` : ""}`, "");
    if (q.type === "fill") out.push(`**${q.answer.join("** or **")}**`, "");
    if (q.type === "order") { q.options.forEach((o, j) => out.push(`${j + 1}. ${o}`)); out.push(""); }
    if (q.type === "match") { q.pairs.forEach((p) => out.push(`- ${p[0]} → ${p[1]}`)); out.push(""); }
    if (q.type === "short" && q.sample_answer) out.push(`**Model answer.** ${q.sample_answer.trim()}`, "");
    out.push(q.explanation.trim(), "");
    if (q.why_not) {
      q.why_not.forEach((w, j) => w && out.push(`- **${LETTERS[j]}:** ${w}`));
      out.push("");
    }
    out.push("</details>", "");
  });
  return out.join("\n").replace(/\n{3,}/g, "\n\n");
}

function lessonPath(slug) {
  const lessons = path.resolve(dir, "../lessons");
  for (const m of fs.readdirSync(lessons)) {
    const full = path.join(lessons, m);
    if (!fs.statSync(full).isDirectory()) continue;
    const f = fs.readdirSync(full).find((x) => x.replace(/^\d\d-/, "") === `${slug}.md`);
    if (f) return `../lessons/${m}/${f}`;
  }
  return "../README.md";
}

const VALID = new Set(Object.keys(TYPE));

export function validate(quiz, id) {
  const errs = [];
  if (!quiz.title) errs.push("missing title");
  if (!Array.isArray(quiz.questions) || quiz.questions.length < 6) errs.push("needs at least 6 questions");
  const ids = new Set();
  for (const q of quiz.questions ?? []) {
    const where = `${id}:${q.id}`;
    if (!q.id || ids.has(q.id)) errs.push(`${where}: missing or duplicate id`);
    ids.add(q.id);
    if (!VALID.has(q.type)) errs.push(`${where}: bad type ${q.type}`);
    if (!["easy", "medium", "hard"].includes(q.difficulty)) errs.push(`${where}: bad difficulty`);
    if (!q.prompt || !q.explanation) errs.push(`${where}: needs prompt and explanation`);
    if (q.type === "single" && !(Number.isInteger(q.answer) && q.options?.[q.answer] !== undefined)) errs.push(`${where}: single needs options and a valid answer index`);
    if (q.type === "multi" && !(Array.isArray(q.answer) && q.answer.every((a) => q.options?.[a] !== undefined))) errs.push(`${where}: multi needs valid answer indices`);
    if (q.type === "numeric" && typeof q.answer !== "number") errs.push(`${where}: numeric needs a number answer`);
    if (q.type === "fill" && !(Array.isArray(q.answer) && q.answer.length)) errs.push(`${where}: fill needs a list of accepted answers`);
    if (q.type === "order" && !(q.options?.length >= 3)) errs.push(`${where}: order needs 3+ options`);
    if (q.type === "match" && !(q.pairs?.length >= 3)) errs.push(`${where}: match needs 3+ pairs`);
    if (q.why_not && q.why_not.length !== q.options?.length) errs.push(`${where}: why_not must have one entry per option`);
  }
  return errs;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  let stale = 0, errors = 0;
  for (const f of fs.readdirSync(dir).filter((x) => x.endsWith(".yml"))) {
    const id = f.replace(/\.yml$/, "");
    const quiz = YAML.parse(fs.readFileSync(path.join(dir, f), "utf8"));
    const errs = validate(quiz, id);
    errs.forEach((e) => console.error(`✗ ${e}`));
    errors += errs.length;
    const md = render(quiz, id);
    const target = path.join(dir, `${id}.md`);
    const current = fs.existsSync(target) ? fs.readFileSync(target, "utf8") : "";
    if (current !== md) {
      stale++;
      if (!check) fs.writeFileSync(target, md);
    }
  }
  if (errors) process.exit(1);
  if (check && stale) {
    console.error(`${stale} quiz markdown file(s) out of date. Run: npm run gen:quizzes`);
    process.exit(1);
  }
  console.log(check ? "quizzes: up to date" : `quizzes: wrote ${stale} file(s)`);
}
