// Lists upstream lessons we copied (mode "import" or "enrich") that changed since the pinned commit.
// Usage: node scripts/upstream-changes.mjs <path-to-a-clone-of-ai-engineering-from-scratch>
// Review each changed lesson, port what matters, then bump pinned_commit in guide/upstream/source.json.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const GUIDE = path.resolve(import.meta.dirname, "../../guide");
const clone = process.argv[2];
if (!clone) { console.error("usage: node scripts/upstream-changes.mjs <clone-path>"); process.exit(2); }
const { pinned_commit } = JSON.parse(fs.readFileSync(path.join(GUIDE, "upstream/source.json"), "utf8"));
const map = JSON.parse(fs.readFileSync(path.join(GUIDE, "upstream/map.json"), "utf8"));
const git = (...a) => execFileSync("git", ["-C", clone, ...a], { encoding: "utf8" });

git("fetch", "--quiet", "origin");
const head = git("rev-parse", "origin/HEAD").trim();
const changed = new Set(git("diff", "--name-only", pinned_commit, head, "--", "phases").split("\n").filter(Boolean)
  .filter((f) => !f.includes("/docs/") || f.endsWith("/docs/en.md")) // English only
  .map((f) => f.split("/").slice(0, 3).join("/")));
const owned = map.filter((e) => (e.mode === "import" || e.mode === "enrich") && changed.has(e.upstream));
console.log(`pinned ${pinned_commit.slice(0, 7)} → upstream ${head.slice(0, 7)}: ${owned.length} owned lesson(s) changed`);
for (const e of owned) console.log(`  [${e.mode}] ${e.upstream}  →  ${e.target ?? e.target_module}`);
