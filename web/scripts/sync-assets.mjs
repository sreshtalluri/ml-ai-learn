// Copy images from ../guide into public/guide-assets so markdown image paths resolve on the site.
import fs from "node:fs";
import path from "node:path";

const guide = path.resolve(import.meta.dirname, "../../guide");
const out = path.resolve(import.meta.dirname, "../public/guide-assets");
fs.rmSync(out, { recursive: true, force: true });

let count = 0;
for (const rel of fs.readdirSync(guide, { recursive: true })) {
  if (!/\.(png|svg|jpe?g|gif|webp)$/i.test(rel) || rel.includes("node_modules") || rel.startsWith(".venv")) continue;
  const dest = path.join(out, rel);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(path.join(guide, rel), dest);
  count++;
}
console.log(`sync-assets: copied ${count} images`);
