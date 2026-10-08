"""Comparing two LLM system versions on a SYNTHETIC eval set: per-case scores, paired bootstrap, failure categories, latency and cost.

Run from guide/:  uv run code/17-llm-evaluation/eval.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

rng = np.random.default_rng(7)
n = 40
# Per-case correctness (1/0) for two versions. v2 is better on retrieval-heavy cases. SYNTHETIC.
v1 = (rng.random(n) < 0.70).astype(int)
v2 = v1.copy()
flip_up = rng.choice(np.where(v1 == 0)[0], 6, replace=False)
flip_down = rng.choice(np.where(v1 == 1)[0], 2, replace=False)
v2[flip_up] = 1
v2[flip_down] = 0
print(f"accuracy v1 = {v1.mean():.3f}, v2 = {v2.mean():.3f}, difference = {v2.mean() - v1.mean():+.3f}")
print(f"cases v2 fixed: {len(flip_up)}, cases v2 broke: {len(flip_down)}")

boot = []
for _ in range(10000):
    idx = rng.integers(0, n, n)
    boot.append(v2[idx].mean() - v1[idx].mean())
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"paired bootstrap 95% CI for the difference: [{lo:+.3f}, {hi:+.3f}]")
unpaired = []
for _ in range(10000):
    unpaired.append(v2[rng.integers(0, n, n)].mean() - v1[rng.integers(0, n, n)].mean())
ulo, uhi = np.percentile(unpaired, [2.5, 97.5])
print(f"unpaired bootstrap 95% CI (ignores pairing): [{ulo:+.3f}, {uhi:+.3f}]")

categories = Counter({"retrieval miss": 6, "unsupported claim": 2, "wrong format": 1, "refused unnecessarily": 1})
v2_categories = Counter({"retrieval miss": 1, "unsupported claim": 2, "wrong format": 1, "refused unnecessarily": 2})
lat1 = rng.lognormal(np.log(1.2), 0.35, n)
lat2 = rng.lognormal(np.log(1.5), 0.35, n)
cost1 = (1800 * 3 + 250 * 15) / 1e6       # illustrative per-request tokens x price per million: input 3, output 15
cost2 = (3200 * 3 + 260 * 15) / 1e6
print(f"p50/p95 latency v1 {np.percentile(lat1, 50):.2f}/{np.percentile(lat1, 95):.2f}s  v2 {np.percentile(lat2, 50):.2f}/{np.percentile(lat2, 95):.2f}s")
print(f"cost per request (illustrative prices) v1 ${cost1:.4f}  v2 ${cost2:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
ax = axes[0]
ax.hist(boot, bins=40, color=PURPLE, alpha=0.7, label="paired")
ax.hist(unpaired, bins=40, color=GRAY, alpha=0.5, label="unpaired")
ax.axvline(0, color="black", lw=1)
ax.set(title="Bootstrap distribution of accuracy(v2) − accuracy(v1)", xlabel="difference", ylabel="count")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
labels = list(categories)
y = np.arange(len(labels))
ax.barh(y - 0.2, [categories[k] for k in labels], height=0.4, color=ORANGE, label="v1")
ax.barh(y + 0.2, [v2_categories[k] for k in labels], height=0.4, color=TEAL, label="v2")
ax.set_yticks(y, labels)
ax.set(title="Failure categories (40 cases)", xlabel="failed cases")
ax.legend(frameon=False, fontsize=8)
ax = axes[2]
ax.boxplot([lat1, lat2], tick_labels=["v1", "v2 (more context)"])
ax.set(title=f"Latency; cost ${cost1:.4f} vs ${cost2:.4f} per request", ylabel="seconds")
for a in axes:
    a.title.set_fontsize(10)
save(fig, "llm-evaluation")
