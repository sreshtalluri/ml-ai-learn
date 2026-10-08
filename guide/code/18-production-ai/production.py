"""Figures and arithmetic for production AI: architecture, reliability math, caching, quantization, trust boundaries, and the plan.

Run from guide/:  uv run code/18-production-ai/production.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none", alpha=0.18))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="none", ec=color, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, color="#52525b", style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=1.2))


# ---------------------------------------------------------------------------
# 1. Production architecture diagram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.set(xlim=(0, 12), ylim=(0, 5))
ax.axis("off")
box(ax, 0.2, 2.1, 1.3, 0.8, "Client", GRAY)
box(ax, 2.0, 2.1, 1.6, 0.8, "API gateway\nauth · rate limits", BLUE)
box(ax, 4.2, 2.0, 2.0, 1.0, "Orchestration\nprompts · routing\nretries · schemas", PURPLE)
box(ax, 7.0, 3.6, 1.6, 0.8, "Cache", GRAY)
box(ax, 7.0, 2.4, 1.6, 0.8, "Retrieval\nindex + filters", TEAL)
box(ax, 7.0, 1.2, 1.6, 0.8, "Model\n(API or self-hosted)", ORANGE)
box(ax, 7.0, 0.1, 1.6, 0.8, "Tools\n(allowlisted)", ORANGE)
box(ax, 9.5, 3.4, 2.2, 0.8, "Logging · tracing\nmonitoring", BLUE)
box(ax, 9.5, 2.2, 2.2, 0.8, "Evaluation\noffline + online", PURPLE)
box(ax, 9.5, 1.0, 2.2, 0.8, "Governance\nprivacy · audit · policy", TEAL)
arrow(ax, (1.5, 2.5), (2.0, 2.5))
arrow(ax, (3.6, 2.5), (4.2, 2.5))
for y in (4.0, 2.8, 1.6, 0.5):
    arrow(ax, (6.2, 2.5), (7.0, y), style="<|-|>")
for y in (3.8, 2.6, 1.4):
    arrow(ax, (8.6, 2.5), (9.5, y), color=GRAY)
ax.text(6, 4.75, "A request flows left to right; every component emits traces to the right-hand column", ha="center", fontsize=10)
save(fig, "production-architecture")

# ---------------------------------------------------------------------------
# 2. Reliability, caching, and quantization arithmetic
# ---------------------------------------------------------------------------
p_fail = 0.02
for retries in (0, 1, 2, 3):
    print(f"independent failure 2%, {retries} retries -> request fails with p = {p_fail ** (retries + 1):.2e}")
chain = 0.99 ** 5
print(f"5 dependencies at 99% each in series -> {chain:.4f} availability")
hit = 0.4
cost_per = 0.012
print(f"cache hit rate 40%: cost per request {cost_per * (1 - hit):.4f} vs {cost_per} (saves {hit:.0%}); cached latency ~50 ms vs ~1.5 s")
params = 7e9
for name, bits in [("fp32", 32), ("fp16/bf16", 16), ("int8", 8), ("int4", 4)]:
    print(f"7B weights in {name}: {params * bits / 8 / 1e9:.1f} GB")
backoff = [min(30, 0.5 * 2**k) for k in range(6)]
print("exponential backoff delays (s):", backoff)

rng = np.random.default_rng(3)
lat = rng.lognormal(np.log(1.2), 0.5, 5000)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
ax = axes[0]
ax.hist(lat, bins=80, color=BLUE, alpha=0.7)
for q, col in [(50, TEAL), (95, ORANGE), (99, PURPLE)]:
    v = np.percentile(lat, q)
    ax.axvline(v, color=col, lw=2, label=f"p{q} = {v:.2f}s")
ax.set(title="Latency is skewed: report percentiles, not the mean", xlabel="seconds (synthetic)", ylabel="requests")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
hits = np.linspace(0, 0.9, 50)
ax.plot(hits * 100, cost_per * (1 - hits) * 1e6 / 1000, color=TEAL, lw=2)
ax.set(title="Cost per 1k requests vs cache hit rate (illustrative $0.012/request)", xlabel="cache hit rate (%)", ylabel="dollars per 1k requests")
ax = axes[2]
labels = ["fp32", "fp16", "int8", "int4"]
gb = [params * b / 8 / 1e9 for b in (32, 16, 8, 4)]
ax.bar(labels, gb, color=[GRAY, BLUE, PURPLE, ORANGE])
for i, g in enumerate(gb):
    ax.text(i, g + 0.6, f"{g:.1f} GB", ha="center", fontsize=9)
ax.axhline(24, color="black", ls="--", lw=1)
ax.text(3.45, 25, "24 GB GPU", ha="right", fontsize=8)
ax.set(title="Memory for 7B weights by precision (weights only)", ylabel="GB")
for a in axes:
    a.title.set_fontsize(9)
save(fig, "reliability-cost-and-observability")

# ---------------------------------------------------------------------------
# 3. Trust boundaries for an LLM application
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set(xlim=(0, 12), ylim=(0, 4.6))
ax.axis("off")
ax.add_patch(FancyBboxPatch((0.2, 0.3), 3.6, 3.9, boxstyle="round,pad=0.02", fc="#fee2e2", ec="#dc2626", lw=1.2))
ax.text(2.0, 3.95, "UNTRUSTED input", ha="center", fontsize=10, color="#dc2626")
for i, t in enumerate(["user messages", "retrieved documents", "web pages, emails", "tool outputs"]):
    box(ax, 0.6, 3.1 - i * 0.75, 2.8, 0.55, t, "#dc2626")
ax.add_patch(FancyBboxPatch((4.4, 0.3), 3.2, 3.9, boxstyle="round,pad=0.02", fc="#ede9fe", ec=PURPLE, lw=1.2))
ax.text(6.0, 3.95, "MODEL (can be manipulated)", ha="center", fontsize=10, color=PURPLE)
box(ax, 4.8, 2.4, 2.4, 0.9, "LLM\nsystem prompt =\nonly trusted text", PURPLE)
box(ax, 4.8, 0.7, 2.4, 1.2, "proposes actions\nas structured\ntool calls", PURPLE)
ax.add_patch(FancyBboxPatch((8.2, 0.3), 3.6, 3.9, boxstyle="round,pad=0.02", fc="#dcfce7", ec="#16a34a", lw=1.2))
ax.text(10.0, 3.95, "ENFORCED by code, not the model", ha="center", fontsize=10, color="#16a34a")
for i, t in enumerate(["schema validation", "allowlist + permission check", "least-privilege credentials", "human approval if risky", "audit log"]):
    box(ax, 8.6, 3.25 - i * 0.62, 2.8, 0.48, t, "#16a34a", fs=8)
arrow(ax, (3.8, 2.2), (4.4, 2.2))
arrow(ax, (7.6, 1.3), (8.2, 1.3))
save(fig, "ai-security")

# ---------------------------------------------------------------------------
# 4. The 12-week plan as a timeline
# ---------------------------------------------------------------------------
weeks = ["framing, vectors, probability, splits", "linear/logistic regression, metrics", "KNN, trees, random forest",
         "boosting and XGBoost", "K-means, DBSCAN, PCA", "TF-IDF, n-grams, embeddings", "neural networks, loss, gradients",
         "optimization, regularization, CNN/RNN", "attention and transformers", "tokenization, decoding, fine-tuning",
         "RAG, evaluation, tool use", "deployment, observability, security"]
tracks = [0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5]
colors = [BLUE, TEAL, ORANGE, PURPLE, BLUE, TEAL]
names = ["Foundations", "Classical ML", "NLP", "Deep learning", "Transformers & LLMs", "AI engineering"]
fig, ax = plt.subplots(figsize=(12, 4.4))
for i, (w, t) in enumerate(zip(weeks, tracks)):
    ax.barh(11 - i, 1, left=i, color=colors[t], alpha=0.85)
    ax.text(i + 1.1, 11 - i, w, va="center", fontsize=8)
ax.set(xlim=(0, 20), yticks=[], xticks=range(0, 13), xlabel="week")
ax.set_title("12 weeks at 5 to 7 hours per week", fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(6)]
ax.legend(handles, names, frameon=False, fontsize=8, loc="lower right")
ax.grid(False)
save(fig, "twelve-week-plan")

# ---------------------------------------------------------------------------
# 5. Project ladder skill matrix
# ---------------------------------------------------------------------------
projects = ["1. Tabular failure lab", "2. Incident triage", "3. RAG eval workbench", "4. Model router", "5. Agent sandbox"]
skills = ["classical ML", "NLP/embeddings", "retrieval", "evaluation", "backend/APIs", "observability", "security"]
M = np.array([
    [3, 0, 0, 3, 1, 2, 0],
    [2, 3, 2, 2, 2, 1, 0],
    [0, 2, 3, 3, 2, 2, 1],
    [1, 1, 1, 3, 3, 3, 2],
    [0, 1, 1, 3, 3, 3, 3],
])
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.imshow(M, cmap="Purples", vmin=0, vmax=3.5)
ax.set_xticks(range(len(skills)), skills, rotation=30, ha="right")
ax.set_yticks(range(len(projects)), projects)
for (i, j), v in np.ndenumerate(M):
    ax.text(j, i, "●" * v if v else "", ha="center", va="center", fontsize=8, color="#27272a")
ax.set_title("Which skills each portfolio project exercises (more dots = more depth)", fontsize=10)
ax.grid(False)
save(fig, "project-ladder")
