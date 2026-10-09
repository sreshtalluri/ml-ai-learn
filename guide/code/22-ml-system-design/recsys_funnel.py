"""A recommendation funnel cost model: latency, compute, and recall per stage, plus the negative-downsampling
calibration fix. All numbers are SYNTHETIC and chosen for round arithmetic; they are not measurements of any real system.

Run from guide/:  uv run code/22-ml-system-design/recsys_funnel.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

# Stage model (same constants as the web lab, web/src/components/labs/RecsysFunnelLab.tsx):
#   latency_s = fixed_s + items_scored_s * per_item_ms_s     (one worker, no sharding)
#   compute_s = items_scored_s * per_item_ms_s               (busy worker-ms per request)
#   recall_s  = 1 - exp(-items_kept_s / tau_s)               (synthetic saturating curve)
OVERHEAD_MS = 20.0  # network, request parsing, logging
STAGES = [
    # name,                 fixed ms, per-item ms, tau
    ("Candidate generation", 12.0, 0.005, 250.0),
    ("Ranking", 15.0, 0.06, 50.0),
    ("Re-ranking", 3.0, 0.10, 10.0),
]


def funnel(k1, k2, k3, per_item=None, budget_ms=150.0):
    """Items scored by each stage: candidate gen returns k1 (cost per returned candidate), ranking scores k1 and
    keeps k2, re-ranking scores k2 and keeps k3."""
    per_item = per_item or [s[2] for s in STAGES]
    scored, kept = [k1, k1, k2], [k1, k2, k3]
    rows = []
    for (name, fixed, _, tau), n, keep, t in zip(STAGES, scored, kept, per_item):
        rows.append(dict(name=name, scored=n, kept=keep, latency=fixed + n * t, compute=n * t,
                         recall=1 - math.exp(-keep / tau)))
    total = OVERHEAD_MS + sum(r["latency"] for r in rows)
    recall = math.prod(r["recall"] for r in rows)
    return rows, total, sum(r["compute"] for r in rows), recall, budget_ms - total


def calibrate(q, w):
    """Undo negative downsampling at rate w: p = q / (q + (1 - q) / w)."""
    return q / (q + (1 - q) / w)


if __name__ == "__main__":
    print("Worked example: 10M-item catalog, keep 1,000 -> 100 -> 20, budget 150 ms (synthetic)\n")
    rows, total, compute, recall, headroom = funnel(1000, 100, 20)
    for r in rows:
        name = r["name"]
        fixed, t, tau = next((s[1], s[2], s[3]) for s in STAGES if s[0] == name)
        print(f"{name:21s} scores {r['scored']:>5}, keeps {r['kept']:>5}: "
              f"{fixed:g} + {r['scored']} x {t:g} = {r['latency']:.1f} ms; "
              f"recall 1 - exp(-{r['kept']}/{tau:g}) = {r['recall']:.4f}")
    print(f"overhead {OVERHEAD_MS:g} ms -> total {total:.1f} ms, headroom {headroom:.1f} ms")
    print(f"compute per request {compute:.1f} worker-ms; at 10,000 QPS that is "
          f"{compute * 10_000 / 1000:.0f} worker-seconds per second (~{compute * 10:.0f} busy workers)")
    print(f"end-to-end synthetic recall {recall:.4f}")

    brute = 10_000_000 * STAGES[1][2]
    print(f"\nRanking the whole catalog: 10,000,000 x 0.06 ms = {brute:,.0f} ms = {brute / 1000:,.0f} s per request")

    print("\nCalibration after keeping 10% of negatives (w = 0.1):")
    for q in (0.5, 0.2):
        print(f"  model output q = {q}: p = {q} / ({q} + {1 - q:g}/0.1) = {calibrate(q, 0.1):.4f}")

    # ---- figure ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    left, colors = 0.0, [GRAY, BLUE, TEAL, PURPLE]
    parts = [("Overhead", OVERHEAD_MS)] + [(r["name"], r["latency"]) for r in rows]
    for i, ((name, ms), c) in enumerate(zip(parts, colors)):
        a1.barh(0, ms, height=0.4, left=left, color=c, edgecolor="white")
        y = 0.3 if i % 2 == 0 else -0.3  # alternate above/below so narrow segments stay readable
        a1.text(left + ms / 2, y, f"{name}\n{ms:.0f} ms", ha="center", va="center", fontsize=8.5, color=c)
        left += ms
    a1.axvline(150, color=ORANGE, lw=2, ls="--")
    a1.text(150, 0.45, " budget 150 ms", color=ORANGE, va="center")
    a1.set(xlim=(0, 175), ylim=(-0.6, 0.6), yticks=[], xlabel="latency per request (ms)",
           title=f"Latency budget: {total:.0f} ms used, {headroom:.0f} ms headroom")
    a1.grid(axis="y", visible=False)

    k1s = np.arange(100, 5001, 50)
    lat = [funnel(int(k), 100, 20)[1] for k in k1s]
    rec = [funnel(int(k), 100, 20)[3] for k in k1s]
    a2.plot(k1s, lat, color=BLUE, label="total latency (ms)")
    a2.axhline(150, color=ORANGE, ls="--", lw=1.5, label="budget")
    a2.set(xlabel="candidates passed to the ranker (k1)", ylabel="latency (ms)",
           title="More candidates: recall saturates, latency does not")
    a3 = a2.twinx()
    a3.plot(k1s, rec, color=TEAL, label="end-to-end recall")
    a3.set(ylim=(0, 1), ylabel="synthetic recall")
    a3.grid(False)
    a3.spines["right"].set_visible(True)
    a2.axvline(1000, color=GRAY, lw=1)
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a3.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    save(fig, "ml-system-design")
