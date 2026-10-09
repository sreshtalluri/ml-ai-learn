"""Serving arithmetic: static vs continuous batching (a step-level simulator), paged KV-cache memory,
prefix caching, and GPU cost per million tokens.

Workloads, prices, and throughputs are SYNTHETIC and labelled as such. The simulator matches the website lab
(web/src/components/labs/BatchingLab.tsx): one time step = one decode iteration, every busy slot emits one token.
Run from guide/:  uv run code/18-llm-inference/batching_sim.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PALETTE, plt, save  # noqa: E402


def simulate(requests, slots, policy):
    """requests: list of (arrival_step, output_tokens). policy: 'static' or 'continuous'.
    Returns per-request finish times and per-slot segments (request, start, end)."""
    order = sorted(range(len(requests)), key=lambda i: (requests[i][0], i))
    queue, nxt = [], 0
    busy_until = [0] * slots
    finish = [None] * len(requests)
    segments = [[] for _ in range(slots)]
    t = 0
    while any(f is None for f in finish):
        while nxt < len(order) and requests[order[nxt]][0] <= t:
            queue.append(order[nxt])
            nxt += 1
        free = [s for s in range(slots) if busy_until[s] <= t]
        # static: a new batch starts only when every slot is free; continuous: any free slot is refilled
        if policy == "continuous" or len(free) == slots:
            for s in free:
                if not queue:
                    break
                i = queue.pop(0)
                busy_until[s] = finish[i] = t + requests[i][1]
                segments[s].append((i, t, finish[i]))
        t += 1
    return finish, segments


def metrics(requests, slots, finish):
    tokens = sum(n for _, n in requests)
    makespan = max(finish)
    lat = sorted(f - a for f, (a, _) in zip(finish, requests))
    p95 = lat[math.ceil(0.95 * len(lat)) - 1]
    return dict(throughput=tokens / makespan, mean=sum(lat) / len(lat), p95=p95,
                idle=1 - tokens / (slots * makespan), makespan=makespan, latencies=lat)


def workload(n, rate, mean_len, sigma, seed):
    """Synthetic: exponential inter-arrival gaps (rate = requests per step), log-normal output lengths."""
    r = np.random.default_rng(seed)
    arrivals = np.floor(np.cumsum(r.exponential(1 / rate, n))).astype(int)
    lens = np.maximum(1, np.round(mean_len * np.exp(sigma * r.standard_normal(n) - sigma ** 2 / 2))).astype(int)
    return list(zip(arrivals.tolist(), lens.tolist()))


# ---------------------------------------------------------------------------
# 1. Worked example: 2 slots, 4 requests at t=0 with output lengths 2, 8, 3, 3
# ---------------------------------------------------------------------------
REQ = [(0, 2), (0, 8), (0, 3), (0, 3)]
print("1. Worked example: K=2 slots, requests (arrival, tokens) =", REQ)
for pol in ("static", "continuous"):
    f, seg = simulate(REQ, 2, pol)
    m = metrics(REQ, 2, f)
    print(f"   {pol:10s} slots={seg}")
    print(f"   {'':10s} finish={f} latency={[x - a for x, (a, _) in zip(f, REQ)]} "
          f"throughput=16/{m['makespan']}={m['throughput']:.4f} tok/step mean={m['mean']:.2f} "
          f"p95={m['p95']} idle={m['idle']:.1%}")

# ---------------------------------------------------------------------------
# 2. Paged KV cache vs contiguous max-length reservation
# ---------------------------------------------------------------------------
per_tok, budget, max_len, actual, block = 131_072, 40e9, 4096, 600, 16
contig = int(budget // (max_len * per_tok))
blocks = math.ceil(actual / block)
paged = int(budget // (blocks * block * per_tok))
print("\n2. KV budget 40 GB, 131,072 B/token (8B-style GQA, BF16), requests use 600 tokens on average")
print(f"   contiguous, reserve 4096: 4096 x 131,072 = {max_len * per_tok:,} B/seq -> {contig} sequences, "
      f"utilisation 600/4096 = {actual / max_len:.1%}")
print(f"   paged, 16-token blocks: ceil(600/16) = {blocks} blocks = {blocks * block} tokens = {blocks * block * per_tok:,} B/seq "
      f"-> {paged} sequences, utilisation 600/{blocks * block} = {actual / (blocks * block):.1%}")

# ---------------------------------------------------------------------------
# 3. Prefix caching
# ---------------------------------------------------------------------------
sys_p, user = 2000, 200
print("\n3. Prefix cache: shared 2000-token system prompt + 200 user tokens")
print(f"   prefill tokens per request: {sys_p + user} -> {user} on a hit ({1 - user / (sys_p + user):.1%} less prefill)")

# ---------------------------------------------------------------------------
# 4. Cost per million tokens (synthetic price and throughput)
# ---------------------------------------------------------------------------
price, tps = 4.00, 1500
per_hour = tps * 3600
print(f"\n4. GPU at $4.00/hour sustaining 1,500 output tokens/s (synthetic)")
print(f"   1500 x 3600 = {per_hour:,} tokens/hour -> 4.00 / 5.4 = ${price / (per_hour / 1e6):.3f} per M tokens at 100% busy")
for u in (0.6, 0.3):
    print(f"   at {u:.0%} utilisation: 0.741 / {u} = ${price / (per_hour / 1e6) / u:.2f} per M tokens")
month = 730
print(f"   one GPU for a month: 4.00 x 730 = ${price * month:,.0f}; capacity {per_hour * month / 1e9:.3f}B tokens")
print(f"   break-even vs a synthetic API at $1.00 per M: {price * month / 1.0:,.0f}M tokens/month "
      f"= {price * month / (per_hour * month / 1e6):.0%} utilisation")

# ---------------------------------------------------------------------------
# 5. Random workloads: latency vs load
# ---------------------------------------------------------------------------
K, MEAN, SIGMA = 4, 24, 0.8
rates = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16])
curves = {"static": [], "continuous": []}
for rate in rates:
    for pol in curves:
        vals = [metrics(w, K, simulate(w, K, pol)[0]) for w in (workload(200, rate, MEAN, SIGMA, s) for s in range(5))]
        curves[pol].append((np.mean([v["p95"] for v in vals]), np.mean([v["throughput"] for v in vals])))
print(f"\n5. Synthetic load sweep, K={K}, mean length {MEAN}, sigma {SIGMA}, 200 requests x 5 seeds (p95 latency in steps)")
for i, rate in enumerate(rates):
    print(f"   rate {rate:.2f}: static p95 {curves['static'][i][0]:6.1f}, continuous p95 {curves['continuous'][i][0]:6.1f}")

# ---------------------------------------------------------------------------
# Figure: two Gantt charts and the load sweep
# ---------------------------------------------------------------------------
demo = workload(14, 0.12, MEAN, SIGMA, 3)
fig = plt.figure(figsize=(11, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1])
for row, pol in enumerate(("static", "continuous")):
    ax = fig.add_subplot(gs[row, 0])
    f, seg = simulate(demo, K, pol)
    m = metrics(demo, K, f)
    for s, segs in enumerate(seg):
        for i, a, b in segs:
            ax.barh(s, b - a, left=a, color=PALETTE[i % 4], alpha=0.85, edgecolor="white")
            if b - a >= 6:
                ax.text((a + b) / 2, s, f"r{i}", ha="center", va="center", color="white", fontsize=8)
    ax.set_yticks(range(K), [f"slot {s}" for s in range(K)])
    ax.set_xlim(0, 240)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{pol.capitalize()} batching: {m['throughput']:.2f} tokens/step, p95 latency {m['p95']} steps, "
                 f"{m['idle']:.0%} idle", fontsize=9.5)
ax.set_xlabel("time step (one decode iteration)")
ax3 = fig.add_subplot(gs[:, 1])
ax3.plot(rates, [c[0] for c in curves["static"]], "o-", color=ORANGE, label="static")
ax3.plot(rates, [c[0] for c in curves["continuous"]], "o-", color=BLUE, label="continuous")
ax3.axvline(K / MEAN, color=GRAY, ls=":", label=f"capacity = K / mean length = {K / MEAN:.3f}")
ax3.set(xlabel="arrival rate (requests per step)", ylabel="p95 latency (steps)", title="Latency vs load (synthetic)")
ax3.legend(fontsize=8)
fig.tight_layout()
save(fig, "serving-llms")
