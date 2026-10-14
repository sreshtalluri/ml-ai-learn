# Adapted from AI Engineering from Scratch (MIT), phases/19-capstone-projects/78-zero-parameter-sharding
"""ZeRO from scratch: sharded Adam across simulated ranks, checked against plain data parallelism,
plus the per-GPU memory and communication arithmetic for DDP and ZeRO stages 1, 2, and 3.

Ranks are simulated in one process (a Python list per rank), so it runs anywhere. The regression data
and the model shapes are SYNTHETIC. Memory numbers are estimates of training state, not measurements.
Run from guide/:  uv run code/27-training-at-scale/zero_sharding.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

GB = 1e9  # decimal gigabytes, like GPU spec sheets


# ---------- 1. Sharded Adam on simulated ranks ----------

def adam(p, g, m, v, t, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam step on any slice of parameters. Elementwise, so it works on a shard."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    p = p - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return p, m, v


def local_grad(w, X, y):
    """Mean squared error gradient for a linear model on this rank's mini-batch."""
    return 2 * X.T @ (X @ w - y) / len(y)


def reduce_scatter_mean(grads, n):
    """Rank r receives the average over ranks of chunk r only."""
    chunks = [np.array_split(g, n) for g in grads]
    return [np.mean([chunks[src][r] for src in range(n)], axis=0) for r in range(n)]


def all_gather(shards):
    """Every rank receives every shard, concatenated in rank order."""
    full = np.concatenate(shards)
    return [full.copy() for _ in shards]


def train(n_ranks, steps, zero, seed=0):
    rng = np.random.default_rng(seed)
    P = 10                                                      # parameters (tiny on purpose)
    w_true = rng.normal(size=P)
    data = []
    for _ in range(n_ranks):                                    # each rank gets its own SYNTHETIC batch
        X = rng.normal(size=(16, P))
        data.append((X, X @ w_true + 0.1 * rng.normal(size=16)))
    w = [np.zeros(P) for _ in range(n_ranks)]                   # replicated weights
    if zero:   # ZeRO-1: rank r keeps Adam state for its slice only
        m = [np.zeros(len(s)) for s in np.array_split(np.zeros(P), n_ranks)]
        v = [x.copy() for x in m]
    else:      # DDP: every rank keeps Adam state for all P parameters
        m = [np.zeros(P) for _ in range(n_ranks)]
        v = [np.zeros(P) for _ in range(n_ranks)]
    for t in range(1, steps + 1):
        grads = [local_grad(w[r], *data[r]) for r in range(n_ranks)]
        if zero:
            g_shard = reduce_scatter_mean(grads, n_ranks)       # comm half 1
            new = []
            for r in range(n_ranks):
                w_shard = np.array_split(w[r], n_ranks)[r]
                p, m[r], v[r] = adam(w_shard, g_shard[r], m[r], v[r], t)
                new.append(p)
            w = all_gather(new)                                 # comm half 2
        else:
            g = np.mean(grads, axis=0)                          # all-reduce (average)
            for r in range(n_ranks):
                w[r], m[r], v[r] = adam(w[r], g, m[r], v[r], t)
    state_floats = sum(len(x) for x in m) + sum(len(x) for x in v)
    return w, state_floats


# ---------- 2. Memory and communication arithmetic ----------

# SYNTHETIC Llama-like shapes, identical to the LoRA lab and the fine-tuning lesson.
PRESETS = {
    "7B-style": dict(d=4096, layers=32, d_ff=11008, kv=4096, vocab=32000),
    "70B-style": dict(d=8192, layers=80, d_ff=28672, kv=1024, vocab=32000),
}


def n_params(c):
    d, f, kv = c["d"], c["d_ff"], c["kv"]
    per_layer = d * d + d * kv + d * kv + d * d + 3 * d * f   # q, k, v, o, gate, up, down
    return per_layer * c["layers"] + 2 * c["vocab"] * d      # + input embedding and untied output head


# bytes per parameter: (weights, gradients, optimizer states)
PRECISION = {
    "bf16 mixed + AdamW": (2, 2, 12),     # optimizer = fp32 master 4 + m 4 + v 4
    "fp32 + AdamW": (4, 4, 8),            # weights are already fp32, so no extra master copy
    "bf16 + 8-bit Adam": (2, 2, 6),       # fp32 master 4 + m 1 + v 1
}
STRATEGIES = ["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]


def activation_bytes(c, tokens=2048, act_bytes=2):
    """Gradient checkpointing on: each layer's input (2 bytes per value) plus one layer's full
    activations (~34 bytes per token per hidden unit in 16-bit). Same rule as the fine-tuning lesson."""
    scale = act_bytes / 2
    return scale * (c["layers"] * 2 * tokens * c["d"] + 34 * tokens * c["d"])


def per_gpu(c, strategy, n, precision="bf16 mixed + AdamW", tokens=2048):
    P = n_params(c)
    bw, bg, bo = PRECISION[precision]
    k = STRATEGIES.index(strategy)          # 0 = DDP, then how many of (optimizer, grads, weights) are sharded
    return dict(
        weights=bw * P / (n if k >= 3 else 1),
        grads=bg * P / (n if k >= 2 else 1),
        optimizer=bo * P / (n if k >= 1 else 1),
        activations=activation_bytes(c, tokens, 4 if precision.startswith("fp32") else 2),
    )


def comm_bytes(c, strategy, n, precision="bf16 mixed + AdamW"):
    """Bytes each GPU sends per step with ring collectives. A ring all-reduce sends 2(N-1)/N of the
    tensor; reduce-scatter and all-gather each send (N-1)/N."""
    if n == 1:
        return 0.0
    P = n_params(c)
    bw, bg, _ = PRECISION[precision]
    f = (n - 1) / n
    if strategy == "DDP":
        return 2 * f * bg * P                      # all-reduce gradients
    if strategy in ("ZeRO-1", "ZeRO-2"):
        return f * bg * P + f * bw * P             # reduce-scatter grads + all-gather updated weights
    return 2 * f * bw * P + f * bg * P             # all-gather weights in forward and backward + reduce-scatter grads


def table(c, n, precision="bf16 mixed + AdamW"):
    ddp = comm_bytes(c, "DDP", n, precision)
    print(f"  {'strategy':8s} {'weights':>8s} {'grads':>8s} {'optim':>8s} {'activ':>7s} {'total GB':>9s} {'comm GB':>8s} {'vs DDP':>6s}")
    for s in STRATEGIES:
        m = per_gpu(c, s, n, precision)
        cb = comm_bytes(c, s, n, precision)
        print(f"  {s:8s} {m['weights'] / GB:8.2f} {m['grads'] / GB:8.2f} {m['optimizer'] / GB:8.2f} {m['activations'] / GB:7.3f}"
              f" {sum(m.values()) / GB:9.2f} {cb / GB:8.2f} {cb / ddp if ddp else 0:6.2f}x")


if __name__ == "__main__":
    print("1) Sharded Adam versus plain data parallelism (4 simulated ranks, SYNTHETIC regression)")
    w_ddp, s_ddp = train(4, 50, zero=False)
    w_zero, s_zero = train(4, 50, zero=True)
    print(f"  max |w_DDP - w_ZeRO1| after 50 steps = {np.max(np.abs(w_ddp[0] - w_zero[0])):.2e}")
    print(f"  all ZeRO ranks hold identical weights: {all(np.array_equal(w_zero[0], w) for w in w_zero)}")
    print(f"  Adam state floats across all ranks: DDP {s_ddp}, ZeRO-1 {s_zero}  (P = 10, N = 4)")

    c7 = PRESETS["7B-style"]
    P7 = n_params(c7)
    print(f"\n2) 7B-style model: P = {P7:,} parameters; activations at 2,048 tokens with checkpointing "
          f"= {activation_bytes(c7) / GB:.3f} GB per GPU")
    for n in (1, 8, 64):
        print(f" N = {n} GPUs, bf16 mixed precision + AdamW (16 bytes/param)")
        table(c7, n)
    c70 = PRESETS["70B-style"]
    print(f"\n3) 70B-style model: P = {n_params(c70):,} parameters, N = 64 GPUs")
    table(c70, 64)
    for n in (8, 16, 32, 64, 128):
        tot = sum(per_gpu(c70, "ZeRO-3", n).values()) / GB
        print(f"  ZeRO-3 on {n:3d} GPUs: {tot:6.2f} GB per GPU  {'fits' if tot <= 80 else 'does not fit'} in 80 GB")

    # ---------- figure ----------
    colors = {"weights": BLUE, "grads": ORANGE, "optimizer": PURPLE, "activations": TEAL}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
    for yi, s in enumerate(reversed(STRATEGIES)):
        left = 0.0
        for k, v in per_gpu(c7, s, 8).items():
            a1.barh(yi, v / GB, left=left, color=colors[k], label=k if yi == 0 else None)
            left += v / GB
        a1.text(left + 1.5, yi, f"{left:.1f} GB", va="center", fontsize=9)
    a1.axvline(80, color="#dc2626", ls="--", lw=1.5)
    a1.text(81, -0.45, "80 GB GPU", color="#dc2626", ha="left", fontsize=9)
    a1.set_yticks(range(4), list(reversed(STRATEGIES)))
    a1.set_xlim(0, 130)
    a1.set_xlabel("memory per GPU (GB, estimate)")
    a1.set_title("7B-style model on 8 GPUs, bf16 + AdamW")
    a1.legend(loc="lower right", fontsize=8, frameon=False)
    ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    for s, col in zip(STRATEGIES, [GRAY, PURPLE, ORANGE, BLUE]):
        a2.plot(ns, [sum(per_gpu(c7, s, int(n)).values()) / GB for n in ns], "o-", color=col, label=s, ms=4)
    a2.axhline(80, color="#dc2626", ls="--", lw=1.2)
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xticks(ns, [str(n) for n in ns])
    a2.set_xlabel("number of GPUs N")
    a2.set_ylabel("GB per GPU (log scale)")
    a2.set_title("Only ZeRO-3 keeps shrinking")
    a2.legend(fontsize=8, frameon=False)
    save(fig, "zero-parameter-sharding")
