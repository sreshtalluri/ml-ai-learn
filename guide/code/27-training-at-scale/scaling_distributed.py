# Adapted from AI Engineering from Scratch (MIT), phases/10-llms-from-scratch/05-scaling-distributed
"""Distributed training from scratch: data parallelism (gradient averaging), tensor parallelism
(column- and row-split matmuls), pipeline parallelism (GPipe bubble), and a per-GPU memory calculator.

Data and weights are SYNTHETIC (fixed seed). Model shapes are SYNTHETIC Llama-like configs.
Memory numbers are estimates of training state, not measurements.
Run from guide/:  uv run code/27-training-at-scale/scaling_distributed.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, plt, save  # noqa: E402

GB = 1e9


# ---------- 1. Data parallelism: average of shard gradients == full-batch gradient ----------

def mse_grad(w, X, y):
    return 2 * X.T @ (X @ w - y) / len(y)


def data_parallel_grad(w, X, y, n_gpus):
    """Each GPU gets an equal shard, computes its local gradient; all-reduce averages them."""
    shards = zip(np.array_split(X, n_gpus), np.array_split(y, n_gpus))
    return np.mean([mse_grad(w, Xs, ys) for Xs, ys in shards], axis=0)


# ---------- 2. Tensor parallelism: split a weight matrix across GPUs ----------

def column_parallel(x, W, n):
    """Each GPU holds a slice of W's output columns and computes a slice of y. Combine: concatenate."""
    return np.concatenate([x @ Ws for Ws in np.array_split(W, n, axis=1)], axis=-1)


def row_parallel(x, W, n):
    """Each GPU holds a slice of W's input rows and the matching slice of x. Combine: sum (an all-reduce)."""
    xs, Ws = np.array_split(x, n, axis=-1), np.array_split(W, n, axis=0)
    return sum(a @ b for a, b in zip(xs, Ws))


def tp_mlp(x, W1, W2, n):
    """Megatron-style MLP: column-split W1, row-split W2, so the hidden activation never leaves its GPU.
    One all-reduce at the end instead of one per matmul."""
    h_parts = [np.maximum(x @ W1s, 0) for W1s in np.array_split(W1, n, axis=1)]   # local, no comm
    return sum(h @ W2s for h, W2s in zip(h_parts, np.array_split(W2, n, axis=0)))  # all-reduce


# ---------- 3. Pipeline parallelism: GPipe schedule ----------

def gpipe_time(p, m):
    """p stages, m micro-batches, forward = 1 tick, backward = 2 ticks per stage per micro-batch.
    All forwards flow down the pipe, then all backwards flow back up."""
    return (m + p - 1) * 1 + (m + p - 1) * 2


def bubble_fraction(p, m):
    busy = 3 * m          # each stage does m forwards and m backwards
    return 1 - busy / gpipe_time(p, m)           # = (p - 1) / (m + p - 1)


# ---------- 4. Memory per GPU ----------

def n_params(d, layers, d_ff, kv, vocab):
    return layers * (2 * d * d + 2 * d * kv + 3 * d * d_ff) + 2 * vocab * d


SEVEN_B = dict(d=4096, layers=32, d_ff=11008, kv=4096, vocab=32000)     # SYNTHETIC, as in the fine-tuning lesson
SEVENTY_B = dict(d=8192, layers=80, d_ff=28672, kv=1024, vocab=32000)   # SYNTHETIC, grouped-query attention


def train_memory(P, tp=1, pp=1, bytes_per_param=16):
    """Training state per GPU when weights are split tp x pp ways and data parallelism replicates the rest.
    16 bytes/param = bf16 weights 2 + bf16 grads 2 + fp32 master 4 + Adam m 4 + Adam v 4."""
    return bytes_per_param * P / (tp * pp)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X, w_true = rng.normal(size=(64, 5)), rng.normal(size=5)
    y = X @ w_true + 0.1 * rng.normal(size=64)                   # SYNTHETIC regression
    w = np.zeros(5)
    full = mse_grad(w, X, y)
    print("1) Data parallelism, 64 examples")
    for n in (1, 2, 4, 8):
        print(f"  {n} GPUs: max |avg shard grad - full grad| = {np.max(np.abs(data_parallel_grad(w, X, y, n) - full)):.1e}")
    print(f"  unequal shards (60 examples on 8 GPUs): diff = "
          f"{np.max(np.abs(data_parallel_grad(w, X[:60], y[:60], 8) - mse_grad(w, X[:60], y[:60]))):.1e}")

    x = rng.normal(size=(4, 512))
    W1, W2 = rng.normal(size=(512, 2048)) / 20, rng.normal(size=(2048, 512)) / 40
    print("\n2) Tensor parallelism, x [4, 512], W1 [512, 2048], W2 [2048, 512]")
    for n in (2, 4, 8):
        e1 = np.max(np.abs(column_parallel(x, W1, n) - x @ W1))
        e2 = np.max(np.abs(row_parallel(x @ W1, W2, n) - (x @ W1) @ W2))
        e3 = np.max(np.abs(tp_mlp(x, W1, W2, n) - np.maximum(x @ W1, 0) @ W2))
        print(f"  {n} GPUs: column {e1:.1e}, row {e2:.1e}, column+row MLP {e3:.1e}")

    print("\n3) GPipe pipeline, forward 1 tick, backward 2 ticks")
    for p, m in ((4, 1), (4, 4), (4, 16), (4, 32), (8, 32)):
        print(f"  p = {p}, m = {m:2d}: step = {gpipe_time(p, m):3d} ticks, bubble = {bubble_fraction(p, m):.1%}"
              f"  [(p-1)/(m+p-1) = {(p - 1)}/{m + p - 1}]")

    print("\n4) Memory per GPU, 16 bytes per parameter, before activations")
    for name, cfg in (("7B-style", SEVEN_B), ("70B-style", SEVENTY_B)):
        P = n_params(**cfg)
        print(f"  {name}: P = {P:,}; full state = {train_memory(P) / GB:.2f} GB")
        for tp, pp in ((1, 1), (8, 1), (8, 2), (8, 4)):
            mem = train_memory(P, tp, pp) / GB
            print(f"    TP {tp} x PP {pp}: {mem:8.2f} GB per GPU  {'fits' if mem <= 80 else 'too big'} for 80 GB")
    print(f"  3D parallelism: TP 8 x PP 16 x DP 128 = {8 * 16 * 128:,} GPUs")

    # ---------- figure ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1.3, 1]})
    p, m = 4, 4
    for s in range(p):
        for mb in range(m):
            a1.barh(s, 1, left=s + mb, color=BLUE, edgecolor="white")
            a1.text(s + mb + 0.5, s, f"F{mb + 1}", ha="center", va="center", color="white", fontsize=8)
            b0 = (m + p - 1) + (p - 1 - s) * 2 + mb * 2
            a1.barh(s, 2, left=b0, color=ORANGE, edgecolor="white")
            a1.text(b0 + 1, s, f"B{mb + 1}", ha="center", va="center", color="white", fontsize=8)
    a1.set_yticks(range(p), [f"stage {s + 1}" for s in range(p)])
    a1.invert_yaxis()
    a1.set_xlim(0, gpipe_time(p, m))
    a1.set_xlabel("time (ticks)")
    a1.set_title(f"GPipe, 4 stages, 4 micro-batches: {bubble_fraction(p, m):.0%} idle (gaps)")
    ms = np.array([1, 2, 4, 8, 16, 32, 64])
    for pp, col in ((2, GRAY), (4, BLUE), (8, PURPLE), (16, ORANGE)):
        a2.plot(ms, [bubble_fraction(pp, int(mm)) for mm in ms], "o-", color=col, ms=3, label=f"p = {pp}")
    a2.set_xscale("log", base=2)
    a2.set_xticks(ms, [str(v) for v in ms])
    a2.set_xlabel("micro-batches m")
    a2.set_ylabel("bubble fraction")
    a2.set_title("More micro-batches, smaller bubble")
    a2.legend(frameon=False, fontsize=9)
    save(fig, "scaling-distributed")
