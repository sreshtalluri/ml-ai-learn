# Adapted from AI Engineering from Scratch (MIT), phases/10-llms-from-scratch/34-gradient-checkpointing
"""Gradient checkpointing from scratch: a stack of NumPy MLP layers trained with full activation storage
versus checkpointing every k layers. Checks the gradients are identical, counts stored activations and
recomputed layer forwards, and prints the memory arithmetic for a SYNTHETIC 7B-style transformer.

Weights and inputs are SYNTHETIC (fixed seed). Memory numbers are estimates, not measurements.
Run from guide/:  uv run code/27-training-at-scale/gradient_checkpointing.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PURPLE, plt, save  # noqa: E402

GB = 1e9


# ---------- 1. A layer, its forward, and its backward ----------

def layer_forward(x, w1, w2):
    """y = x + relu(x W1) W2   (a residual MLP block). Returns y and what backward needs."""
    h = x @ w1
    a = np.maximum(h, 0)
    return x + a @ w2, (x, h, a)


def layer_backward(gy, cache, w1, w2):
    x, h, a = cache
    gw2 = a.T @ gy
    gh = (gy @ w2.T) * (h > 0)
    gw1 = x.T @ gh
    gx = gy + gh @ w1.T          # residual path + MLP path
    return gx, (gw1, gw2)


# ---------- 2. Plain training step: keep every layer's cache ----------

def backward_full(x, params, gout):
    caches, stored = [], 0
    for w1, w2 in params:
        x, cache = layer_forward(x, w1, w2)
        caches.append(cache)
        stored += sum(t.size for t in cache)       # every intermediate stays alive
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        gout, grads[i] = layer_backward(gout, caches[i], *params[i])
    return grads, stored, 0


# ---------- 3. Checkpointed step: keep only every k-th layer input, recompute the rest ----------

def backward_checkpointed(x, params, gout, k):
    L = len(params)
    boundaries = {}                                   # segment start index -> its input
    for i, (w1, w2) in enumerate(params):
        if i % k == 0:
            boundaries[i] = x                          # the only tensors kept from the forward pass
        x, _ = layer_forward(x, w1, w2)                # cache thrown away
    grads, recomputed, peak = [None] * L, 0, 0
    for start in reversed(range(0, L, k)):
        end = min(start + k, L)
        h, caches = boundaries[start], []
        for i in range(start, end):                    # re-run this segment's forward
            h, cache = layer_forward(h, *params[i])
            caches.append(cache)
            recomputed += 1
        live = sum(b.size for b in boundaries.values()) + sum(t.size for c in caches for t in c)
        peak = max(peak, live)
        for j in reversed(range(start, end)):
            gout, grads[j] = layer_backward(gout, caches[j - start], *params[j])
    return grads, peak, recomputed


# ---------- 4. Transformer-scale arithmetic (same rule as the fine-tuning lesson) ----------

def activation_bytes(layers, tokens, d, k=None):
    """~34 bytes per token per hidden unit per layer in 16-bit with fused attention.
    Checkpointing every k layers keeps L/k boundary inputs (2 bytes each value) plus k layers' full activations
    during the backward of one segment. k = 1 is the usual per-layer checkpointing."""
    full = 34 * tokens * d
    if k is None:
        return layers * full
    return math.ceil(layers / k) * 2 * tokens * d + k * full


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    L, d, ff, batch = 8, 16, 64, 4                          # SYNTHETIC toy network
    params = [(rng.normal(0, 0.2, (d, ff)), rng.normal(0, 0.2, (ff, d))) for _ in range(L)]
    x = rng.normal(size=(batch, d))
    gout = rng.normal(size=(batch, d))

    g_full, stored_full, _ = backward_full(x, params, gout)
    print(f"1) Toy network: {L} residual MLP layers, d = {d}, ff = {ff}, batch = {batch}")
    print(f"  full storage: {stored_full:,} floats kept, 0 layers recomputed")
    for k in (1, 2, 3, 4, 8):
        g_ck, peak, rec = backward_checkpointed(x, params, gout, k)
        diff = max(np.max(np.abs(a - b)) for ga, gb in zip(g_full, g_ck) for a, b in zip(ga, gb))
        print(f"  k = {k}: peak {peak:,} floats, {rec} layer forwards recomputed, max grad diff {diff:.1e}")

    print("\n2) Compute: forward = 1 unit, backward = 2 units per layer")
    print(f"  no checkpointing: 1 + 2 = 3 units; checkpointing: 1 + 1 + 2 = 4 units -> +{(4 / 3 - 1):.1%} per step")

    Lt, T, D = 32, 2048, 4096                                # SYNTHETIC 7B-style model, 2,048 tokens
    print(f"\n3) 7B-style model: {Lt} layers, d = {D}, {T:,} tokens per micro-batch, 16-bit activations")
    none = activation_bytes(Lt, T, D)
    print(f"  no checkpointing: {Lt} x 34 x {T} x {D} = {none / GB:.3f} GB")
    for k in (1, 2, 4, 6, 8, 16, 32):
        print(f"  checkpoint every {k:2d} layers: {activation_bytes(Lt, T, D, k) / GB:6.3f} GB")
    print(f"  sqrt(L) = {math.sqrt(Lt):.2f}; but per-layer checkpointing (k = 1) wins here because a stored "
          f"input (2 bytes) is 17x smaller than a layer's full activations (34 bytes)")
    print(f"  at 8,192 tokens: none {activation_bytes(Lt, 8192, D) / GB:.2f} GB, k = 1 {activation_bytes(Lt, 8192, D, 1) / GB:.2f} GB")

    # ---------- figure ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7))
    ks = list(range(1, 33))
    a1.plot(ks, [activation_bytes(Lt, T, D, k) / GB for k in ks], "o-", color=BLUE, ms=3, label="checkpoint every k layers")
    a1.axhline(none / GB, color=ORANGE, ls="--", label=f"no checkpointing ({none / GB:.2f} GB)")
    a1.set_xlabel("segment size k (layers)")
    a1.set_ylabel("activation memory (GB, estimate)")
    a1.set_title("7B-style model, 2,048 tokens")
    a1.legend(frameon=False, fontsize=9)
    tokens = np.array([1024, 2048, 4096, 8192, 16384, 32768])
    a2.plot(tokens, [activation_bytes(Lt, int(t), D) / GB for t in tokens], "o-", color=ORANGE, label="no checkpointing")
    a2.plot(tokens, [activation_bytes(Lt, int(t), D, 1) / GB for t in tokens], "o-", color=PURPLE, label="checkpoint every layer")
    a2.axhline(80, color="#dc2626", ls=":", lw=1.2)
    a2.text(1100, 83, "80 GB GPU", color="#dc2626", fontsize=9)
    a2.set_xticks([1024, 8192, 16384, 32768], ["1K", "8K", "16K", "32K"])
    a2.set_xlabel("tokens per micro-batch")
    a2.set_ylabel("GB")
    a2.set_title("Activations grow linearly with tokens")
    a2.legend(frameon=False, fontsize=9)
    save(fig, "gradient-checkpointing")
