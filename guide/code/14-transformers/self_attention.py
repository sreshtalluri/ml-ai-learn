"""Scaled dot-product self-attention: the course example, a masked multi-token example, and why we divide by sqrt(d_k).

Run from guide/:  uv run code/14-transformers/self_attention.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, plt, save  # noqa: E402


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, causal=False):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                         # [n_q, n_k]
    if causal:
        scores = np.where(np.triu(np.ones_like(scores, dtype=bool), k=1), -np.inf, scores)
    weights = softmax(scores)                               # each row sums to 1
    return weights @ V, weights                             # [n_q, d_v], [n_q, n_k]


# ---------------------------------------------------------------------------
# 1. The course example: one query, two keys
# ---------------------------------------------------------------------------
q = np.array([[1.0, 0.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[10.0, 0.0], [0.0, 6.0]])
raw = q @ K.T
print("Course example")
print(f"  scores q·k     = {raw.ravel()}")
print(f"  scaled /sqrt(2)= {(raw / np.sqrt(2)).ravel().round(4)}")
out, w = attention(q, K, V)
print(f"  e^scores       = {np.exp(raw / np.sqrt(2)).ravel().round(4)}  sum={np.exp(raw / np.sqrt(2)).sum():.4f}")
print(f"  weights        = {w.ravel().round(4)}")
print(f"  output         = {out.ravel().round(4)}")

# ---------------------------------------------------------------------------
# 2. Self-attention over a toy sentence (SYNTHETIC embeddings and projections)
# ---------------------------------------------------------------------------
tokens = ["the", "cat", "sat", "on", "the", "mat"]
rng = np.random.default_rng(0)
d_model, d_k = 8, 4
vocab = {t: rng.normal(size=d_model) for t in dict.fromkeys(tokens)}
pos = np.array([[np.sin(p / 10 ** (2 * (i // 2) / d_model)) if i % 2 == 0 else np.cos(p / 10 ** (2 * (i // 2) / d_model))
                 for i in range(d_model)] for p in range(len(tokens))])
X = np.array([vocab[t] for t in tokens]) + pos                       # [6, 8]
W_q, W_k, W_v = (rng.normal(size=(d_model, d_k)) / np.sqrt(d_model) for _ in range(3))
Q, Kx, Vx = X @ W_q, X @ W_k, X @ W_v                                # each [6, 4]
out_full, w_full = attention(Q, Kx, Vx)
out_causal, w_causal = attention(Q, Kx, Vx, causal=True)
print(f"\nShapes: X {X.shape}  W_q {W_q.shape}  Q {Q.shape}  QK^T {(Q @ Kx.T).shape}  output {out_causal.shape}")
print(f"  row sums {w_causal.sum(axis=1).round(6)}")
print(f"  causal row 3 ('sat') weights: {w_causal[2].round(3)}  (zeros after position 3)")

# Multi-head bookkeeping: d_model = h * d_head
h, d_model_big = 8, 512
print(f"\nMulti-head: d_model={d_model_big}, heads={h}, d_head={d_model_big // h}; per-layer attention params = 4 * d_model^2 = {4 * d_model_big**2:,}")

# ---------------------------------------------------------------------------
# 3. Why scale? Dot products of random vectors grow like sqrt(d_k)
# ---------------------------------------------------------------------------
for d in (4, 64, 512):
    a, b = rng.normal(size=(2, 10000, d))
    dots = (a * b).sum(axis=1)
    print(f"  d_k={d:4d}: std of q·k = {dots.std():6.2f}  (sqrt(d_k) = {np.sqrt(d):.2f})")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
for ax, w_, title in [(axes[0], w_full, "Encoder style: every token sees every token"), (axes[1], w_causal, "Decoder style: causal mask")]:
    im = ax.imshow(w_, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6), tokens)
    ax.set_yticks(range(6), [f"{i + 1}:{t}" for i, t in enumerate(tokens)])
    ax.set(title=title, xlabel="key (attended to)")
    ax.title.set_fontsize(10)
    ax.grid(False)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{w_[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if w_[i, j] > 0.5 else "#27272a")
axes[0].set_ylabel("query (attending)")

ax = axes[2]
d = 512
a = rng.normal(size=(8, d))
qv = rng.normal(size=d)
raw_scores = a @ qv
ax.bar(np.arange(8) - 0.2, softmax(raw_scores), width=0.4, color=ORANGE, label="softmax(q·k), unscaled")
ax.bar(np.arange(8) + 0.2, softmax(raw_scores / np.sqrt(d)), width=0.4, color=BLUE, label="softmax(q·k / √d_k)")
ax.set(title=f"Why divide by √d_k (d_k = {d}, random vectors)", xlabel="key", ylabel="attention weight")
ax.title.set_fontsize(10)
ax.legend(frameon=False, fontsize=8)
save(fig, "self-attention")
