# Adapted from AI Engineering from Scratch (MIT), phases/07-transformers-deep-dive/15-attention-variants
"""Attention masks: full, causal, sliding window, local + strided, global tokens.
Counts attended cells, grows the receptive field across layers, and sizes the KV cache.

Run from guide/:  uv run code/14-transformers/attention_variants.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Masks as boolean matrices: allowed[i, j] is True when query i may read key j
# ---------------------------------------------------------------------------
def causal(n):
    i, j = np.indices((n, n))
    return j <= i                                   # [n, n], lower triangle


def sliding_window(n, w):
    i, j = np.indices((n, n))
    return (j <= i) & (i - j < w)                   # the last w tokens, self included


def local_strided(n, w, stride):
    i, j = np.indices((n, n))
    return sliding_window(n, w) | ((j <= i) & (j % stride == stride - 1))


def with_global(mask, global_ids):
    m = mask.copy()
    m[:, global_ids] = True                         # everyone reads the global tokens
    m[global_ids, :] = True                         # global tokens read everyone
    return m


def render(mask, label):
    print(f"{label}: {int(mask.sum())} of {mask.size} cells")
    for i, row in enumerate(mask):
        print(f"  {i:>2} |" + "".join("x" if v else "." for v in row))


n, w = 16, 4
masks = {
    "full (bidirectional)": np.ones((n, n), dtype=bool),
    "causal": causal(n),
    f"sliding window w={w}": sliding_window(n, w),
    "local w=2 + stride 4": local_strided(n, 2, 4),
    "local radius 2 + global token 0 (bidirectional)": with_global(
        np.abs(np.subtract.outer(np.arange(n), np.arange(n))) <= 2, [0]),
}
print("=== attended cells on a 16-token sequence ===")
for label, m in masks.items():
    print(f"  {label:<46} {int(m.sum()):>4} of {n * n}")
print()
render(sliding_window(8, 4), "sliding window, n=8, w=4")
assert sliding_window(n, n).sum() == causal(n).sum()   # w >= n recovers full causal


# ---------------------------------------------------------------------------
# 2. Receptive field: information hops one window per layer
# ---------------------------------------------------------------------------
def reach_after(mask, layers):
    """Which keys can influence each query after stacking `layers` copies of the mask."""
    R = np.eye(len(mask), dtype=bool)
    for _ in range(layers):
        R = (mask.astype(int) @ R.astype(int)) > 0  # one more attention hop
    return R


print("\n=== receptive field of the last query (position 15), sliding window w=4 ===")
swa = sliding_window(n, w)
for L in range(1, 6):
    seen = int(reach_after(swa, L)[n - 1].sum())
    formula = min(n, L * (w - 1) + 1)
    print(f"  {L} layer(s): {seen:>2} tokens   (formula L*(w-1)+1 = {L * (w - 1) + 1}, capped at {n} -> {formula})")
    assert seen == formula


# ---------------------------------------------------------------------------
# 3. KV cache: a sliding-window layer keeps only the last w tokens
# ---------------------------------------------------------------------------
def kv_bytes(layers, n_kv, d_head, tokens, bytes_per=2):
    return 2 * layers * n_kv * d_head * tokens * bytes_per   # K and V


print("\n=== KV cache for one 131,072-token sequence (synthetic 80-layer, 8 KV heads, d_head 128, BF16) ===")
N, layers, n_kv, d_head = 131_072, 80, 8, 128
full = kv_bytes(layers, n_kv, d_head, N)
print(f"  every layer full          : {full / 1e9:6.2f} GB")
for W in (4096, 1024):
    print(f"  every layer window {W:>4}   : {kv_bytes(layers, n_kv, d_head, W) / 1e9:6.2f} GB")
n_global = layers // 6                               # 5 local : 1 global
mix = kv_bytes(n_global, n_kv, d_head, N) + kv_bytes(layers - n_global, n_kv, d_head, 1024)
print(f"  5:1 local:global, W=1024  : {mix / 1e9:6.2f} GB   ({n_global} global + {layers - n_global} local layers, {full / mix:.1f}x smaller)")


# ---------------------------------------------------------------------------
# 4. Figure: four masks for one query, plus receptive-field growth
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
q = 12
panels = [("Causal", causal(n), BLUE), (f"Sliding window (w={w})", swa, TEAL),
          ("Local w=2 + stride 4", local_strided(n, 2, 4), PURPLE),
          ("Local radius 2 + global token 0", masks["local radius 2 + global token 0 (bidirectional)"], ORANGE)]
for ax, (title, m, color) in zip(axes, panels):
    img = np.ones((n, n, 3)) * 0.96
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    img[m] = rgb * 0.35 + 0.65                      # light tint for allowed cells
    img[q, m[q]] = rgb                              # query 12's row, full colour
    ax.imshow(img, interpolation="nearest")
    ax.set_title(f"{title}\n{int(m.sum())} cells, query {q} sees {int(m[q].sum())}", fontsize=9)
    ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 5, 10, 15]); ax.grid(False)
    ax.set_xlabel("key position"); ax.set_ylabel("query position")
Ls = np.arange(1, 9)
for W, color in ((2, PURPLE), (4, TEAL), (8, BLUE)):
    axes[4].plot(Ls, [min(64, L * (W - 1) + 1) for L in Ls], "o-", color=color, label=f"w={W}")
axes[4].set_title("Receptive field of the last token\n(64-token sequence, sliding window)", fontsize=9)
axes[4].set_xlabel("layers stacked"); axes[4].set_ylabel("tokens that can influence it"); axes[4].legend()
fig.tight_layout()
save(fig, "attention-variants")
