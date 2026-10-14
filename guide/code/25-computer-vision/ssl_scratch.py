# Adapted from AI Engineering from Scratch (MIT), phases/04-computer-vision/17-self-supervised-vision
"""Self-supervised vision pieces in NumPy: the SimCLR InfoNCE (NT-Xent) loss, how temperature and batch
size change it, MAE random masking, and why DINO needs both centring and sharpening to avoid collapse.

All embeddings are SYNTHETIC random vectors. Deterministic.
Run from guide/:  uv run code/25-computer-vision/ssl_scratch.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, TEAL, plt, save  # noqa: E402

rng = np.random.default_rng(0)


def normalize(z):
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


# ---------- 1. InfoNCE / NT-Xent: each view must pick out its twin among 2N - 1 candidates ----------
def info_nce(z1, z2, tau=0.1):
    """z1, z2: [N, D] unit vectors; row i of z1 and row i of z2 are two views of image i."""
    n = len(z1)
    z = np.concatenate([z1, z2])                      # [2N, D]
    sim = z @ z.T / tau                               # [2N, 2N] cosine / temperature
    np.fill_diagonal(sim, -np.inf)                    # a view is never its own candidate
    target = np.concatenate([np.arange(n, 2 * n), np.arange(n)])   # view i's twin sits N rows away
    return float(-log_softmax(sim)[np.arange(2 * n), target].mean())


# ---------- 2. MAE masking: keep 25 percent of patches, hide the rest ----------
def random_mask(num_patches, ratio=0.75, seed=0):
    perm = np.random.default_rng(seed).permutation(num_patches)
    keep = int(num_patches * (1 - ratio))
    return np.sort(perm[:keep]), np.sort(perm[keep:])


# ---------- 3. DINO teacher: centring + sharpening ----------
def teacher_probs(logits, centre, temp):
    return np.exp(log_softmax((logits - centre) / temp))


def collapse_report(p):
    entropy = -(p * np.log(p + 1e-12)).sum(-1).mean()
    return len(set(p.argmax(-1))), entropy


if __name__ == "__main__":
    print("== worked example: N = 2 images, 4 views, tau = 0.5 ==")
    z1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    z2 = np.array([[0.8, 0.6], [0.6, 0.8]])
    z = np.concatenate([z1, z2])
    print("cosine matrix of the 4 views:\n", (z @ z.T).round(3))
    print(f"InfoNCE = {info_nce(z1, z2, tau=0.5):.4f}")

    print("\n== identical vs random pairs, N = 16, D = 32, tau = 0.1 ==")
    a = normalize(rng.standard_normal((16, 32)))
    print(f"identical views: {info_nce(a, a, 0.1):.4f}")
    print(f"random views:    {info_nce(a, normalize(rng.standard_normal((16, 32))), 0.1):.4f}   "
          f"(chance level log(2N - 1) = log 31 = {np.log(31):.4f})")

    print("\n== temperature sweep (aligned = noisy copies, random = unrelated) ==")
    noisy = normalize(a + 0.3 * rng.standard_normal(a.shape))
    rand = normalize(rng.standard_normal(a.shape))
    taus = [0.05, 0.1, 0.2, 0.5, 1.0]
    al, ra = [], []
    for t in taus:
        al.append(info_nce(a, noisy, t))
        ra.append(info_nce(a, rand, t))
        print(f"tau {t:<4}: aligned {al[-1]:.4f}   random {ra[-1]:.4f}")

    print("\n== negatives per anchor = 2N - 2 ==")
    for n in (32, 256, 4096):
        print(f"batch N = {n:>4}: {2 * n - 2:>4} negatives, chance-level loss log(2N - 1) = {np.log(2 * n - 1):.3f}")

    print("\n== MAE masking ==")
    vis, hid = random_mask(196)
    print(f"196 patches -> encoder sees {len(vis)}, decoder reconstructs {len(hid)}; first visible: {vis[:6].tolist()}")
    print(f"encoder self-attention scores: full {197 ** 2:,} vs visible-only {(len(vis) + 1) ** 2:,} "
          f"({197 ** 2 / (len(vis) + 1) ** 2:.1f}x fewer)")

    print("\n== DINO: 64 samples, 16 output dims, a shared offset dominates dim 3 ==")
    logits = rng.standard_normal((64, 16))
    logits[:, 3] += 6.0                               # every sample's logits share one big direction
    centre = logits.mean(0)
    for name, c, t in [("no centring, sharp (temp 0.04)", 0, 0.04),
                       ("centring, no sharpening (temp 10)", centre, 10.0),
                       ("centring + sharpening (temp 0.04)", centre, 0.04)]:
        k, h = collapse_report(teacher_probs(logits, c, t))
        print(f"{name:<36} distinct argmax = {k:>2} / 16, mean entropy = {abs(h):.3f} (uniform = {np.log(16):.3f})")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(taus, al, "o-", color=TEAL, label="aligned views")
    axes[0].plot(taus, ra, "o-", color=ORANGE, label="random views")
    axes[0].axhline(np.log(31), color=BLUE, ls="--", lw=1, label="chance: log 31")
    axes[0].set(xscale="log", xlabel="temperature τ", ylabel="InfoNCE loss", title="InfoNCE vs temperature (N = 16, SYNTHETIC)")
    axes[0].legend(fontsize=8)
    grid = np.zeros(196)
    grid[vis] = 1
    axes[1].imshow(grid.reshape(14, 14), cmap="Blues", vmin=-0.3, vmax=1)
    axes[1].set(title="MAE: 49 of 196 patches visible (75% masked)", xticks=[], yticks=[])
    axes[1].grid(False)
    save(fig, "self-supervised-vision")
