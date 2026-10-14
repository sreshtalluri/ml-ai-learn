# Adapted from AI Engineering from Scratch (MIT), phases/04-computer-vision/14-vision-transformers
"""A Vision Transformer forward pass in NumPy: patchify, patch embedding (and proof that it equals a
conv with kernel = stride = patch size), [CLS] token, position embeddings, one pre-LN encoder block,
the ViT-B/16 parameter count, and how patch size sets the token count and attention cost.

Weights are random and the image is SYNTHETIC; the point is shapes and arithmetic. Deterministic.
Run from guide/:  uv run code/25-computer-vision/vit_scratch.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, plt, save  # noqa: E402

rng = np.random.default_rng(0)


# ---------- 1. Image -> patches -> tokens ----------
def patchify(img, p):
    """img: [H, W, C] -> [N, p*p*C] with N = (H/p) * (W/p), patches in row-major order."""
    h, w, c = img.shape
    assert h % p == 0 and w % p == 0, "image size must be divisible by the patch size"
    x = img.reshape(h // p, p, w // p, p, c)        # split both spatial axes into (grid, within-patch)
    x = x.transpose(0, 2, 1, 3, 4)                  # [gh, gw, p, p, C]: group pixels by patch
    return x.reshape(-1, p * p * c)


def conv_patch_embed(img, kernel, p):
    """The same thing as a conv with kernel = stride = p (no padding). kernel: [D, p, p, C]."""
    h, w, _ = img.shape
    out = []
    for i in range(0, h, p):
        for j in range(0, w, p):
            out.append(np.tensordot(kernel, img[i:i + p, j:j + p], axes=([1, 2, 3], [0, 1, 2])))
    return np.array(out)


# ---------- 2. One pre-LN transformer encoder block ----------
def layer_norm(x, eps=1e-6):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def encoder_block(x, P, heads):
    n, d = x.shape
    dh = d // heads
    h = layer_norm(x)                                                   # pre-LN: normalise the input of the sublayer
    q, k, v = ((h @ P[w]).reshape(n, heads, dh).transpose(1, 0, 2) for w in ("Wq", "Wk", "Wv"))
    a = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(dh))                 # [heads, n, n]: every patch sees every patch
    x = x + (a @ v).transpose(1, 0, 2).reshape(n, d) @ P["Wo"]          # residual 1
    x = x + gelu(layer_norm(x) @ P["W1"]) @ P["W2"]                     # residual 2: MLP d -> 4d -> d
    return x, a


def vit_params(d=768, depth=12, p=16, img=224, c=3, classes=1000, mlp=4):
    n = (img // p) ** 2
    patch = p * p * c * d + d
    tokens = d + (n + 1) * d                                           # [CLS] + position embeddings
    block = 2 * 2 * d + (d * 3 * d + 3 * d) + (d * d + d) + (d * mlp * d + mlp * d) + (mlp * d * d + d)
    return patch + tokens + depth * block + 2 * d + d * classes + classes, block


if __name__ == "__main__":
    # tiny ViT: 32x32 RGB image, 8x8 patches, width 16, 2 heads
    H, P_, C, D, HEADS = 32, 8, 3, 16, 2
    img = rng.random((H, H, C))                                         # SYNTHETIC pixels
    patches = patchify(img, P_)
    W_e = rng.standard_normal((P_ * P_ * C, D)) * 0.02
    tok = patches @ W_e                                                 # patch embedding = one matmul
    kernel = W_e.T.reshape(D, P_, P_, C)
    print(f"image {img.shape} -> patches {patches.shape} -> tokens {tok.shape}")
    print(f"matmul patch embedding == conv(kernel=stride={P_}): {np.allclose(tok, conv_patch_embed(img, kernel, P_))}")

    cls = rng.standard_normal((1, D)) * 0.02
    pos = rng.standard_normal((tok.shape[0] + 1, D)) * 0.02
    x = np.concatenate([cls, tok]) + pos
    print(f"with [CLS] and position embeddings: {x.shape}")
    params = {k: rng.standard_normal(s) * 0.1 for k, s in
              [("Wq", (D, D)), ("Wk", (D, D)), ("Wv", (D, D)), ("Wo", (D, D)), ("W1", (D, 4 * D)), ("W2", (4 * D, D))]}
    y, attn = encoder_block(x, params, HEADS)
    print(f"after one encoder block: {y.shape}; attention maps {attn.shape}; rows sum to 1: {np.allclose(attn.sum(-1), 1)}")
    print(f"classifier reads y[0] (the [CLS] token): shape {y[0].shape}")

    print("\n== ViT-B/16 parameter count ==")
    total, block = vit_params()
    print(f"patch embedding: {16 * 16 * 3 * 768 + 768:,}")
    print(f"one encoder block: {block:,}")
    print(f"total (224x224, 1000 classes): {total:,}")

    print("\n== patch size vs tokens and attention cost at 224x224 ==")
    rows = []
    for p in (32, 16, 14, 8):
        n = (224 // p) ** 2
        rows.append((p, n, (n + 1) ** 2))
        print(f"patch {p:>2}: {224 // p}x{224 // p} = {n:>4} patch tokens, {(n + 1) ** 2:>9,} attention scores per head per layer")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.4]})
    yy, xx = np.mgrid[:224, :224]
    pic = 0.3 + 0.4 * (xx / 224) + 0.3 * (((xx - 120) ** 2 + (yy - 100) ** 2) < 55 ** 2)
    axes[0].imshow(pic, cmap="Blues", vmin=0, vmax=1.2)
    for t in range(0, 225, 16):
        axes[0].axhline(t - 0.5, color="white", lw=0.8)
        axes[0].axvline(t - 0.5, color="white", lw=0.8)
    axes[0].set(title="224x224 SYNTHETIC image, 16x16 patches\n= 14x14 grid = 196 tokens", xticks=[], yticks=[])
    axes[0].grid(False)
    labels = [f"p={p}" for p, _, _ in rows]
    axes[1].bar(labels, [r[1] for r in rows], color=BLUE, label="tokens")
    ax2 = axes[1].twinx()
    ax2.plot(labels, [r[2] / 1e3 for r in rows], "o-", color=ORANGE, label="attention scores (thousands)")
    ax2.set_ylabel("attention scores per head per layer (thousands)", color=ORANGE)
    ax2.grid(False)
    axes[1].set(title="Smaller patches: 4x tokens, ~16x attention", ylabel="patch tokens")
    save(fig, "vision-transformers")
