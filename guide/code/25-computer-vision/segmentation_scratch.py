# Adapted from AI Engineering from Scratch (MIT), phases/04-computer-vision/07-semantic-segmentation-unet
"""Semantic segmentation pieces in NumPy: pixel cross-entropy, soft Dice, per-class IoU, the
class-imbalance trap, why U-Net needs skip connections, the U-Net shape/parameter count, and
why some transposed convolutions make checkerboards.

Images and masks are SYNTHETIC (shapes on a plain background). Deterministic.
Run from guide/:  uv run code/25-computer-vision/segmentation_scratch.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import plt, save  # noqa: E402


# ---------- 1. Losses and metrics on a pixel grid ----------
def softmax(z, axis=0):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def pixel_cross_entropy(logits, target):
    """logits: [C, H, W]; target: [H, W] integer class ids. Mean over pixels."""
    p = softmax(logits, axis=0)
    h, w = target.shape
    p_true = p[target, np.arange(h)[:, None], np.arange(w)[None, :]]   # prob of the true class at each pixel
    return float(-np.log(p_true + 1e-12).mean())


def soft_dice_loss(logits, target, eps=1e-6):
    """1 - mean over classes of 2|P∩Y| / (|P| + |Y|), with P = softmax probabilities."""
    c = logits.shape[0]
    p = softmax(logits, axis=0)
    y = np.eye(c)[target].transpose(2, 0, 1)            # one-hot [C, H, W]
    inter = (p * y).sum(axis=(1, 2))
    denom = p.sum(axis=(1, 2)) + y.sum(axis=(1, 2))
    return float(1 - ((2 * inter + eps) / (denom + eps)).mean())


def iou_per_class(pred, target, c):
    out = []
    for k in range(c):
        inter = np.sum((pred == k) & (target == k))
        union = np.sum((pred == k) | (target == k))
        out.append(inter / union if union else np.nan)    # nan = class absent from both: skip it in the mean
    return np.array(out)


def dice_per_class(pred, target, c):
    return np.array([2 * np.sum((pred == k) & (target == k)) / (np.sum(pred == k) + np.sum(target == k))
                     for k in range(c)])


# ---------- 2. Synthetic data: a small object on a big background ----------
def make_scene(size=64):
    yy, xx = np.mgrid[:size, :size]
    mask = np.zeros((size, size), dtype=int)
    mask[(xx - 20) ** 2 + (yy - 22) ** 2 < 6 ** 2] = 1                  # small circle: class 1
    mask[(np.abs(xx - 44) < 5) & (np.abs(yy - 42) < 5)] = 2             # small square: class 2
    rng = np.random.default_rng(0)
    img = np.choose(mask, [0.2, 0.8, 0.55]) + 0.05 * rng.standard_normal(mask.shape)
    return img, mask


def down_up(mask, factor, c):
    """Squeeze the mask through a bottleneck of size/factor (majority vote per block), then blow it back up.
    This is what a decoder WITHOUT skip connections has to work with."""
    s = mask.shape[0] // factor
    onehot = np.eye(c)[mask].reshape(s, factor, s, factor, c).mean(axis=(1, 3))   # [s, s, C] class fractions
    coarse = onehot.argmax(-1)
    return np.repeat(np.repeat(coarse, factor, 0), factor, 1)


# ---------- 3. U-Net shapes and parameter count (bilinear-upsample variant) ----------
def double_conv(cin, cout):            # conv3x3 (no bias) + BN + ReLU, twice
    return cin * cout * 9 + 2 * cout + cout * cout * 9 + 2 * cout


def unet_params(base=64, cin=3, classes=2):
    ch = [base, base * 2, base * 4, base * 8, base * 16]
    n = double_conv(cin, ch[0]) + sum(double_conv(ch[i], ch[i + 1]) for i in range(4))   # encoder + bottleneck
    for i in range(4, 0, -1):                                   # decoder: concat skip, then DoubleConv
        n += double_conv(ch[i] + ch[i - 1], ch[i - 1])
    return n + base * classes + classes                         # final 1x1 conv


def transposed_overlap(kernel, stride, n_in=8):
    """How many input pixels write into each output pixel of a 1-D transposed conv."""
    out = np.zeros(stride * (n_in - 1) + kernel, dtype=int)
    for i in range(n_in):
        out[i * stride: i * stride + kernel] += 1
    return out[kernel:-kernel]                                  # interior, away from the edges


if __name__ == "__main__":
    img, mask = make_scene()
    frac = np.bincount(mask.ravel(), minlength=3) / mask.size
    print(f"class pixel fractions (background, circle, square): {frac.round(4).tolist()}")

    print("\n== the imbalance trap: predict background everywhere ==")
    lazy = np.zeros_like(mask)
    print(f"pixel accuracy = {np.mean(lazy == mask):.4f}")
    print(f"IoU per class  = {np.round(iou_per_class(lazy, mask, 3), 4).tolist()}  -> mIoU {np.nanmean(iou_per_class(lazy, mask, 3)):.4f}")

    print("\n== worked example: 4x4 mask with a 2x2 object ==")
    tgt = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    prd = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0]])
    iou = iou_per_class(prd, tgt, 2)
    dice = dice_per_class(prd, tgt, 2)
    print(f"foreground: |P|={np.sum(prd == 1)}, |Y|={np.sum(tgt == 1)}, |P∩Y|={np.sum((prd == 1) & (tgt == 1))}")
    print(f"IoU fg = {iou[1]:.4f}, Dice fg = {dice[1]:.4f}, check 2*IoU/(1+IoU) = {2 * iou[1] / (1 + iou[1]):.4f}")
    print(f"pixel accuracy = {np.mean(prd == tgt):.4f}")

    print("\n== losses on logits ==")
    good = np.stack([(mask == k) * 4.0 for k in range(3)])          # confident and right
    flat = np.zeros((3,) + mask.shape)                              # uniform 1/3 everywhere
    bg_only = np.stack([np.full(mask.shape, 4.0), np.zeros(mask.shape), np.zeros(mask.shape)])
    for name, lg in [("confident & correct", good), ("uniform (untrained)", flat), ("always background", bg_only)]:
        print(f"{name:<22} CE = {pixel_cross_entropy(lg, mask):.4f}   soft Dice loss = {soft_dice_loss(lg, mask):.4f}")

    print("\n== why skip connections: decode from a bottleneck only ==")
    for f in (2, 4, 8, 16):
        rec = down_up(mask, f, 3)
        print(f"bottleneck 1/{f:<2} ({64 // f}x{64 // f}): mIoU after upsampling = {np.nanmean(iou_per_class(rec, mask, 3)):.4f}")

    print("\n== U-Net size ==")
    s = 256
    for d in range(5):
        print(f"level {d}: {64 * 2 ** d:>4} channels x {s >> d}x{s >> d}")
    print(f"params base=64: {unet_params(64):,}   base=32: {unet_params(32):,}")

    print("\n== transposed conv overlap (interior pixels) ==")
    for k, st in [(3, 2), (4, 2), (2, 2)]:
        print(f"kernel {k}, stride {st}: contributions per output pixel = {transposed_overlap(k, st).tolist()}")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    rec16 = down_up(mask, 8, 3)
    panels = [(img, "SYNTHETIC image", "Greys_r"), (mask, "ground-truth mask", "viridis"),
              (rec16, f"decoded from 8x8 bottleneck\n(no skips) mIoU {np.nanmean(iou_per_class(rec16, mask, 3)):.2f}", "viridis"),
              (lazy, f"'always background'\npixel acc {np.mean(lazy == mask):.3f}, mIoU {np.nanmean(iou_per_class(lazy, mask, 3)):.2f}", "viridis")]
    for ax, (m, t, cmap) in zip(axes, panels):
        ax.imshow(m, cmap=cmap, vmin=0, vmax=2 if cmap == "viridis" else None)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    save(fig, "semantic-segmentation-unet")
