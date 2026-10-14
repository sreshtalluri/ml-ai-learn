# Adapted from AI Engineering from Scratch (MIT), phases/19-capstone-projects/45-gradient-clipping-amp
"""Gradient clipping by global L2 norm, float16 range limits, and a dynamic loss scaler, in NumPy.

The gradients and the loss-spike schedule are SYNTHETIC. Deterministic (fixed seed).
Run from guide/:  uv run code/27-training-at-scale/clipping_amp.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PURPLE, plt, save  # noqa: E402


# ---------- 1. Clip by global L2 norm ----------

def clip_global_norm(grads, max_norm):
    """Scale every gradient by the same factor so the norm of all of them together is <= max_norm."""
    total = np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads))
    scale = min(1.0, max_norm / (total + 1e-6))
    return [g * scale for g in grads], total


# ---------- 2. What float16 can and cannot hold ----------

def fp16_demo():
    info = np.finfo(np.float16)
    print(f"  float16 max = {float(info.max):.0f}, smallest normal = {float(info.tiny):.2e}, "
          f"smallest subnormal = {float(np.float16(2.0**-24)):.2e}")
    with np.errstate(over="ignore"):
        print(f"  70000 in fp16       -> {np.float16(70000.0)}")     # overflow
    print(f"  1e-8 in fp16        -> {np.float16(1e-8)}")            # underflow to zero
    print(f"  1e-8 * 65536 in fp16 -> {float(np.float16(1e-8 * 65536)):.3e}  (scaled, survives)")
    print(f"  1 + 1e-4 in fp16    -> {np.float16(1.0) + np.float16(1e-4)}  (lost: spacing near 1 is {float(info.eps):.2e})")


# ---------- 3. Dynamic loss scaler ----------

class LossScaler:
    """Same policy as torch.amp.GradScaler: halve on overflow (and skip), double after `interval` clean steps."""

    def __init__(self, scale=2.0**16, interval=2000):
        self.scale, self.interval, self.clean = scale, interval, 0

    def unscale(self, grads16):
        grads = [g.astype(np.float32) / self.scale for g in grads16]
        finite = all(np.all(np.isfinite(g)) for g in grads)
        return grads, finite

    def update(self, finite):
        if not finite:
            self.scale /= 2
            self.clean = 0
        else:
            self.clean += 1
            if self.clean == self.interval:
                self.scale *= 2
                self.clean = 0


def amp_step(true_grads, scaler, max_norm=1.0):
    """Backward in fp16 with the scaled loss, then unscale, check, clip, and (maybe) step."""
    with np.errstate(over="ignore"):
        grads16 = [(g * scaler.scale).astype(np.float16) for g in true_grads]  # scaled fp16 grads
    grads, finite = scaler.unscale(grads16)
    norm = clipped = None
    if finite:
        clipped, norm = clip_global_norm(grads, max_norm)
    scaler.update(finite)                       # runs on every step, skipped or not
    return finite, norm, clipped


if __name__ == "__main__":
    print("1) Global-norm clipping, max_norm = 1.0")
    g = [np.array([3.0, 4.0]), np.array([12.0])]
    clipped, norm = clip_global_norm(g, 1.0)
    print(f"  grads {[x.tolist() for x in g]}, global norm = {norm:.1f}")
    print(f"  clipped {[np.round(x, 4).tolist() for x in clipped]}, new norm = "
          f"{np.sqrt(sum(np.sum(x**2) for x in clipped)):.4f}")
    print("  direction kept: ratio of first two entries before/after =", 3 / 4, "/", round(clipped[0][0] / clipped[0][1], 4))

    print("\n2) float16 range")
    fp16_demo()

    print("\n3) Dynamic loss scaling on SYNTHETIC gradients (tiny values ~1e-7, one spike at step 6)")
    rng = np.random.default_rng(0)
    scaler = LossScaler(scale=2.0**16, interval=3)        # short interval so doubling shows up
    log = []
    for step in range(1, 13):
        true = [rng.normal(0, 1e-7, size=1000)]
        if step == 6:
            true = [true[0] * 1e7]                         # a bad batch: gradients around 1
        scale_used = scaler.scale
        finite, norm, _ = amp_step(true, scaler)
        log.append((step, scale_used, finite, norm))
        status = f"STEP  norm={norm:.2e}" if finite else "SKIP  overflow, scale halved"
        print(f"  step {step:2d}  scale=2^{int(np.log2(scale_used)):2d}  {status}")

    print("\n  same tiny gradients without scaling:")
    tiny = rng.normal(0, 1e-7, size=1000)
    raw16 = tiny.astype(np.float16)
    print(f"  fraction flushed to zero in fp16 = {np.mean(raw16 == 0):.3f}; "
          f"with scale 2^16 = {np.mean((tiny * 2**16).astype(np.float16) == 0):.3f}")

    # Clipping keeps a training run alive: SYNTHETIC 1-D quadratic with one huge gradient spike.
    def run(max_norm):
        w, path = 2.0, []
        for t in range(40):
            grad = 2 * w + (400.0 if t == 15 else 0.0)        # spike at step 15
            if max_norm is not None:
                grad = float(np.clip(grad, -max_norm, max_norm))  # 1-D: global-norm clip == value clip
            w -= 0.1 * grad
            path.append(w * w)
        return path

    no_clip, with_clip = run(None), run(1.0)
    print(f"\n4) Loss spike: max loss after the spike without clipping = {max(no_clip[15:]):.1f}, "
          f"with clip 1.0 = {max(with_clip[15:]):.3f}")

    # ---------- figure ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7))
    a1.plot(no_clip, color=ORANGE, label="no clipping")
    a1.plot(with_clip, color=BLUE, label="clip global norm at 1.0")
    a1.axvline(15, color="#9ca3af", ls=":")
    a1.text(15.5, max(no_clip) * 0.8, "bad batch", fontsize=9)
    a1.set_yscale("log")
    a1.set_xlabel("step")
    a1.set_ylabel("loss (log scale)")
    a1.set_title("One gradient spike")
    a1.legend(frameon=False, fontsize=9)
    steps = [s for s, *_ in log]
    a2.step(steps, [np.log2(s) for _, s, _, _ in log], where="post", color=PURPLE)
    for s, sc, ok, _ in log:
        if not ok:
            a2.plot(s, np.log2(sc), "x", color="#dc2626", ms=10, mew=2)
    a2.set_xlabel("step")
    a2.set_ylabel("log2(loss scale)")
    a2.set_yticks([16, 17])
    a2.set_title("Loss scale: x = overflow, skip, halve")
    save(fig, "gradient-clipping-amp")
