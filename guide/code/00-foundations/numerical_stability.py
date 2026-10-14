# Adapted from AI Engineering from Scratch (MIT), phases/01-math-foundations/13-numerical-stability
"""Floating-point limits, the log-sum-exp trick, stable softmax and cross-entropy, fp16 vs bf16,
loss scaling, and centered-difference gradient checks.

All inputs are small hand-picked numbers (synthetic). bfloat16 is emulated by rounding float32 bit
patterns, since NumPy has no bfloat16 type.
Run from guide/:  uv run code/00-foundations/numerical_stability.py
"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)   # we overflow on purpose


# --- bfloat16 emulation ------------------------------------------------------
def to_bf16(x):
    """Round float32 to bfloat16 (keep the top 16 bits, round to nearest even), return as float32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


# --- Unstable vs stable softmax and log-sum-exp -----------------------------------
def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()


def softmax_stable(z):
    e = np.exp(z - z.max())        # largest exponent is exp(0) = 1: cannot overflow
    return e / e.sum()


def logsumexp_naive(z):
    return np.log(np.exp(z).sum())


def logsumexp_stable(z):
    m = z.max()
    return m + np.log(np.exp(z - m).sum())   # sum >= 1, so the log is never log(0)


def cross_entropy_naive(z, y):
    return -np.log(softmax_stable(z)[y])     # even a stable softmax can underflow to 0 here


def cross_entropy_stable(z, y):
    return -(z[y] - logsumexp_stable(z))     # log-softmax: never takes log of a probability


# --- Gradient checking -------------------------------------------------------------
def f(x):
    return logsumexp_stable(x)                # gradient of log-sum-exp is softmax(x)


def numerical_grad(fn, x, h=1e-5):
    g = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (fn(xp) - fn(xm)) / (2 * h)   # centered difference: error O(h^2)
    return g


def rel_error(a, b):
    return np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-12)


def main():
    print("=== Formats (from np.finfo; bf16 from its bit layout) ===")
    for name, t in [("float64", np.float64), ("float32", np.float32), ("float16", np.float16)]:
        fi = np.finfo(t)
        print(f"{name:8s} max={float(fi.max):.4g}  min normal={float(fi.tiny):.4g}  eps={float(fi.eps):.4g}")
    print(f"bfloat16 max={float(np.uint32(0x7F7F0000).view(np.float32)):.4g}  min normal={2.0**-126:.4g}  eps={2.0**-7:.4g}")

    print("\n=== Rounding ===")
    print(f"0.1 + 0.2 = {0.1 + 0.2!r}  (== 0.3? {0.1 + 0.2 == 0.3}; isclose? {math.isclose(0.1 + 0.2, 0.3)})")

    x = np.float32(10_000) + np.arange(1, 5, dtype=np.float32)          # 10001..10004, true var 1.25
    naive_var = (x * x).mean() - x.mean() ** 2
    two_pass = ((x - x.mean()) ** 2).mean()
    print(f"variance of {x.tolist()} in float32: E[x^2]-E[x]^2 = {naive_var:.4f}, two-pass = {two_pass:.4f}")

    print("\n=== Softmax of [100, 101, 102] in float32 ===")
    z = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    print(f"exp(z) = {np.exp(z)}  -> naive softmax = {softmax_naive(z)}")
    print(f"z - max = {z - z.max()}  -> stable softmax = {np.round(softmax_stable(z), 4)}")

    print("\n=== log-sum-exp ===")
    for z in (np.array([1000.0, 1001.0]), np.array([-1000.0, -1001.0])):
        print(f"z={z.tolist()}: naive {logsumexp_naive(z)}, stable {logsumexp_stable(z):.4f}")

    print("\n=== Cross-entropy from logits [0, 200], true class 0 (float32) ===")
    z = np.array([0.0, 200.0], dtype=np.float32)
    print(f"-log(softmax): {cross_entropy_naive(z, 0)}   log-softmax: {cross_entropy_stable(z, 0):.4f}")

    print("\n=== fp16 vs bf16 ===")
    for v in (70_000.0, 1e-8, 1 + 1 / 256, 0.1):
        print(f"{v!r:>22}: fp16 {float(np.float16(v))!r:>24}  bf16 {float(to_bf16(v))!r}")

    print("\n=== Loss scaling (fp16 gradients) ===")
    grads = np.array([2e-8, 1e-7, 2e-4], dtype=np.float32)
    scale = 1024.0
    unscaled = grads.astype(np.float16)
    scaled = (grads * scale).astype(np.float16)
    recovered = scaled.astype(np.float32) / scale
    print(f"true grads        {grads}")
    print(f"cast to fp16      {unscaled.astype(np.float32)}")
    print(f"x1024,fp16,/1024   {recovered}")

    print("\n=== Gradient check: d/dx logsumexp(x) = softmax(x) ===")
    x = np.array([0.5, -1.2, 2.0])
    analytic = softmax_stable(x)
    numeric = numerical_grad(f, x)
    print(f"analytic {np.round(analytic, 6)}  numeric {np.round(numeric, 6)}  rel error {rel_error(analytic, numeric):.2e}")
    buggy = analytic.copy()
    buggy[0] *= 1.01                               # a 1% bug in one component
    print(f"with a 1% bug: rel error {rel_error(buggy, numeric):.2e}")

    # --- Figure -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    tops = np.linspace(0, 120, 241, dtype=np.float32)
    naive = [logsumexp_naive(np.array([t, t - 1, t - 2], dtype=np.float32)) for t in tops]
    stable = [logsumexp_stable(np.array([t, t - 1, t - 2], dtype=np.float32)) for t in tops]
    axes[0].plot(tops, stable, color=TEAL, lw=3, label="stable (subtract max)")
    axes[0].plot(tops, np.where(np.isfinite(naive), naive, np.nan), color=ORANGE, ls="--", label="naive: inf once Σexp > 3.4e38")
    axes[0].axvline(88.72 - math.log(1 + math.exp(-1) + math.exp(-2)), color=GRAY, lw=1)
    axes[0].set(title="log-sum-exp in float32", xlabel="largest logit", ylabel="log Σ exp(z)")
    axes[0].legend(loc="upper left")

    hs = np.logspace(-13, -1, 49)
    x0 = np.array([0.5, -1.2, 2.0])
    true = softmax_stable(x0)[0]
    centered = [abs((f(x0 + [h, 0, 0]) - f(x0 - [h, 0, 0])) / (2 * h) - true) for h in hs]
    forward = [abs((f(x0 + [h, 0, 0]) - f(x0)) / h - true) for h in hs]
    axes[1].loglog(hs, forward, color=ORANGE, label="forward (f(x+h)-f(x))/h")
    axes[1].loglog(hs, centered, color=BLUE, label="centered, /(2h)")
    axes[1].set(title="Gradient-check error vs step h", xlabel="h", ylabel="|numeric - analytic|")
    axes[1].legend(loc="lower left", fontsize=8)

    names = ["float32", "float16", "bfloat16"]
    max_exp = [math.log10(3.4e38), math.log10(65504), math.log10(3.39e38)]
    digits = [-math.log10(2.0**-23), -math.log10(2.0**-10), -math.log10(2.0**-7)]
    xs = np.arange(3)
    axes[2].bar(xs - 0.2, max_exp, 0.4, color=BLUE, label="range: log10(max)")
    axes[2].bar(xs + 0.2, digits, 0.4, color=TEAL, label="precision: decimal digits")
    for i in range(3):
        axes[2].text(i - 0.2, max_exp[i] + 0.6, f"{max_exp[i]:.1f}", ha="center", fontsize=8)
        axes[2].text(i + 0.2, digits[i] + 0.6, f"{digits[i]:.1f}", ha="center", fontsize=8)
    axes[2].set_xticks(xs, names)
    axes[2].set(title="Range vs precision", ylim=(0, 52))
    axes[2].legend(loc="upper center", fontsize=8, ncol=2)
    save(fig, "numerical-stability")


if __name__ == "__main__":
    main()
