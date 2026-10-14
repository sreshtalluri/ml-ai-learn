# Adapted from AI Engineering from Scratch (MIT), phases/18-ethics-safety-alignment/22-differential-privacy-for-llms
"""Differential privacy from scratch: the Laplace mechanism, then DP gradient descent with per-example clipping,
Gaussian noise, and a Renyi-DP accountant. All data is SYNTHETIC.

The training loop is FULL-BATCH DP gradient descent (every example used every step), so the accountant below is
exact for it: T compositions of a Gaussian mechanism. Real DP-SGD samples mini-batches and gets a tighter epsilon
from privacy amplification by subsampling; libraries such as Opacus account for that.
Run from guide/:  uv run code/21-safety-security/differential_privacy.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, TEAL, plt, save  # noqa: E402

rng = np.random.default_rng(22)


# ---------- 1. Laplace mechanism on a count ----------
def laplace_count(true_count, epsilon, sensitivity=1.0):
    return true_count + rng.laplace(0.0, sensitivity / epsilon)


def laplace_demo():
    eps, b = 0.5, 1 / 0.5
    print("=== Laplace mechanism: count of patients with condition X (synthetic) ===")
    print(f"sensitivity 1, epsilon {eps} -> noise scale b = 1/{eps} = {b}")
    # Neighbouring datasets: true counts 100 (D) and 101 (D'). Probability that the released value lands in S = [99.5, 100.5].
    cdf = lambda x, mu: 0.5 * math.exp((x - mu) / b) if x < mu else 1 - 0.5 * math.exp(-(x - mu) / b)
    p_d, p_d2 = cdf(100.5, 100) - cdf(99.5, 100), cdf(100.5, 101) - cdf(99.5, 101)
    print(f"P(output in [99.5, 100.5] | count=100) = {p_d:.4f};  | count=101) = {p_d2:.4f};  ratio {p_d / p_d2:.3f} <= e^eps = {math.exp(eps):.3f}")
    draws = [laplace_count(100, eps) for _ in range(5)]
    print("five releases:", [round(x, 1) for x in draws])
    return p_d, p_d2


# ---------- 2. Renyi-DP accountant for T Gaussian mechanisms with noise multiplier sigma ----------
ALPHAS = np.concatenate([np.linspace(1.01, 10, 900), np.linspace(10, 512, 2000)])


def epsilon(sigma, steps, delta):
    """RDP of one Gaussian mechanism at order a is a / (2 sigma^2); T steps add; convert with log(1/delta)/(a-1)."""
    eps = steps * ALPHAS / (2 * sigma**2) + math.log(1 / delta) / (ALPHAS - 1)
    i = int(np.argmin(eps))
    return float(eps[i]), float(ALPHAS[i])


TRUE_W = np.linspace(-0.5, 0.5, 20)


# ---------- 3. DP gradient descent for logistic regression ----------
def make_data(n):
    X = rng.normal(size=(n, 20))                                 # 20 features, weak individual signals
    y = (X @ TRUE_W + rng.normal(0, 0.5, n) > 0).astype(float)
    return X, y


def dp_gd(X, y, sigma, clip, steps, lr=0.5):
    n, w = len(y), np.zeros(X.shape[1] + 1)
    Xb = np.hstack([X, np.ones((n, 1))])                         # [n, 21]: 20 features + a bias column
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = (p - y)[:, None] * Xb                                # per-example gradients, [n, 21]
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        g = g / np.maximum(1.0, norms / clip)                    # clip each row to L2 norm <= clip
        noisy_sum = g.sum(0) + rng.normal(0, sigma * clip, size=w.shape)  # sensitivity of the sum is `clip`
        w -= lr * noisy_sum / n
    return w


def accuracy(w, X, y):
    return float((((np.hstack([X, np.ones((len(y), 1))]) @ w) > 0) == y).mean())


if __name__ == "__main__":
    p_d, p_d2 = laplace_demo()
    assert p_d / p_d2 <= math.exp(0.5) + 1e-9

    delta, steps, clip = 1e-5, 100, 1.0
    print(f"\n=== Gaussian accountant: T={steps} full-batch steps, delta={delta} ===")
    e, a = epsilon(4.0, steps, delta)
    print(f"sigma=4: best order a={a:.2f}: {steps}*{a:.2f}/(2*16) + ln(1e5)/({a:.2f}-1) = {e:.2f}")

    print("\n=== DP gradient descent, 100 steps, clip C=1 (synthetic data) ===")
    Xte, yte = make_data(5_000)
    rows = []
    for n in (200, 2_000):
        X, y = make_data(n)
        for sigma in (0.0, 1.0, 4.0, 16.0, 64.0, 256.0):
            accs = [accuracy(dp_gd(X, y, sigma, clip, steps), Xte, yte) for _ in range(5)]
            eps = epsilon(sigma, steps, delta)[0] if sigma else float("inf")
            rows.append((n, sigma, eps, float(np.mean(accs))))
            print(f"n={n:>5}  sigma={sigma:>4}  epsilon={eps:>7.2f}  test accuracy={np.mean(accs):.3f} (mean of 5 runs)")

    sig = np.geomspace(1, 100, 60)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for T, col in [(10, TEAL), (100, BLUE), (1000, ORANGE)]:
        a1.plot(sig, [epsilon(s, T, delta)[0] for s in sig], color=col, label=f"T = {T} steps")
    a1.set(xscale="log", yscale="log", title="Privacy budget (full-batch, delta = 1e-5)",
           xlabel="noise multiplier sigma", ylabel="epsilon")
    a1.legend(frameon=False)
    for n, col in [(200, ORANGE), (2_000, BLUE)]:
        pts = [(e, acc) for nn, s, e, acc in rows if nn == n and s > 0]
        a2.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, label=f"n = {n:,} examples")
        a2.axhline([acc for nn, s, e, acc in rows if nn == n and s == 0][0], color=col, ls=":", lw=1)
    a2.set(xscale="log", title="Utility vs privacy (dotted: no noise)", xlabel="epsilon (smaller = more private)",
           ylabel="test accuracy")
    a2.legend(frameon=False)
    save(fig, "differential-privacy")
