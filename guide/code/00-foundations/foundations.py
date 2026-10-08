"""Module 0 figures and worked examples: vectors, matrices, derivatives, gradients, probability, NumPy.

Run from guide/:  uv run code/00-foundations/foundations.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

# ---------------------------------------------------------------------------
# Vectors and matrices
# ---------------------------------------------------------------------------
x = np.array([2, 1.5])            # 2 bedrooms, 1.5 thousand sq ft
w = np.array([30, 120])           # $k per bedroom, $k per thousand sq ft
b = 50
print(f"House price: x·w = {x @ w}, + b = {x @ w + b} ($k)")
a, c = np.array([3, 1]), np.array([1, 2])
print(f"a·c = {a @ c}, |a|={np.linalg.norm(a):.3f}, |c|={np.linalg.norm(c):.3f}, cos θ = {a @ c / np.linalg.norm(a) / np.linalg.norm(c):.4f}")
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[1, 0, 2], [0, 1, 1]])
print(f"A {A.shape} @ B {B.shape} = {(A @ B).shape}\n{A @ B}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
for v, col, lab in [(a, BLUE, "a = (3, 1)"), (c, ORANGE, "c = (1, 2)"), (a + c, PURPLE, "a + c = (4, 3)")]:
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    ax.text(v[0] + 0.08, v[1] + 0.08, lab, color=col, fontsize=9)
ax.annotate("", xy=a + c, xytext=a, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1, ls="--"))
ax.set(title="Vector addition: tip to tail", xlim=(-0.3, 5), ylim=(-0.3, 3.6), aspect="equal")
ax = axes[1]
u = a / np.linalg.norm(a)
proj = (c @ u) * u
ax.annotate("", xy=a, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
ax.annotate("", xy=c, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
ax.plot([c[0], proj[0]], [c[1], proj[1]], ls="--", color=GRAY)
ax.plot([0, proj[0]], [0, proj[1]], color=TEAL, lw=5, alpha=0.5, label=f"projection of c on a = {c @ u:.2f}")
ax.text(1.6, 0.3, f"a·c = 5 = |a| × (c's projection)\n= {np.linalg.norm(a):.2f} × {c @ u:.2f}", fontsize=9)
ax.set(title="Dot product as projection", xlim=(-0.3, 3.6), ylim=(-0.3, 2.6), aspect="equal")
ax.legend(frameon=False, fontsize=8, loc="upper right")
save(fig, "vectors-and-matrices")

# ---------------------------------------------------------------------------
# Calculus
# ---------------------------------------------------------------------------
f = lambda t: t**2 - 4 * t + 5          # noqa: E731
df = lambda t: 2 * t - 4                # noqa: E731
print(f"\nf(3)={f(3)}, f'(3)={df(3)}, finite difference={(f(3.001) - f(2.999)) / 0.002:.4f}")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
t = np.linspace(-0.5, 4.5, 200)
ax.plot(t, f(t), color=BLUE, lw=2, label="f(t) = t² − 4t + 5")
for t0, col in [(0.5, ORANGE), (2, TEAL), (3.5, PURPLE)]:
    tt = np.linspace(t0 - 0.9, t0 + 0.9, 10)
    ax.plot(tt, f(t0) + df(t0) * (tt - t0), color=col, lw=1.5, label=f"slope at t={t0}: {df(t0):+.0f}")
    ax.plot(t0, f(t0), "o", color=col)
ax.set(title="Derivative = slope of the tangent line", xlabel="t", ylabel="f(t)")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
gx, gy = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
L = gx**2 + 2 * gy**2
ax.contour(*np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200)),
           np.add.outer(2 * np.linspace(-2, 2, 200) ** 2, np.linspace(-2, 2, 200) ** 2), levels=10, colors="#d4d4d8")
ax.quiver(gx, gy, -2 * gx, -4 * gy, color=ORANGE, angles="xy", scale=60, width=0.004)
ax.set(title="−∇L for L = x² + 2y²: arrows point downhill", xlabel="x", ylabel="y", aspect="equal")
save(fig, "calculus-for-ml")

# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------
prior, sens, fpr = 0.01, 0.95, 0.05
post = sens * prior / (sens * prior + fpr * (1 - prior))
print(f"\nBayes: P(disease | positive) = {sens}*{prior} / ({sens}*{prior} + {fpr}*{1 - prior}) = {post:.4f}")
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
z = np.linspace(-4, 8, 400)
for mu, s, col in [(0, 1, BLUE), (2, 2, ORANGE)]:
    ax.plot(z, np.exp(-((z - mu) ** 2) / (2 * s**2)) / (s * np.sqrt(2 * np.pi)), color=col, lw=2, label=f"Normal(μ={mu}, σ={s})")
ax.set(title="Normal distributions", xlabel="x", ylabel="density")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
k = np.arange(0, 11)
from math import comb  # noqa: E402
ax.bar(k, [comb(10, i) * 0.3**i * 0.7 ** (10 - i) for i in k], color=TEAL)
ax.set(title="Binomial(n=10, p=0.3): heads in 10 flips", xlabel="k successes", ylabel="probability")
ax = axes[2]
temp = rng.normal(25, 6, 300)                           # SYNTHETIC
ice = 20 + 3 * temp + rng.normal(0, 10, 300)
drown = 2 + 0.15 * temp + rng.normal(0, 1, 300)
r = np.corrcoef(ice, drown)[0, 1]
ax.scatter(ice, drown, s=8, color=PURPLE, alpha=0.6)
ax.set(title=f"Correlated (r = {r:.2f}), not causal:\nboth driven by temperature", xlabel="ice-cream sales", ylabel="swimming incidents")
ax.title.set_fontsize(10)
print(f"ice cream vs incidents correlation r = {r:.3f}")
save(fig, "probability-and-statistics")

# ---------------------------------------------------------------------------
# Python toolkit: vectorization and broadcasting
# ---------------------------------------------------------------------------
X = rng.normal(size=(1000, 3)) * [1, 10, 100] + [0, 50, 500]   # SYNTHETIC, very different scales
Z = (X - X.mean(axis=0)) / X.std(axis=0)                       # broadcasting: [1000,3] - [3]
print(f"\nmeans before {X.mean(0).round(1)}, after {Z.mean(0).round(6)}; stds after {Z.std(0).round(6)}")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for i, col in enumerate([BLUE, ORANGE, PURPLE]):
    axes[0].hist(X[:, i], bins=40, color=col, alpha=0.6, label=f"feature {i}")
    axes[1].hist(Z[:, i], bins=40, color=col, alpha=0.5, label=f"feature {i}")
axes[0].set(title="Raw features: wildly different scales", xlabel="value")
axes[1].set(title="Standardized: (X − mean) / std", xlabel="z-score")
axes[0].legend(frameon=False, fontsize=8)
save(fig, "python-toolkit")
