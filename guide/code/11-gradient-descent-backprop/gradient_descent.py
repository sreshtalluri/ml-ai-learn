"""Gradient descent, momentum, and Adam on a narrow quadratic bowl. Worked steps + figure.

Loss: f(x, y) = 0.5 * (x^2 + 10 y^2)    gradient: (x, 10 y)
Run from guide/:  uv run code/11-gradient-descent-backprop/gradient_descent.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


def f(p):
    return 0.5 * (p[0] ** 2 + 10 * p[1] ** 2)


def grad(p):
    return np.array([p[0], 10 * p[1]])


def gd(p, lr, steps):
    path = [p.copy()]
    for _ in range(steps):
        p = p - lr * grad(p)
        path.append(p.copy())
    return np.array(path)


def momentum(p, lr, steps, beta=0.9):
    v, path = np.zeros(2), [p.copy()]
    for _ in range(steps):
        v = beta * v + grad(p)
        p = p - lr * v
        path.append(p.copy())
    return np.array(path)


def adam(p, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    m, s, path = np.zeros(2), np.zeros(2), [p.copy()]
    for t in range(1, steps + 1):
        g = grad(p)
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * g**2
        p = p - lr * (m / (1 - b1**t)) / (np.sqrt(s / (1 - b2**t)) + eps)
        path.append(p.copy())
    return np.array(path)


start = np.array([-3.0, 1.6])
print("Worked example, plain gradient descent")
for lr in (0.1, 0.19, 0.21):
    path = gd(start, lr, 3)
    print(f"  lr={lr}: " + "  ".join(f"({x:.3f}, {y:.3f}) f={f((x, y)):.3f}" for x, y in path))

m1 = momentum(start, 0.02, 2)
a1 = adam(start, 0.1, 1)
print(f"\nMomentum lr=0.02, two steps: {m1[1].round(4).tolist()} -> {m1[2].round(4).tolist()}")
print(f"Adam lr=0.1, one step: {a1[1].round(4).tolist()}")
for name, path in [("GD 0.1", gd(start, 0.1, 50)), ("GD 0.19", gd(start, 0.19, 50)), ("momentum 0.02", momentum(start, 0.02, 50)), ("Adam 0.1", adam(start, 0.1, 50))]:
    print(f"  after 50 steps {name:14s} loss={f(path[-1]):.2e}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
ax = axes[0]
gx, gy = np.meshgrid(np.linspace(-3.5, 3.5, 300), np.linspace(-2, 2, 300))
ax.contour(gx, gy, 0.5 * (gx**2 + 10 * gy**2), levels=np.geomspace(0.05, 40, 12), colors="#d4d4d8", linewidths=0.8)
runs = [("GD η=0.1", gd(start, 0.1, 30), BLUE), ("GD η=0.19 (zig-zag)", gd(start, 0.19, 30), ORANGE),
        ("Momentum η=0.02", momentum(start, 0.02, 30), PURPLE), ("Adam η=0.1", adam(start, 0.1, 30), TEAL)]
for name, path, c in runs:
    ax.plot(path[:, 0], path[:, 1], "-o", ms=2.5, lw=1.3, color=c, label=name)
ax.plot(0, 0, "k*", ms=10)
ax.set(title="Paths on f = ½(x² + 10y²), 30 steps", xlabel="θ₁", ylabel="θ₂", xlim=(-3.5, 3.5), ylim=(-2, 2))
ax.legend(frameon=False, fontsize=8, loc="lower right")

ax = axes[1]
for name, path, c in runs:
    ax.plot([f(p) for p in path], color=c, label=name)
ax.plot([f(p) for p in gd(start, 0.21, 30)], color="#dc2626", ls="--", label="GD η=0.21 (diverges)")
ax.set(title="Loss per step", xlabel="step", ylabel="loss (log scale)", yscale="log", ylim=(1e-6, 1e3))
ax.legend(frameon=False, fontsize=8)
save(fig, "gradient-descent")
