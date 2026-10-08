"""Convolution and pooling by hand, an unrolled RNN, and diffusion forward-noising, with figures.

Run from guide/:  uv run code/13-deep-architectures/architectures.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Convolution (cross-correlation, as deep-learning libraries compute it)
# ---------------------------------------------------------------------------
img = np.array([
    [0, 0, 0, 9, 9],
    [0, 0, 0, 9, 9],
    [0, 0, 0, 9, 9],
    [0, 0, 0, 9, 9],
    [0, 0, 0, 9, 9],
], dtype=float)
kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)   # vertical-edge detector


def conv2d(x, k, stride=1, pad=0):
    x = np.pad(x, pad)
    kh, kw = k.shape
    oh, ow = (x.shape[0] - kh) // stride + 1, (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i * stride:i * stride + kh, j * stride:j * stride + kw] * k)
    return out


fmap = conv2d(img, kernel)
print("feature map (no padding, stride 1):\n", fmap)
print("top-left cell:", img[0:3, 0:3].ravel(), "*", kernel.ravel(), "=", np.sum(img[0:3, 0:3] * kernel))
print("top-middle cell:", np.sum(img[0:3, 1:4] * kernel))
print("output size formula (5 - 3 + 2*0)/1 + 1 =", (5 - 3) // 1 + 1, "| with padding 1:", conv2d(img, kernel, pad=1).shape, "| stride 2:", conv2d(img, kernel, stride=2).shape)
x4 = np.array([[1, 3, 2, 1], [4, 6, 5, 0], [1, 2, 9, 8], [3, 1, 4, 7]], dtype=float)
pooled = x4.reshape(2, 2, 2, 2).max(axis=(1, 3))
print("max pool 2x2 of\n", x4, "\n->\n", pooled)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw={"width_ratios": [5, 3, 3, 2]})
for ax, m, title, cmap in [(axes[0], img, "input image 5×5", "Greys"), (axes[1], kernel, "kernel 3×3 (vertical edges)", "coolwarm"),
                           (axes[2], fmap, "feature map 3×3", "coolwarm"), (axes[3], pooled, "max-pool 2×2\nof a 4×4 map", "Purples")]:
    ax.imshow(m, cmap=cmap)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=9, color="black")
    ax.set(title=title, xticks=[], yticks=[])
    ax.title.set_fontsize(9)
    ax.grid(False)
axes[0].add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor=ORANGE, lw=2.5))
save(fig, "convolutional-networks")

# ---------------------------------------------------------------------------
# 2. A scalar RNN unrolled through time, and gradient decay through time
# ---------------------------------------------------------------------------
w_x, w_h, b = 0.5, 0.8, 0.0
xs = [1.0, 0.0, 2.0]
h = 0.0
print("\nRNN h_t = tanh(w_x x_t + w_h h_{t-1} + b), w_x=0.5, w_h=0.8")
for t, xt in enumerate(xs, 1):
    pre = w_x * xt + w_h * h + b
    h = np.tanh(pre)
    print(f"  t={t}: pre-activation {pre:.4f}  h={h:.4f}")
steps = np.arange(0, 50)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
for wh, col in [(0.5, BLUE), (0.9, TEAL), (1.1, ORANGE)]:
    ax.plot(steps, wh ** steps, color=col, lw=2, label=f"|w_h| = {wh}")
ax.set(yscale="log", title="Gradient flowing back k steps ∝ w_hᵏ (linear RNN)", xlabel="k (steps back in time)", ylabel="relative gradient")
ax.title.set_fontsize(10)
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
T = 12
rng = np.random.default_rng(1)
seq = rng.normal(size=T)
hs, h = [], 0.0
for xt in seq:
    h = np.tanh(0.9 * xt + 0.7 * h)
    hs.append(h)
ax.bar(range(T), seq, color=PURPLE, alpha=0.35, label="input xₜ")
ax.plot(range(T), hs, "-o", color=ORANGE, label="hidden state hₜ")
ax.set(title="The hidden state carries a running summary", xlabel="time step t")
ax.title.set_fontsize(10)
ax.legend(frameon=False, fontsize=8)
save(fig, "recurrent-networks")

# ---------------------------------------------------------------------------
# 3. Diffusion forward process on a SYNTHETIC 2D spiral
# ---------------------------------------------------------------------------
n = 1000
theta = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
x0 = np.c_[theta * np.cos(theta), theta * np.sin(theta)] / 5 + rng.normal(0, 0.05, (n, 2))
betas = np.linspace(1e-4, 0.05, 200)
alpha_bar = np.cumprod(1 - betas)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharex=True, sharey=True)
for ax, t in zip(axes, [0, 20, 60, 199]):
    a = alpha_bar[t]
    xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * rng.normal(size=x0.shape)
    ax.scatter(*xt.T, s=3, color=BLUE)
    ax.set(title=f"t = {t + 1}: ᾱ = {a:.3f}", xticks=[], yticks=[], xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    ax.title.set_fontsize(9)
print(f"\ndiffusion: alpha_bar at t=1 {alpha_bar[0]:.4f}, t=200 {alpha_bar[-1]:.4f}")
fig.suptitle("Forward diffusion: add a little Gaussian noise each step until only noise remains (the model learns to reverse it)", fontsize=10)
save(fig, "autoencoders-diffusion-and-transfer")
