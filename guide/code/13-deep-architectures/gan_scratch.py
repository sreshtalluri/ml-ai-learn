# Adapted from AI Engineering from Scratch (MIT), phases/08-generative-ai/03-gans-generator-discriminator
"""A 1-D GAN in NumPy: generator and discriminator MLPs with hand-written backprop.

Data is SYNTHETIC: a two-mode Gaussian mixture at -2 and +2 (std 0.4).
Shows the saturating vs non-saturating generator gradient, D(real)/D(fake) over training,
and mode coverage. Fixed seed, CPU only, runs in a few seconds.
Run from guide/:  uv run code/13-deep-architectures/gan_scratch.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


def sample_real(rng, n):
    return (np.where(rng.random(n) < 0.5, -2.0, 2.0) + 0.4 * rng.standard_normal(n))[:, None]


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


class MLP:
    """in -> h -> h -> out with LeakyReLU(0.2); backward() also returns d(loss)/d(input)."""

    def __init__(self, rng, d_in, d_out, h=32):
        s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2 / a)  # noqa: E731
        self.p = {"W1": s(d_in, h), "b1": np.zeros(h), "W2": s(h, h), "b2": np.zeros(h),
                  "W3": s(h, d_out), "b3": np.zeros(d_out)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        p = self.p
        a1 = x @ p["W1"] + p["b1"]
        h1 = np.where(a1 > 0, a1, 0.2 * a1)
        a2 = h1 @ p["W2"] + p["b2"]
        h2 = np.where(a2 > 0, a2, 0.2 * a2)
        self.cache = (x, a1, h1, a2, h2)
        return h2 @ p["W3"] + p["b3"]

    def backward(self, d_out):
        x, a1, h1, a2, h2 = self.cache
        p = self.p
        g = {"W3": h2.T @ d_out, "b3": d_out.sum(0)}
        d2 = (d_out @ p["W3"].T) * np.where(a2 > 0, 1, 0.2)
        g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
        d1 = (d2 @ p["W2"].T) * np.where(a1 > 0, 1, 0.2)
        g["W1"], g["b1"] = x.T @ d1, d1.sum(0)
        return g, d1 @ p["W1"].T                  # parameter grads, input grad

    def adam(self, g, lr, b1=0.5, b2=0.999):      # beta1 = 0.5 is the DCGAN default
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh, vh = self.m[k] / (1 - b1**self.t), self.v[k] / (1 - b2**self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def d_step(D, real, fake, lr):
    """Binary cross-entropy: real -> 1, fake -> 0. d(BCE)/d(logit) = sigmoid(logit) - label."""
    x = np.vstack([real, fake])
    y = np.vstack([np.ones_like(real), np.zeros_like(fake)])
    p = sigmoid(D.forward(x))
    g, _ = D.backward((p - y) / len(x))
    D.adam(g, lr)
    return p[: len(real)].mean(), p[len(real):].mean()


def g_step(G, D, z, lr, non_saturating=True):
    fake = G.forward(z)
    p = sigmoid(D.forward(fake))
    # non-saturating: minimise -log D(G(z))   -> dL/dlogit = p - 1
    # saturating:     minimise log(1 - D(G(z))) -> dL/dlogit = -p
    d_logit = (p - 1) if non_saturating else -p
    _, d_fake = D.backward(d_logit / len(z))      # gradient flows through D to its input...
    g, _ = G.backward(d_fake)                    # ...then into G (D is NOT updated)
    G.adam(g, lr)


def train(seed, steps=3000, non_saturating=True, snap_at=(), d_warmup=0, g_start=0.0):
    rng = np.random.default_rng(seed)
    snap_rng = np.random.default_rng(seed + 1000)   # separate stream: snapshots don't change training
    G, D = MLP(rng, 4, 1), MLP(rng, 1, 1)
    G.p["b3"] += g_start                             # optional: start G's samples far from the data
    for _ in range(d_warmup):                        # optional: let D get confident before G moves
        d_step(D, sample_real(rng, 64), G.forward(rng.standard_normal((64, 4))), lr=1e-3)
    log, snaps = [], {}
    for step in range(steps + 1):
        if step in snap_at:
            snaps[step] = G.forward(snap_rng.standard_normal((2000, 4)))[:, 0]
        if step == steps:
            break
        real = sample_real(rng, 64)
        fake = G.forward(rng.standard_normal((64, 4)))
        d_real, d_fake = d_step(D, real, fake, lr=1e-3)
        g_step(G, D, rng.standard_normal((64, 4)), lr=1e-3, non_saturating=non_saturating)
        log.append((d_real, d_fake))
    xs = G.forward(rng.standard_normal((4000, 4)))[:, 0]
    return xs, np.array(log), snaps


def coverage(xs):
    left = np.mean(np.abs(xs + 2) < 1)
    right = np.mean(np.abs(xs - 2) < 1)
    return left, right


def w1(a, b):
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def main():
    print("== Worked example: generator gradient when D confidently rejects a fake ==")
    p = 0.01
    a = np.log(p / (1 - p))
    print(f"D(G(z)) = {p}, logit = {a:.3f}")
    print(f"saturating     loss log(1 - D): d/dlogit = -D       = {-p:+.3f}")
    print(f"non-saturating loss -log D:     d/dlogit = -(1 - D) = {-(1 - p):+.3f}  ({(1 - p) / p:.0f}x larger)")

    ref = sample_real(np.random.default_rng(99), 4000)[:, 0]
    snap_at = (0, 300, 1000, 3000)
    xs, log, snaps = train(seed=1, snap_at=snap_at)
    print("\n== Non-saturating GAN, 3000 steps ==")
    print(f"{'step':>5} | {'D(real)':>7} | {'D(fake)':>7}")
    for s in [1, 100, 300, 1000, 2000, 3000]:
        r, f = log[max(0, s - 50):s].mean(axis=0)
        print(f"{s:>5} | {r:>7.2f} | {f:>7.2f}")
    l, r = coverage(xs)
    print(f"mode coverage: {l:.2f} near -2, {r:.2f} near +2 (data: 0.50 / 0.50 minus tails)")
    print(f"W1 to data: {w1(xs, ref):.3f}")

    print("\n== Hard start: G's samples begin near x = 8, D pre-trained 300 steps, 1500 steps ==")
    print(f"{'seed':>4} | {'D(fake) at start':>16} | {'near -2':>7} | {'near +2':>7}")
    collapsed = None
    for seed in range(3):
        xs_h, log_h, _ = train(seed, steps=1500, d_warmup=300, g_start=8.0)
        collapsed = xs_h if collapsed is None else collapsed
        l, r = coverage(xs_h)
        print(f"{seed:>4} | {log_h[0, 1]:>16.3f} | {l:>7.2f} | {r:>7.2f}")

    # ---------------------------- figure ----------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(-4.5, 4.5, 70)
    ax[0].hist(ref, bins=bins, density=True, color=GRAY, alpha=0.4, label="real data")
    for (s, x), c in zip(snaps.items(), [ORANGE, PURPLE, TEAL, BLUE]):
        ax[0].hist(x, bins=bins, density=True, histtype="step", lw=1.6, color=c, label=f"G at step {s}")
    ax[0].set(title="Generator samples during training", xlabel="x", ylim=(0, 1.1))
    ax[0].legend(fontsize=8)
    ax[1].hist(ref, bins=bins, density=True, color=GRAY, alpha=0.4, label="real data")
    ax[1].hist(xs, bins=bins, density=True, histtype="step", lw=1.6, color=BLUE, label="normal start")
    ax[1].hist(collapsed, bins=bins, density=True, histtype="step", lw=1.6, color=ORANGE,
               label="hard start: mode collapse")
    ax[1].set(title="Final generators: both modes vs one", xlabel="x")
    ax[1].legend(fontsize=8)
    save(fig, "gans-generator-discriminator")


if __name__ == "__main__":
    main()
