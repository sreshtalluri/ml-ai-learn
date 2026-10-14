# Adapted from AI Engineering from Scratch (MIT), phases/08-generative-ai/13-flow-matching-rectified-flows
"""Flow matching and one reflow (rectified flow) iteration on 1-D data, in NumPy.

Data is SYNTHETIC: a two-mode Gaussian mixture at -2 and +2 (std 0.3).
Convention (as in SD3): x_t = (1 - t) * x0 + t * x1, x0 = data, x1 = noise, t in [0, 1].
The network v(x, t) learns the velocity x1 - x0; sampling integrates from t = 1 down to t = 0.
Fixed seed, CPU only, runs in a few seconds.
Run from guide/:  uv run code/13-deep-architectures/flow_matching_scratch.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

rng = np.random.default_rng(0)


def sample_data(n):
    side = np.where(rng.random(n) < 0.5, -2.0, 2.0)
    return side + 0.3 * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# A two-hidden-layer MLP v(x, t) with hand-written backprop and Adam
# ---------------------------------------------------------------------------
def features(x, t):
    # [n, 4]: position, time, and two Fourier features of time
    return np.stack([x, t, np.sin(np.pi * t), np.cos(np.pi * t)], axis=1)


class MLP:
    def __init__(self, d_in=4, h=64):
        s = lambda a, b: rng.standard_normal((a, b)) / np.sqrt(a)  # noqa: E731
        self.p = {"W1": s(d_in, h), "b1": np.zeros(h), "W2": s(h, h), "b2": np.zeros(h),
                  "W3": s(h, 1), "b3": np.zeros(1)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step = 0

    def forward(self, X):
        p = self.p
        h1 = np.tanh(X @ p["W1"] + p["b1"])          # [n, h]
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])         # [n, h]
        out = (h2 @ p["W3"] + p["b3"])[:, 0]         # [n]
        self.cache = (X, h1, h2)
        return out

    def backward(self, d_out):
        X, h1, h2 = self.cache
        p = self.p
        d_out = d_out[:, None]                        # [n, 1]
        g = {"W3": h2.T @ d_out, "b3": d_out.sum(0)}
        d2 = (d_out @ p["W3"].T) * (1 - h2**2)        # back through tanh
        g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
        d1 = (d2 @ p["W2"].T) * (1 - h1**2)
        g["W1"], g["b1"] = X.T @ d1, d1.sum(0)
        return g

    def adam(self, g, lr=3e-3, b1=0.9, b2=0.999):
        self.step += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh, vh = self.m[k] / (1 - b1**self.step), self.v[k] / (1 - b2**self.step)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def __call__(self, x, t):
        return self.forward(features(x, t))


# ---------------------------------------------------------------------------
# Training: regress v(x_t, t) onto the straight-line velocity x1 - x0
# ---------------------------------------------------------------------------
def fm_train_step(net, x0, x1):
    t = rng.random(len(x0))                  # t ~ Uniform(0, 1)
    xt = (1 - t) * x0 + t * x1               # a point on the straight line
    target = x1 - x0                         # its constant velocity
    pred = net(xt, t)
    loss = np.mean((pred - target) ** 2)
    net.adam(net.backward(2 * (pred - target) / len(x0)))
    return loss


def train(net, pairs_fn, steps=3000, batch=256):
    for _ in range(steps):
        x0, x1 = pairs_fn(batch)
        loss = fm_train_step(net, x0, x1)
    return loss


def sample(net, x1, n_steps, keep_path=False):
    x, dt = x1.copy(), 1.0 / n_steps
    path = [x.copy()]
    for i in range(n_steps):
        t = np.full_like(x, 1.0 - i * dt)    # integrate from t = 1 (noise) to t = 0 (data)
        x = x - dt * net(x, t)               # Euler step
        path.append(x.copy())
    return (x, np.array(path)) if keep_path else x


def w1(a, b):
    """1-D Wasserstein-1 distance between equal-size samples: mean gap of sorted values."""
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def main():
    print("== Worked example: one training pair ==")
    x0, x1, t = 2.0, -0.5, 0.25
    xt = (1 - t) * x0 + t * x1
    v = x1 - x0
    print(f"x0 = {x0}, x1 = {x1}, t = {t}:  x_t = {xt:.3f}, target velocity = {v:.3f}")
    print(f"one Euler step back to t = 0: x_t - t * v = {xt - t * v:.3f} (recovers x0)")

    print("\n== Flow matching, independent (random) pairing ==")
    net = MLP()
    loss = train(net, lambda n: (sample_data(n), rng.standard_normal(n)))
    print(f"final training loss: {loss:.3f} (cannot reach 0: one x_t has many possible x1 - x0)")
    print(f"v(x=0, t=1) = {net(np.array([0.0]), np.array([1.0]))[0]:+.3f}"
          f"   v(x=1, t=1) = {net(np.array([1.0]), np.array([1.0]))[0]:+.3f}"
          "   (theory: v(x, 1) = x - E[x0] = x)")

    ref = sample_data(4000)
    z = rng.standard_normal(4000)
    results = {}
    print(f"{'steps':>5} | {'W1 to data':>10} | {'frac |x|>1':>10}")
    for k in [1, 2, 4, 8, 32]:
        xs = sample(net, z, k)
        results[k] = xs
        print(f"{k:>5} | {w1(xs, ref):>10.3f} | {np.mean(np.abs(xs) > 1):>10.2f}")

    print("\n== Reflow: retrain on (noise, sample) pairs produced by the first model ==")
    z_pairs = rng.standard_normal(20000)
    x_pairs = sample(net, z_pairs, 64)                    # deterministic ODE couples z -> x
    net2 = MLP()

    def coupled(n):
        idx = rng.integers(0, len(z_pairs), n)
        return x_pairs[idx], z_pairs[idx]

    train(net2, coupled)
    print(f"{'steps':>5} | {'W1 before':>9} | {'W1 after reflow':>15}")
    reflow = {}
    for k in [1, 2, 4]:
        reflow[k] = sample(net2, z, k)
        print(f"{k:>5} | {w1(results[k], ref):>9.3f} | {w1(reflow[k], ref):>15.3f}")

    # ---------------------------- figure ----------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    zz = rng.standard_normal(40)
    xx = sample_data(40)
    for a, b in zip(xx, zz):
        ax[0].plot([0, 1], [a, b], color=GRAY, lw=0.8, alpha=0.7)
    _, path = sample(net, np.linspace(-2.5, 2.5, 25), 64, keep_path=True)
    ts = np.linspace(1, 0, path.shape[0])
    for j in range(path.shape[1]):
        ax[0].plot(ts, path[:, j], color=BLUE, lw=1.2)
    ax[0].set(title="Training lines (gray), ODE paths (blue)", xlabel="t (0 = data, 1 = noise)",
              ylabel="x")

    bins = np.linspace(-4, 4, 60)
    ax[1].hist(ref, bins=bins, density=True, color=GRAY, alpha=0.4, label="data")
    for k, c in [(1, ORANGE), (4, PURPLE), (32, TEAL)]:
        ax[1].hist(results[k], bins=bins, density=True, histtype="step", lw=1.6, color=c, label=f"{k} step(s)")
    ax[1].set(title="Samples vs Euler steps", xlabel="x")
    ax[1].legend(fontsize=8)

    ax[2].hist(ref, bins=bins, density=True, color=GRAY, alpha=0.4, label="data")
    ax[2].hist(results[1], bins=bins, density=True, histtype="step", lw=1.6, color=ORANGE, label="1 step, before")
    ax[2].hist(reflow[1], bins=bins, density=True, histtype="step", lw=1.6, color=BLUE, label="1 step, after reflow")
    ax[2].set(title="After one reflow, 1 step works", xlabel="x")
    ax[2].legend(fontsize=8)
    save(fig, "flow-matching-rectified-flows")


if __name__ == "__main__":
    main()
