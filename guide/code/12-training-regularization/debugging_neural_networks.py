# Adapted from AI Engineering from Scratch (MIT), phases/03-deep-learning-core/13-debugging-neural-networks
"""A debugging toolkit for a tiny NumPy MLP: initial-loss check, gradient check, overfit-one-batch,
per-layer dead-ReLU and gradient-norm stats, update/weight ratios, and an LR range test. Then four
deliberately broken runs and the diagnostic that catches each.

Data: SYNTHETIC 4-class Gaussian blobs in 10 dimensions, fixed seed.
Run from guide/:  uv run code/12-training-regularization/debugging_neural_networks.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

K, D, H = 4, 10, 32                       # classes, input features, hidden units


def make_data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.5, (K, D))
    y = rng.integers(0, K, n)
    X = centers[y] + rng.normal(0, 1.0, (n, D))   # SYNTHETIC blobs
    return X, y


# --- A two-hidden-layer MLP with explicit forward and backward --------------------------
def init(seed=0, w_scale=None, b_hidden=0.0, out_scale=0.1):
    rng = np.random.default_rng(seed)
    sizes = [D, H, H, K]
    params = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        s = w_scale if w_scale is not None else math.sqrt(2 / a)        # He init for ReLU layers
        if i == len(sizes) - 2 and w_scale is None:
            s *= out_scale                                              # small output layer: near-uniform start
        bias = np.full(b, b_hidden if i < 2 else 0.0)
        params.append([rng.normal(0, s, (a, b)), bias])
    return params


def forward(params, X):
    acts, h = [X], X
    for i, (W, b) in enumerate(params):
        z = h @ W + b
        h = np.maximum(z, 0) if i < len(params) - 1 else z            # last layer: raw logits
        acts.append(h)
    return acts


def loss_and_grads(params, X, y, relu_mask_bug=False):
    acts = forward(params, X)
    z = acts[-1]
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    loss = -logp[np.arange(len(y)), y].mean()
    d = np.exp(logp)
    d[np.arange(len(y)), y] -= 1
    d /= len(y)                                                         # dL/dlogits = (softmax - onehot) / n
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        W, _ = params[i]
        grads[i] = [acts[i].T @ d, d.sum(0)]
        if i > 0:
            d = d @ W.T
            if not relu_mask_bug:
                d = d * (acts[i] > 0)                                   # ReLU passes gradient only where active
    return loss, grads, acts


# --- Diagnostics --------------------------------------------------------------------
def grad_check(params, X, y, relu_mask_bug=False, h=1e-6):
    _, grads, _ = loss_and_grads(params, X, y, relu_mask_bug)
    worst = 0.0
    for (W, _), (gW, _) in zip(params, grads):
        for idx in [(0, 0), (1, 2), (3, 1)]:                            # spot-check a few entries per layer
            old = W[idx]
            W[idx] = old + h
            lp = loss_and_grads(params, X, y)[0]
            W[idx] = old - h
            lm = loss_and_grads(params, X, y)[0]
            W[idx] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - gW[idx]) / max(abs(num), abs(gW[idx]), 1e-12))
    return worst


def layer_stats(params, grads, acts, lr):
    rows = []
    for i, ((W, _), (gW, _)) in enumerate(zip(params, grads)):
        dead = (acts[i + 1] <= 0).all(0).mean() if i < len(params) - 1 else float("nan")
        rows.append((i + 1, dead, np.linalg.norm(gW), lr * np.linalg.norm(gW) / np.linalg.norm(W)))
    return rows


def train(params, X, y, lr=0.1, steps=300, batch=64, seed=0):
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps):
        idx = rng.integers(0, len(y), batch)
        loss, grads, _ = loss_and_grads(params, X[idx], y[idx])
        losses.append(loss)
        if not np.isfinite(loss):
            break
        for p, g in zip(params, grads):
            p[0] -= lr * g[0]
            p[1] -= lr * g[1]
    return losses


def accuracy(params, X, y):
    return (forward(params, X)[-1].argmax(1) == y).mean()


def lr_range_test(X, y, lo=1e-4, hi=10.0, steps=120):
    params = init()
    lrs = np.geomspace(lo, hi, steps)
    rng = np.random.default_rng(1)
    out = []
    for lr in lrs:
        idx = rng.integers(0, len(y), 64)
        loss, grads, _ = loss_and_grads(params, X[idx], y[idx])
        if not np.isfinite(loss) or loss > 4 * math.log(K):
            break
        out.append((lr, loss))
        for p, g in zip(params, grads):
            p[0] -= lr * g[0]
            p[1] -= lr * g[1]
    return np.array(out)


def main():
    np.seterr(all="ignore")
    X, y = make_data()
    Xtr, ytr, Xva, yva = X[:600], y[:600], X[600:], y[600:]

    print("=== 1. Initial loss (expect ln K) ===")
    print(f"ln {K} = {math.log(K):.4f}")
    for name, p in [("He, output x0.1", init()), ("He everywhere", init(out_scale=1.0)),
                    ("weights std 3.0", init(w_scale=3.0))]:
        print(f"{name:16s} step-0 loss = {loss_and_grads(p, Xtr, ytr)[0]:.4f}")

    print("\n=== 2. Gradient check (relative error, float64) ===")
    p = init()
    print(f"correct backward:       {grad_check(p, Xtr[:16], ytr[:16]):.2e}")
    print(f"ReLU mask forgotten:    {grad_check(p, Xtr[:16], ytr[:16], relu_mask_bug=True):.2e}")

    print("\n=== 3. Overfit one batch of 16 ===")
    p = init()
    xb, yb = Xtr[:16], ytr[:16]
    for step in range(301):
        loss, grads, _ = loss_and_grads(p, xb, yb)
        if step in (0, 100, 300):
            print(f"step {step:3d}: loss {loss:.4f}")
        for q, g in zip(p, grads):
            q[0] -= 0.1 * g[0]
            q[1] -= 0.1 * g[1]
    print(f"batch accuracy {accuracy(p, xb, yb):.2f}")

    print("\n=== 4. Four runs: one healthy, three broken ===")
    rng = np.random.default_rng(7)
    runs = {
        "healthy (lr 0.1)": (init(), ytr, 0.1),
        "lr too high (lr 30)": (init(), ytr, 30.0),
        "labels shuffled": (init(), rng.permutation(ytr), 0.1),
        "dead ReLUs (bias -3)": (init(b_hidden=-3.0), ytr, 0.1),
    }
    curves = {}
    for name, (p, labels, lr) in runs.items():
        curves[name] = train(p, Xtr, labels, lr=lr)
        last = curves[name][-1]
        print(f"{name:22s} final train loss {last:8.4f}  val acc {accuracy(p, Xva, yva):.3f}")
        runs[name] = (p, labels, lr)

    print("\n=== 5. Per-layer stats at step 0 ===")
    print("layer  dead-unit frac  ||grad W||   lr*||grad||/||W||")
    for name in ("healthy (lr 0.1)", "dead ReLUs (bias -3)"):
        p0 = init() if name.startswith("healthy") else init(b_hidden=-3.0)
        _, g0, a0 = loss_and_grads(p0, Xtr[:64], ytr[:64])
        print(name)
        for layer, dead, gn, ratio in layer_stats(p0, g0, a0, 0.1):
            print(f"  {layer}       {dead:6.2f}        {gn:.2e}     {ratio:.1e}")

    print("\n=== 6. LR range test ===")
    sweep = lr_range_test(Xtr, ytr)
    best = sweep[np.argmin(sweep[:, 1])]
    print(f"swept {len(sweep)} steps up to lr {sweep[-1, 0]:.2f}; lowest loss {best[1]:.4f} at lr {best[0]:.3f}")
    print(f"rule of thumb: pick about 10x below that -> {best[0] / 10:.3f}")

    # --- Figure ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    colors = [BLUE, ORANGE, PURPLE, TEAL]
    for (name, c), col in zip(curves.items(), colors):
        c = np.array(c)
        axes[0].plot(np.where(np.isfinite(c), np.minimum(c, 5), np.nan), color=col, label=name)
    axes[0].axhline(math.log(K), color=GRAY, ls="--", lw=1)
    axes[0].text(200, math.log(K) + 0.08, f"ln {K} = chance", fontsize=8, color=INK_TXT)
    axes[0].set(title="Loss curves: four runs", xlabel="step", ylabel="train loss (clipped at 5)", ylim=(0, 5))
    axes[0].legend(fontsize=8, loc="center right")

    axes[1].semilogx(sweep[:, 0], sweep[:, 1], color=BLUE)
    axes[1].axvline(best[0], color=ORANGE, ls="--", lw=1)
    axes[1].axvline(best[0] / 10, color=TEAL, ls="--", lw=1)
    axes[1].text(best[0] * 0.75, 3.0, "lowest", fontsize=8, color=ORANGE, ha="right")
    axes[1].text(best[0] / 10 * 1.1, 1.75, "÷10", fontsize=8, color=TEAL)
    axes[1].set(title="LR range test", xlabel="learning rate (log)", ylabel="mini-batch loss")

    labels_ = ["layer 1", "layer 2", "output"]
    for off, (name, col) in zip((-0.2, 0.2), (("healthy (lr 0.1)", BLUE), ("dead ReLUs (bias -3)", TEAL))):
        p0 = init() if name.startswith("healthy") else init(b_hidden=-3.0)
        _, g0, a0 = loss_and_grads(p0, Xtr[:64], ytr[:64])
        norms = [max(r[2], 1e-12) for r in layer_stats(p0, g0, a0, 0.1)]
        axes[2].bar(np.arange(3) + off, norms, 0.4, color=col, label=name)
    axes[2].set_yscale("log")
    axes[2].set_xticks(range(3), labels_)
    axes[2].set(title="Gradient norm per layer at step 0", ylabel="||dL/dW|| (log)")
    axes[2].legend(fontsize=8)
    save(fig, "debugging-neural-networks")


INK_TXT = "#52525b"

if __name__ == "__main__":
    main()
