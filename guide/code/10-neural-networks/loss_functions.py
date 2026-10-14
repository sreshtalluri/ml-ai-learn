# Adapted from AI Engineering from Scratch (MIT), phases/03-deep-learning-core/05-loss-functions
"""Loss functions from scratch: MSE, MAE, Huber, binary and categorical cross-entropy (with their
gradients), label smoothing, focal loss, and InfoNCE. Then MSE vs cross-entropy training the same
small network on a SYNTHETIC circle dataset.

Run from guide/:  uv run code/10-neural-networks/loss_functions.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402


# --- Regression losses ------------------------------------------------------------
def mse(pred, y):
    return np.mean((pred - y) ** 2)


def mae(pred, y):
    return np.mean(np.abs(pred - y))


def huber(pred, y, delta=1.0):
    r = np.abs(pred - y)
    return np.mean(np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta)))   # quadratic near 0, linear far out


# --- Binary cross-entropy, from logits ------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce_with_logits(z, y):
    # -[y log s(z) + (1-y) log(1-s(z))], rewritten so exp only sees non-positive numbers
    return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))


def grad_wrt_logit(z, y, loss):
    """dLoss/dz for one example with a sigmoid output p = s(z)."""
    p = sigmoid(z)
    if loss == "bce":
        return p - y                              # the sigmoid's slope cancels out
    return 2 * (p - y) * p * (1 - p)              # MSE on p: chain rule leaves p(1-p) in


# --- Softmax cross-entropy and friends ------------------------------------------------
def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def cross_entropy(logits, target, smoothing=0.0):
    """Mean CE over a batch. logits [n, K], target [n] ints. smoothing spreads eps over all K classes."""
    n, k = logits.shape
    q = np.full((n, k), smoothing / k)
    q[np.arange(n), target] += 1 - smoothing
    return -(q * log_softmax(logits)).sum(axis=1).mean()


def ce_grad(logits, target):
    """dCE/dlogits for one example = softmax(z) - onehot(y)."""
    g = np.exp(log_softmax(logits))
    g[target] -= 1
    return g


def focal_loss(p_true, gamma=2.0, alpha=1.0):
    return -alpha * (1 - p_true) ** gamma * np.log(p_true)       # (1-p)^gamma down-weights easy examples


def info_nce(anchor, positive, negatives, tau=0.1):
    """Cross-entropy where the 'correct class' is the positive among [positive, *negatives]."""
    cands = np.vstack([positive, negatives])
    sims = cands @ anchor / (np.linalg.norm(cands, axis=1) * np.linalg.norm(anchor))   # cosine similarity
    return -log_softmax(sims / tau)[0]


# --- Train the same network with MSE vs BCE ----------------------------------------------
def circle_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = ((X**2).sum(axis=1) < 1.5).astype(float)                # SYNTHETIC: inside a circle = 1
    return X, y


def train(X, y, loss, epochs=300, lr=0.5, hidden=16, seed=1):
    rng = np.random.default_rng(seed)
    W1, b1 = rng.normal(0, 1, (2, hidden)), np.zeros(hidden)
    W2, b2 = rng.normal(0, 1, hidden), 0.0
    W2[:] *= 2.0                                                   # a deliberately overconfident start
    acc = []
    for _ in range(epochs):
        H = np.maximum(X @ W1 + b1, 0)                             # [n, hidden]
        z = H @ W2 + b2                                            # [n]
        dz = grad_wrt_logit(z, y, loss) / len(y)                   # the only line that differs
        dH = np.outer(dz, W2) * (H > 0)
        W2 -= lr * H.T @ dz
        b2 -= lr * dz.sum()
        W1 -= lr * X.T @ dH
        b1 -= lr * dH.sum(axis=0)
        acc.append(((z > 0) == (y > 0.5)).mean())
    return acc


def main():
    print("=== Regression: one outlier (synthetic) ===")
    y = np.array([3.0, 5.0, 4.0, 6.0])
    pred = np.array([2.5, 5.5, 4.0, 16.0])                      # last prediction is off by 10
    print(f"residuals {pred - y}")
    print(f"MSE = {mse(pred, y):.4f}   MAE = {mae(pred, y):.4f}   Huber(delta=1) = {huber(pred, y):.4f}")

    print("\n=== BCE vs MSE for true label 1 ===")
    for p in (0.99, 0.9, 0.5, 0.1, 0.01):
        print(f"p={p:<5} BCE={-math.log(p):.4f}  MSE={(1 - p) ** 2:.4f}")

    print("\n=== Gradient w.r.t. the logit, true label 1 ===")
    for z in (-5.0, 0.0, 5.0):
        print(f"z={z:+.0f}  p={sigmoid(z):.4f}  dBCE/dz={grad_wrt_logit(z, 1, 'bce'):+.4f}  dMSE/dz={grad_wrt_logit(z, 1, 'mse'):+.4f}")
    zs = np.array([-5.0, 0.0, 5.0])
    assert abs(bce_with_logits(zs, np.ones(3)) - np.mean(-np.log(sigmoid(zs)))) < 1e-12

    print("\n=== Softmax cross-entropy ===")
    logits = np.array([2.0, 1.0, 0.1])
    print(f"logits {logits} -> softmax {np.round(np.exp(log_softmax(logits)), 4)}")
    print(f"CE (true class 0) = {cross_entropy(logits[None], np.array([0])):.4f}")
    print(f"grad = softmax - onehot = {np.round(ce_grad(logits, 0), 4)}")
    # finite-difference check of the gradient
    h, num = 1e-6, np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        num[i] = (cross_entropy((logits + e)[None], np.array([0])) - cross_entropy((logits - e)[None], np.array([0]))) / (2 * h)
    print(f"finite-difference check max error = {np.abs(num - ce_grad(logits, 0)).max():.1e}")
    print(f"label smoothing eps=0.1: CE = {cross_entropy(logits[None], np.array([0]), 0.1):.4f}")

    print("\n=== Focal loss (gamma=2) vs CE ===")
    for p in (0.9, 0.5, 0.1):
        print(f"p_true={p}: CE={-math.log(p):.4f}  focal={focal_loss(p):.4f}  weight (1-p)^2={(1 - p) ** 2:.2f}")

    print("\n=== InfoNCE (synthetic 8-d embeddings, 7 negatives) ===")
    rng = np.random.default_rng(42)
    anchor = rng.normal(size=8)
    positive = anchor + rng.normal(0, 0.1, 8)
    negatives = rng.normal(size=(7, 8))
    for tau in (0.07, 0.1, 0.5):
        print(f"tau={tau}: loss={info_nce(anchor, positive, negatives, tau):.4f}")
    print(f"uniform guess over 8 candidates would be ln 8 = {math.log(8):.4f}")

    print("\n=== Same network, same init, MSE vs BCE (synthetic circle data) ===")
    X, yc = circle_data()
    curves = {name: train(X, yc, name) for name in ("mse", "bce")}
    for name, acc in curves.items():
        print(f"{name}: accuracy at epoch 10={acc[9]:.3f}, 50={acc[49]:.3f}, 300={acc[-1]:.3f}")

    # --- Figure ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    p = np.linspace(0.005, 1, 400)
    axes[0].plot(p, -np.log(p), color=BLUE, label="cross-entropy  −ln p")
    axes[0].plot(p, (1 - p) ** 2, color=ORANGE, label="squared error  (1−p)²")
    axes[0].set(title="Loss when the true label is 1", xlabel="predicted p", ylabel="loss", ylim=(0, 5))
    axes[0].legend()

    z = np.linspace(-8, 8, 400)
    axes[1].plot(z, -grad_wrt_logit(z, 1, "bce"), color=BLUE, label="cross-entropy: 1 − p")
    axes[1].plot(z, -grad_wrt_logit(z, 1, "mse"), color=ORANGE, label="MSE: 2(1−p)·p(1−p)")
    axes[1].axvspan(-8, -3, color=GRAY, alpha=0.15)
    axes[1].text(-7.7, 0.55, "confidently\nwrong", fontsize=8)
    axes[1].set(title="Push on the logit (−dL/dz), label 1", xlabel="logit z", ylabel="gradient size")
    axes[1].legend(loc="upper right", fontsize=8)

    for name, color in (("bce", BLUE), ("mse", ORANGE)):
        axes[2].plot(curves[name], color=color, label="cross-entropy" if name == "bce" else "MSE")
    axes[2].set(title="Same net, same init: training accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0.4, 1.0))
    axes[2].legend(loc="lower right")
    save(fig, "loss-functions")


if __name__ == "__main__":
    main()
