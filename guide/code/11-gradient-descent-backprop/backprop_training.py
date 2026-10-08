"""Backpropagation worked example, vanishing gradients, and training diagnostics (early stopping, L2, schedules, search).

Run from guide/:  uv run code/11-gradient-descent-backprop/backprop_training.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. The course's one-neuron example, with a finite-difference check
# ---------------------------------------------------------------------------
x1, x2, w1, w2, b, y, lr = 2.0, 1.0, 0.5, -1.0, 0.0, 1.0, 0.1


def loss(w1, w2, b):
    p = sigmoid(w1 * x1 + w2 * x2 + b)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


z = w1 * x1 + w2 * x2 + b
p = sigmoid(z)
dz = p - y
grads = np.array([dz * x1, dz * x2, dz])
h = 1e-6
numeric = np.array([(loss(w1 + h, w2, b) - loss(w1 - h, w2, b)) / (2 * h),
                    (loss(w1, w2 + h, b) - loss(w1, w2 - h, b)) / (2 * h),
                    (loss(w1, w2, b + h) - loss(w1, w2, b - h)) / (2 * h)])
print(f"z={z}, p={p}, L={loss(w1, w2, b):.4f}, dL/dz={dz}")
print(f"analytic grads {grads}, numeric {numeric.round(6)}")
nw = np.array([w1, w2, b]) - lr * grads
print(f"updated (w1, w2, b) = {nw}, new z = {nw[0] * x1 + nw[1] * x2 + nw[2]:.2f}, new p = {sigmoid(nw[0] * x1 + nw[1] * x2 + nw[2]):.4f}")

# ---------------------------------------------------------------------------
# 2. Vanishing gradients: product of local derivatives through depth
# ---------------------------------------------------------------------------
depths = np.arange(1, 31)
print(f"\nbest case sigmoid chain: 0.25^10 = {0.25 ** 10:.2e}, 0.25^20 = {0.25 ** 20:.2e}")
rng = np.random.default_rng(0)


def grad_norm_through(depth, act, width=64):
    """Backpropagate a unit gradient through `depth` random layers and report its norm."""
    hs, Ws = [rng.normal(size=width)], []
    for _ in range(depth):
        W = rng.normal(0, np.sqrt(1 / width) if act == "sigmoid" else np.sqrt(2 / width), (width, width))
        zz = W @ hs[-1]
        hs.append(sigmoid(zz) if act == "sigmoid" else np.maximum(0, zz))
        Ws.append((W, zz))
    g = np.ones(width)
    for W, zz in reversed(Ws):
        local = sigmoid(zz) * (1 - sigmoid(zz)) if act == "sigmoid" else (zz > 0).astype(float)
        g = W.T @ (g * local)
    return np.linalg.norm(g)


sig = [np.mean([grad_norm_through(d, "sigmoid") for _ in range(5)]) for d in depths]
rel = [np.mean([grad_norm_through(d, "relu") for _ in range(5)]) for d in depths]
print(f"gradient norm at the first layer, depth 20: sigmoid {sig[19]:.2e}, ReLU+He init {rel[19]:.2e}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(depths, sig, color=ORANGE, lw=2, label="sigmoid, Xavier-style init")
ax.plot(depths, rel, color=BLUE, lw=2, label="ReLU, He init")
ax.set(yscale="log", title="Gradient reaching the first layer vs network depth", xlabel="number of layers", ylabel="gradient norm (log)")
ax.legend(frameon=False, fontsize=8)
save(fig, "backpropagation")

# ---------------------------------------------------------------------------
# 3. Training diagnostics on a SYNTHETIC classification task
# ---------------------------------------------------------------------------
X, yy = make_classification(n_samples=1200, n_features=40, n_informative=8, flip_y=0.08, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(X, yy, test_size=0.4, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)


def curves(alpha, epochs=200):
    m = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=alpha, learning_rate_init=1e-3, random_state=0)
    tr, va = [], []
    for _ in range(epochs):
        m.partial_fit(Xtr, ytr, classes=[0, 1])
        p_tr = np.clip(m.predict_proba(Xtr)[:, 1], 1e-7, 1 - 1e-7)
        p_va = np.clip(m.predict_proba(Xva)[:, 1], 1e-7, 1 - 1e-7)
        tr.append(-np.mean(ytr * np.log(p_tr) + (1 - ytr) * np.log(1 - p_tr)))
        va.append(-np.mean(yva * np.log(p_va) + (1 - yva) * np.log(1 - p_va)))
    return np.array(tr), np.array(va)


tr0, va0 = curves(1e-5)
tr1, va1 = curves(1.0)
best = int(np.argmin(va0))
print(f"\nno regularization: best val loss {va0[best]:.3f} at epoch {best + 1}, final val loss {va0[-1]:.3f}")
print(f"L2 alpha=1.0: best val loss {va1.min():.3f}, final {va1[-1]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
ax = axes[0]
ax.plot(tr0, color=BLUE, label="train (no reg.)")
ax.plot(va0, color=ORANGE, label="validation (no reg.)")
ax.plot(tr1, color=BLUE, ls="--", lw=1, label="train (L2 α=1)")
ax.plot(va1, color=ORANGE, ls="--", lw=1, label="validation (L2 α=1)")
ax.axvline(best, color=GRAY, ls=":", lw=1.5)
ax.text(best + 3, va0[best] + 0.15, "early-stopping point", fontsize=8, color=GRAY)
ax.set(title="Training vs validation loss", xlabel="epoch", ylabel="log loss", ylim=(0, 1.5))
ax.legend(frameon=False, fontsize=7)

ax = axes[1]
steps = np.arange(1000)
warm = 50
cosine = np.where(steps < warm, steps / warm, 0.5 * (1 + np.cos(np.pi * (steps - warm) / (1000 - warm))))
stepd = 0.1 ** (steps // 300)
ax.plot(steps, np.ones_like(steps, dtype=float), color=GRAY, label="constant")
ax.plot(steps, stepd, color=PURPLE, label="step decay (×0.1 every 300)")
ax.plot(steps, cosine, color=TEAL, label="linear warmup + cosine decay")
ax.set(title="Learning-rate schedules (relative to peak)", xlabel="step", ylabel="learning rate / peak")
ax.legend(frameon=False, fontsize=7)

ax = axes[2]


def score(lr_, reg):   # SYNTHETIC validation surface: only the learning rate matters much
    return 0.9 - 0.15 * (np.log10(lr_) + 2.3) ** 2 - 0.01 * (np.log10(reg) + 2) ** 2


g = np.logspace(-4, -1, 3)
gr = np.logspace(-4, 0, 3)
G1, G2 = np.meshgrid(g, gr)
R1, R2 = 10 ** rng.uniform(-4, -1, 9), 10 ** rng.uniform(-4, 0, 9)
ax.scatter(G1, G2, color=BLUE, s=50, label=f"grid (9): best {score(G1, G2).max():.3f}")
ax.scatter(R1, R2, color=ORANGE, s=50, marker="^", label=f"random (9): best {score(R1, R2).max():.3f}")
ax.set(xscale="log", yscale="log", title="Grid vs random search, 9 trials each", xlabel="learning rate (matters)", ylabel="regularization (barely matters)")
ax.legend(frameon=False, fontsize=7)
print(f"search: grid tried {len(set(G1.ravel()))} distinct learning rates, random tried 9; best grid {score(G1, G2).max():.3f} vs random {score(R1, R2).max():.3f}")
for a in axes:
    a.title.set_fontsize(10)
save(fig, "training-and-regularization")
