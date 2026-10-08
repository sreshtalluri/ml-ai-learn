"""Forward pass of a small MLP by hand and in NumPy, with tensor shapes; activations and the XOR figure.

Run from guide/:  uv run code/10-neural-networks/forward_pass.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. Worked example: 2 inputs -> 3 hidden (ReLU) -> 1 output (sigmoid)
# ---------------------------------------------------------------------------
x = np.array([1.0, 2.0])                                   # [2]
W1 = np.array([[0.5, -1.0], [1.0, 1.0], [-0.5, 0.25]])     # [3, 2]  (hidden, inputs)
b1 = np.array([0.0, -1.0, 0.5])                            # [3]
W2 = np.array([[1.0, -0.5, 2.0]])                          # [1, 3]  (outputs, hidden)
b2 = np.array([0.25])                                      # [1]

z1 = W1 @ x + b1
a1 = relu(z1)
z2 = W2 @ a1 + b2
y_hat = sigmoid(z2)
print("Single example")
print(f"  z1 = W1 x + b1 = {z1}")
print(f"  a1 = ReLU(z1)  = {a1}")
print(f"  z2 = W2 a1 + b2 = {z2}")
print(f"  y_hat = sigmoid(z2) = {y_hat.round(4)}")
print(f"  parameters: W1 {W1.size} + b1 {b1.size} + W2 {W2.size} + b2 {b2.size} = {W1.size + b1.size + W2.size + b2.size}")

# Batched: rows are examples. X [batch, in] @ W1.T [in, hidden] -> [batch, hidden]
X = np.array([[1.0, 2.0], [0.0, 0.0], [-1.0, 1.0], [2.0, -1.0]])
H = relu(X @ W1.T + b1)
Y = sigmoid(H @ W2.T + b2)
print(f"\nBatch of {len(X)}: X {X.shape} -> H {H.shape} -> Y {Y.shape}")
print(f"  Y = {Y.ravel().round(4)}")

# Two linear layers with no activation collapse into one linear layer:
Wc, bc = W2 @ W1, W2 @ b1 + b2
print(f"\nWithout ReLU: W2(W1x + b1) + b2 = {(W2 @ (W1 @ x + b1) + b2)} == Wc x + bc = {Wc @ x + bc}")

# ---------------------------------------------------------------------------
# 2. Figure: activation functions, and XOR (linear model vs MLP)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
z = np.linspace(-4, 4, 400)
ax = axes[0]
from math import erf  # noqa: E402
gelu = np.array([0.5 * v * (1 + erf(v / np.sqrt(2))) for v in z])
for name, val, c in [("ReLU", relu(z), BLUE), ("GELU", gelu, PURPLE), ("sigmoid", sigmoid(z), ORANGE), ("tanh", np.tanh(z), TEAL)]:
    ax.plot(z, val, color=c, lw=2, label=name)
ax.set(title="Activation functions", xlabel="z", ylabel="f(z)", ylim=(-1.3, 4))
ax.legend(frameon=False, fontsize=8)

rng = np.random.default_rng(0)
Xx = rng.uniform(-1, 1, (400, 2))                          # SYNTHETIC XOR: label 1 when signs differ
yx = (Xx[:, 0] * Xx[:, 1] < 0).astype(int)
gx, gy = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
grid = np.c_[gx.ravel(), gy.ravel()]
for ax, model, title in [
    (axes[1], LogisticRegression(), "Linear model on XOR"),
    (axes[2], MLPClassifier(hidden_layer_sizes=(8,), activation="relu", max_iter=3000, random_state=1), "MLP, one hidden ReLU layer"),
]:
    model.fit(Xx, yx)
    acc = model.score(Xx, yx)
    print(f"{title}: training accuracy {acc:.2f}")
    ax.contourf(gx, gy, model.predict(grid).reshape(gx.shape), levels=[-0.5, 0.5, 1.5], colors=[BLUE, ORANGE], alpha=0.15)
    ax.scatter(*Xx[yx == 0].T, s=8, color=BLUE)
    ax.scatter(*Xx[yx == 1].T, s=8, color=ORANGE, marker="s")
    ax.set(title=f"{title} (acc {acc:.2f})", xlabel="x₁")
    ax.title.set_fontsize(10)
axes[1].set_ylabel("x₂")
save(fig, "neural-network-forward-pass")
