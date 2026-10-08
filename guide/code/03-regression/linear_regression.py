"""Linear regression: worked example, from-scratch gradient descent, scikit-learn, and figures.

Run from guide/:  uv run code/03-regression/linear_regression.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402

# ---------------------------------------------------------------------------
# 1. The worked example from the lesson (every step printed)
# ---------------------------------------------------------------------------
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 5.0])
w, b, lr = 1.0, 0.0, 0.1
n = len(x)

y_hat = w * x + b
residuals = y - y_hat
squared = residuals**2
mse = squared.mean()
dw = (2 / n) * np.sum((y_hat - y) * x)
db = (2 / n) * np.sum(y_hat - y)
w_new, b_new = w - lr * dw, b - lr * db
mse_new = np.mean((y - (w_new * x + b_new)) ** 2)

print("Worked example")
print(f"  predictions      {y_hat}")
print(f"  residuals y-yhat {residuals}")
print(f"  squared errors   {squared}")
print(f"  MSE              {mse:.4f}")
print(f"  dMSE/dw          {dw:.4f}")
print(f"  dMSE/db          {db:.4f}")
print(f"  updated w, b     {w_new:.4f}, {b_new:.4f}")
print(f"  new MSE          {mse_new:.4f}")


# ---------------------------------------------------------------------------
# 2. From scratch: batch gradient descent on a synthetic dataset
# ---------------------------------------------------------------------------
def fit_gd(x, y, lr=0.05, steps=500):
    w, b = 0.0, 0.0
    history = []
    for _ in range(steps):
        y_hat = w * x + b
        error = y_hat - y
        w -= lr * (2 / len(x)) * np.sum(error * x)
        b -= lr * (2 / len(x)) * np.sum(error)
        history.append(np.mean(error**2))
    return w, b, history


rng = np.random.default_rng(42)
X = rng.uniform(0, 10, 40)                      # SYNTHETIC: house size (hundreds of sq ft)
Y = 3.0 * X + 8 + rng.normal(0, 3, 40)          # SYNTHETIC: price, true slope 3, bias 8

w_gd, b_gd, history = fit_gd(X, Y, lr=0.01, steps=2000)
sk = LinearRegression().fit(X.reshape(-1, 1), Y)
print("\nSynthetic dataset (true w=3, b=8)")
print(f"  gradient descent  w={w_gd:.3f}  b={b_gd:.3f}")
print(f"  scikit-learn      w={sk.coef_[0]:.3f}  b={sk.intercept_:.3f}")

# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
grid = np.linspace(0, 10, 50)
pred = w_gd * X + b_gd
for xi, yi, pi in zip(X, Y, pred):
    ax.plot([xi, xi], [yi, pi], color=ORANGE, lw=1, alpha=0.7)
ax.scatter(X, Y, color=BLUE, s=22, zorder=3, label="observations (synthetic)")
ax.plot(grid, w_gd * grid + b_gd, color=TEAL, lw=2, label=f"fit: ŷ = {w_gd:.2f}x + {b_gd:.2f}")
ax.plot([], [], color=ORANGE, lw=1, label="residuals")
ax.set(title="Residuals are vertical gaps", xlabel="x (feature)", ylabel="y (target)")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
ax.plot(history, color=BLUE, lw=2)
ax.set(title="Gradient descent lowers MSE", xlabel="step", ylabel="MSE", yscale="log")
ax.axhline(np.mean((Y - sk.predict(X.reshape(-1, 1))) ** 2), color=GRAY, ls="--", lw=1)
ax.text(len(history) * 0.55, np.mean((Y - sk.predict(X.reshape(-1, 1))) ** 2) * 1.15,
        "closed-form optimum", color=GRAY, fontsize=8)

save(fig, "linear-regression")
