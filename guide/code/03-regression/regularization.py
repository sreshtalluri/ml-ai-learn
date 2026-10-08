"""Ridge, lasso, and elastic net coefficient paths; regression metrics with and without an outlier.

Run from guide/:  uv run code/03-regression/regularization.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import GRAY, PALETTE, plt, save  # noqa: E402

rng = np.random.default_rng(1)
n = 80
x1 = rng.normal(size=n)
x2 = x1 + rng.normal(0, 0.1, n)               # nearly a copy of x1 (multicollinearity)
x3 = rng.normal(size=n)
noise = rng.normal(size=(n, 3))                # three useless features
X = StandardScaler().fit_transform(np.c_[x1, x2, x3, noise])
y = 3 * x1 + 1.5 * x3 + rng.normal(0, 1, n)    # SYNTHETIC: only x1 and x3 matter
names = ["x1", "x2 (≈x1)", "x3", "noise1", "noise2", "noise3"]

ols = LinearRegression().fit(X, y)
print("OLS coefficients:   ", ols.coef_.round(2))
print("Ridge alpha=10:     ", Ridge(alpha=10).fit(X, y).coef_.round(2))
print("Lasso alpha=0.3:    ", Lasso(alpha=0.3).fit(X, y).coef_.round(2))

# Worked one-feature example matching the lesson: x=[1,2,3], y=[2,4,5]
xs, ys = np.array([1, 2, 3.0]), np.array([2, 4, 5.0])
sxy = np.mean((xs - xs.mean()) * (ys - ys.mean()))
sxx = np.mean((xs - xs.mean()) ** 2)
print(f"\nOne feature: sxy={sxy:.4f} sxx={sxx:.4f} OLS w={sxy / sxx:.4f}")
for lam in (0.5, 1, 2):
    print(f"  lambda={lam}: ridge w={sxy / (sxx + lam):.4f}  lasso w={max(abs(sxy) - lam / 2, 0) / sxx:.4f}")

# Metrics with an outlier
y_true = np.array([200, 250, 300, 350, 400.0])
y_pred = np.array([210, 240, 310, 340, 405.0])
y_pred_out = y_pred.copy(); y_pred_out[4] = 300
for label, yp in [("good predictions", y_pred), ("one big miss", y_pred_out)]:
    print(f"\n{label}: MAE={mean_absolute_error(y_true, yp):.1f} RMSE={mean_squared_error(y_true, yp) ** 0.5:.1f} "
          f"R2={r2_score(y_true, yp):.3f} MAPE={mean_absolute_percentage_error(y_true, yp) * 100:.1f}%")

alphas = np.geomspace(1e-3, 1e3, 60)
ridge_path = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
lasso_alphas = np.geomspace(1e-3, 3, 60)
lasso_path = np.array([Lasso(alpha=a, max_iter=20000).fit(X, y).coef_ for a in lasso_alphas])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for j, nm in enumerate(names):
    c = PALETTE[j] if j < 3 else GRAY
    axes[0].plot(alphas, ridge_path[:, j], color=c, lw=2 if j < 3 else 1, label=nm)
    axes[1].plot(lasso_alphas, lasso_path[:, j], color=c, lw=2 if j < 3 else 1, label=nm)
axes[0].set(xscale="log", title="Ridge (L2): weights shrink smoothly", xlabel="α (λ)", ylabel="coefficient")
axes[1].set(xscale="log", title="Lasso (L1): weights hit exactly zero", xlabel="α (λ)")
axes[0].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.axhline(0, color="black", lw=0.6)
save(fig, "regularization-and-regression-metrics")
