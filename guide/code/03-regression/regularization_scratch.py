"""Ridge (closed form), lasso (coordinate descent with soft-thresholding) and the four regression metrics,
from scratch in NumPy, checked against scikit-learn. Data is SYNTHETIC.

Objective here matches the lesson: (1/n) * sum (y - Xw - b)^2 + lam * penalty. scikit-learn uses
(1/2n) for Lasso and no 1/n for Ridge, so the script converts lam before comparing.

Run from guide/:  uv run code/03-regression/regularization_scratch.py
"""
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def ridge(X, y, lam):
    """Center, then solve (X^T X / n + lam I) w = X^T y / n. The bias is the mean and is not penalized."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    n, d = X.shape
    w = np.linalg.solve(Xc.T @ Xc / n + lam * np.eye(d), Xc.T @ yc / n)
    return w, ym - xm @ w


def soft_threshold(z, g):
    """Shrink z toward 0 by g; anything inside [-g, g] becomes exactly 0."""
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def lasso(X, y, lam, sweeps=5000):
    """Coordinate descent: solve for one weight at a time with the others fixed."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    n, d = X.shape
    w = np.zeros(d)
    sxx = (Xc**2).mean(axis=0)                     # s_xx for each column
    for _ in range(sweeps):
        for j in range(d):
            r = yc - Xc @ w + Xc[:, j] * w[j]      # residual ignoring feature j
            sxy = Xc[:, j] @ r / n                 # s_xy between feature j and that residual
            w[j] = soft_threshold(sxy, lam / 2) / sxx[j]
    return w, ym - xm @ w


def metrics(y, yhat):
    e = y - yhat
    return {"MAE": np.mean(np.abs(e)),
            "RMSE": np.sqrt(np.mean(e**2)),
            "R2": 1 - np.sum(e**2) / np.sum((y - y.mean()) ** 2),
            "MAPE%": 100 * np.mean(np.abs(e / y))}


if __name__ == "__main__":
    # Worked example from the lesson: x = [1, 2, 3], y = [2, 4, 5].
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 5.0])
    for lam in (0.5, 1, 2):
        print(f"lambda={lam}: ridge w={ridge(x, y, lam)[0][0]:.4f}  lasso w={lasso(x, y, lam)[0][0]:.4f}")

    # SYNTHETIC: 6 standardized features, only x1 and x3 matter, x2 is a near-copy of x1.
    rng = np.random.default_rng(1)
    n = 80
    x1 = rng.normal(size=n)
    X = np.c_[x1, x1 + rng.normal(0, 0.1, n), rng.normal(size=(n, 4))]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y_syn = 3 * X[:, 0] + 1.5 * X[:, 2] + rng.normal(0, 1, n)

    w_r, _ = ridge(X, y_syn, 0.1)
    w_l, _ = lasso(X, y_syn, 0.6)
    sk_r = Ridge(alpha=0.1 * n).fit(X, y_syn).coef_          # sklearn Ridge: no 1/n on the loss
    sk_l = Lasso(alpha=0.6 / 2, tol=1e-10, max_iter=100_000).fit(X, y_syn).coef_  # sklearn Lasso: 1/(2n) on the loss
    print(f"\nscratch ridge lam=0.1: {w_r.round(2)}")
    print(f"sklearn Ridge(alpha=8):  {sk_r.round(2)}")
    print(f"scratch lasso lam=0.6: {w_l.round(2)}")
    print(f"sklearn Lasso(alpha=0.3): {sk_l.round(2)}")

    y_true = np.array([200, 250, 300, 350, 400.0])
    y_pred = np.array([210, 240, 310, 340, 405.0])
    y_miss = y_pred.copy()
    y_miss[4] = 300
    for label, yp in [("good predictions", y_pred), ("one big miss", y_miss)]:
        m = metrics(y_true, yp)
        print(f"\n{label}: " + " ".join(f"{k}={v:.3f}" if k == "R2" else f"{k}={v:.1f}" for k, v in m.items()))

    # Checks against the library and the hand numbers.
    assert np.allclose(w_r, sk_r, atol=1e-6) and np.allclose(w_l, sk_l, atol=1e-4)
    assert abs(lasso(x, y, 2)[0][0]) < 1e-12 and abs(ridge(x, y, 1)[0][0] - 0.6) < 1e-12
    m = metrics(y_true, y_miss)
    assert np.isclose(m["MAE"], mean_absolute_error(y_true, y_miss))
    assert np.isclose(m["RMSE"], mean_squared_error(y_true, y_miss) ** 0.5)
    assert np.isclose(m["R2"], r2_score(y_true, y_miss))
    assert np.isclose(m["MAPE%"], 100 * mean_absolute_percentage_error(y_true, y_miss))
