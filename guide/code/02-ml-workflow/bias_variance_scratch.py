# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/10-bias-variance
"""Measure bias² and variance directly: redraw the training set many times, refit, and look at how
the predictions spread. Data is SYNTHETIC (a known true function plus Gaussian noise).

Run from guide/:  uv run code/02-ml-workflow/bias_variance_scratch.py
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

NOISE = 0.5


def true_f(x):
    return np.sin(1.5 * x) + 0.5 * x


def sample(n, rng):
    x = rng.uniform(-3, 3, n)
    return x, true_f(x) + rng.normal(0, NOISE, n)


def fit_poly(x, y, degree, lam=0.0):
    """Least squares on [1, x, x², ...]. lam > 0 adds an L2 penalty (not on the intercept)."""
    X = np.vander(x, degree + 1, increasing=True)          # [n, degree+1]
    P = lam * np.eye(degree + 1)
    P[0, 0] = 0
    return np.linalg.solve(X.T @ X + P + 1e-10 * np.eye(degree + 1), X.T @ y)


def predict_poly(x, w):
    return np.vander(x, len(w), increasing=True) @ w


def decompose(degree, lam=0.0, n_sets=200, n_train=50, seed=42):
    rng = np.random.default_rng(seed)
    x_test = np.linspace(-2.5, 2.5, 100)
    preds = np.array([predict_poly(x_test, fit_poly(*sample(n_train, rng), degree, lam))
                      for _ in range(n_sets)])              # [n_sets, 100]
    avg = preds.mean(axis=0)
    bias2 = np.mean((avg - true_f(x_test)) ** 2)            # how far the AVERAGE model is from the truth
    var = np.mean(preds.var(axis=0))                        # how much models disagree with each other
    return bias2, var


if __name__ == "__main__":
    print(f"irreducible noise σ² = {NOISE**2:.3f}")
    print(f"{'degree':>6} {'bias²':>8} {'variance':>9} {'expected MSE':>13}")
    for d in (1, 3, 5, 8, 10):
        b2, v = decompose(d)
        print(f"{d:>6} {b2:>8.4f} {v:>9.4f} {b2 + v + NOISE**2:>13.4f}")

    print("degree 10 with an L2 penalty:")
    for lam in (0.0, 0.1, 10.0):
        b2, v = decompose(10, lam=lam)
        print(f"  λ={lam:<5} bias² {b2:.4f}  variance {v:.4f}")

    rng = np.random.default_rng(0)
    x, y = sample(60, rng)
    model = make_pipeline(PolynomialFeatures(9), StandardScaler(), Ridge())
    alphas = [1e-4, 1e-2, 1, 100]
    _, va = validation_curve(model, x[:, None], y, param_name="ridge__alpha", param_range=alphas,
                             cv=5, scoring="neg_mean_squared_error")
    for a, s in zip(alphas, -va.mean(axis=1)):
        print(f"sklearn validation_curve: alpha={a:<7} CV MSE {s:.3f}")

    b1, v1 = decompose(1)
    b10, v10 = decompose(10)
    assert b1 > b10 and v10 > v1
    print("checks pass: low degree = high bias, high degree = high variance")
