# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/02-linear-regression, phases/01-math-foundations/17-linear-systems
"""Multiple linear regression two ways from scratch (normal equation, batch gradient descent), why
feature scaling matters for gradient descent, and the scikit-learn equivalent.
Data is SYNTHETIC with known true weights.

Run from guide/:  uv run code/03-regression/linear_regression_scratch.py
"""
import numpy as np
from sklearn.linear_model import LinearRegression


def add_bias(X):
    return np.c_[np.ones(len(X)), X]                       # [n, p] -> [n, p+1], column of 1s for b


def fit_normal_equation(X, y):
    """Solve (XᵀX) θ = Xᵀy. np.linalg.solve is stabler than forming the inverse."""
    Xb = add_bias(X)
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)             # θ = [b, w1, ..., wp]


def fit_gradient_descent(X, y, lr=0.1, steps=1000):
    Xb = add_bias(X)
    theta = np.zeros(Xb.shape[1])
    n = len(y)
    for _ in range(steps):
        error = Xb @ theta - y                              # [n]
        grad = (2 / n) * Xb.T @ error                       # [p+1], dMSE/dθ
        theta -= lr * grad
    return theta


def r2(y, y_hat):
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    sqft = rng.uniform(0.5, 4.0, n)                         # thousands of sq ft
    beds = rng.integers(1, 6, n).astype(float)
    age = rng.uniform(0, 80, n)                             # years: a much larger scale
    X = np.c_[sqft, beds, age]
    y = 50 + 120 * sqft + 15 * beds - 0.8 * age + rng.normal(0, 20, n)   # true θ = [50, 120, 15, -0.8]

    theta_ne = fit_normal_equation(X, y)
    print(f"normal equation θ = {theta_ne.round(3)},  R² = {r2(y, add_bias(X) @ theta_ne):.4f}")

    # Raw features: lr must be tiny because 'age' has a large scale, so convergence crawls
    theta_raw = fit_gradient_descent(X, y, lr=1e-4, steps=1000)
    print(f"GD raw features, lr=1e-4, 1000 steps: θ = {theta_raw.round(3)}")
    cond_raw = np.linalg.cond(add_bias(X).T @ add_bias(X))

    # Standardized features: the same 1000 steps converge
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    theta_z = fit_gradient_descent(Z, y, lr=0.1, steps=1000)
    w = theta_z[1:] / sd                                    # undo the scaling to compare
    b = theta_z[0] - np.sum(theta_z[1:] * mu / sd)
    print(f"GD standardized, lr=0.1, 1000 steps: θ = {np.r_[b, w].round(3)}")
    cond_z = np.linalg.cond(add_bias(Z).T @ add_bias(Z))
    print(f"condition number of XᵀX: raw {cond_raw:,.0f}, standardized {cond_z:.1f}")

    sk = LinearRegression().fit(X, y)
    print(f"scikit-learn: b = {sk.intercept_:.3f}, w = {sk.coef_.round(3)}")

    assert np.allclose(np.r_[sk.intercept_, sk.coef_], theta_ne, atol=1e-6)
    assert np.allclose(np.r_[b, w], theta_ne, atol=1e-3)
    assert not np.allclose(theta_raw, theta_ne, atol=1.0)
    print("checks pass: normal equation = sklearn = standardized GD")
