# Adapted from AI Engineering from Scratch (MIT), phases/01-math-foundations/04-calculus-for-ml
"""Numerical derivatives, a gradient check, the chain rule, and gradient descent, from scratch.

Run from guide/:  uv run code/00-foundations/derivatives_scratch.py
"""
import numpy as np


def derivative(f, x, h=1e-5):
    """Central difference: slope of the chord from x-h to x+h."""
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient(f, w, h=1e-5):
    """One central difference per coordinate: a vector of partial derivatives."""
    w = np.asarray(w, dtype=float)
    g = np.zeros_like(w)
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)
    return g


def L(p):
    return p[0] ** 2 + 2 * p[1] ** 2


def grad_L(p):
    return np.array([2 * p[0], 4 * p[1]])            # analytic gradient


def descend(grad, w0, lr=0.1, steps=50):
    w = np.asarray(w0, dtype=float)
    for _ in range(steps):
        w = w - lr * grad(w)                         # step against the gradient
    return w


if __name__ == "__main__":
    f = lambda t: t**2 - 4 * t + 5                   # noqa: E731
    print(f"f'(3): numerical {derivative(f, 3.0):.6f}, analytic {2 * 3 - 4}")

    w = np.array([1.0, 1.0])
    num, ana = gradient(L, w), grad_L(w)
    rel_err = np.linalg.norm(num - ana) / np.linalg.norm(num + ana)
    print(f"∇L(1, 1): numerical {num.round(6)}, analytic {ana}, relative error {rel_err:.1e}")

    # Chain rule (worked example 3): L(w) = (y - w x)^2 with x = 2, y = 7. dL/dw = 2 (w x - y) x
    x, y, w0 = 2.0, 7.0, 1.0
    chain = 2 * (w0 * x - y) * x
    print(f"chain rule dL/dw at w=1: {chain}, numerical {derivative(lambda v: (y - v * x) ** 2, w0):.6f}")
    print(f"one step, lr=0.1: w = {w0 - 0.1 * chain}, new prediction {(w0 - 0.1 * chain) * x}")

    for lr in (0.1, 0.6):
        print(f"gradient descent lr={lr}: from (1, 1) to {descend(grad_L, [1, 1], lr=lr).round(4)}")

    assert rel_err < 1e-8
    assert abs(chain - derivative(lambda v: (y - v * x) ** 2, w0)) < 1e-6
    assert np.allclose(descend(grad_L, [1, 1], lr=0.1), 0, atol=1e-3)
    print("all checks pass")
