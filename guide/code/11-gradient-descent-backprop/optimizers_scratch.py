# Adapted from AI Engineering from Scratch (MIT), phases/01-math-foundations/08-optimization, phases/03-deep-learning-core/06-optimizers
"""Four optimizers as small classes with the same step(params, grads) interface that PyTorch
uses: SGD, SGD with momentum, Adam, and AdamW. They reproduce the course's worked steps on the
narrow bowl, then race on the Rosenbrock valley.

Run from guide/:  uv run code/11-gradient-descent-backprop/optimizers_scratch.py
"""
import numpy as np


# --- 1. Plain gradient descent: one line of state-free arithmetic ----------------
class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, p, g):
        return p - self.lr * g


# --- 2. Momentum: keep a running sum of past gradients (the velocity) ------------
class Momentum:
    def __init__(self, lr, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, p, g):
        self.v = g if self.v is None else self.beta * self.v + g
        return p - self.lr * self.v


# --- 3. Adam: momentum on the gradient, a per-parameter scale from its square ----
class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, weight_decay
        self.m = self.s = None
        self.t = 0

    def step(self, p, g):
        if self.m is None:
            self.m, self.s = np.zeros_like(p), np.zeros_like(p)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g          # average gradient
        self.s = self.b2 * self.s + (1 - self.b2) * g ** 2     # average squared gradient
        m_hat = self.m / (1 - self.b1 ** self.t)               # undo the bias toward 0
        s_hat = self.s / (1 - self.b2 ** self.t)
        return p - self.lr * (m_hat / (np.sqrt(s_hat) + self.eps) + self.wd * p)


# --- 4. AdamW is Adam with weight decay applied to the weights, not the gradient --
def AdamW(lr, weight_decay=0.01, **kw):
    return Adam(lr, weight_decay=weight_decay, **kw)


def run(opt, grad, start, steps):
    p, path = np.array(start, float), [np.array(start, float)]
    for _ in range(steps):
        p = opt.step(p, grad(p))
        path.append(p.copy())
    return np.array(path)


def bowl(p):
    return 0.5 * (p[0] ** 2 + 10 * p[1] ** 2)


def bowl_grad(p):
    return np.array([p[0], 10 * p[1]])


def rosenbrock(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


def rosenbrock_grad(p):
    return np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    start = (-3.0, 1.6)
    print("== Course worked example: f = 0.5 (t1^2 + 10 t2^2), start (-3, 1.6), f = 17.3 ==")
    for name, opt in [("GD lr=0.1", SGD(0.1)), ("momentum lr=0.02", Momentum(0.02)),
                      ("Adam lr=0.1", Adam(0.1))]:
        path = run(opt, bowl_grad, start, 2)
        print(f"{name:>17}: step 1 {path[1]}, step 2 {path[2]}, f after 2 = {bowl(path[2]):.4f}")

    print("\n== Learning rate on the bowl (steep direction curvature 10, so lr must be < 0.2) ==")
    for lr in (0.05, 0.19, 0.21):
        path = run(SGD(lr), bowl_grad, start, 50)
        print(f"lr={lr}: f after 50 steps = {bowl(path[-1]):.3g}")

    print("\n== Adam bias correction: first step with and without it (g = 0.5) ==")
    m1, s1 = 0.1 * 0.5, 0.001 * 0.25
    print(f"raw m/sqrt(s) = {m1 / np.sqrt(s1):.4f}; corrected = {(m1 / 0.1) / np.sqrt(s1 / 0.001):.4f}")

    print("\n== Rosenbrock valley, start (-1.5, 2), minimum at (1, 1), 3,000 steps ==")
    for name, opt in [("SGD", SGD(1e-4)), ("momentum", Momentum(1e-4)), ("Adam", Adam(0.02))]:
        path = run(opt, rosenbrock_grad, (-1.5, 2.0), 3000)
        print(f"{name:>8}: end {path[-1]}, loss {rosenbrock(path[-1]):.2e}")

    print("\n== AdamW decay on a parameter with zero gradient ==")
    opt = AdamW(0.1, weight_decay=0.1)
    path = run(opt, lambda p: np.zeros_like(p), (1.0,), 10)
    print(f"after 10 steps: {path[-1][0]:.4f} (= 0.99^10 = {0.99 ** 10:.4f})")
