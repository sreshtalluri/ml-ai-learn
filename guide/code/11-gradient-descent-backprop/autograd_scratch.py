# Adapted from AI Engineering from Scratch (MIT), phases/01-math-foundations/05-chain-rule-and-autodiff, phases/03-deep-learning-core/03-backpropagation, phases/03-deep-learning-core/10-mini-framework
"""A scalar reverse-mode autodiff engine in about 60 lines. Every operation records its inputs and
a local derivative rule; backward() visits the graph in reverse topological order and applies
the chain rule. It reproduces the course's nine-step example, passes a finite-difference
check, and trains a tiny MLP on XOR.

Run from guide/:  uv run code/11-gradient-descent-backprop/autograd_scratch.py
"""
import math
import random


# --- 1. A number that remembers where it came from -------------------------------
class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = data, 0.0
        self.parents = parents
        self._backward = lambda: None           # how to push my grad to my parents

    def __repr__(self):
        return f"Value({self.data:.4f}, grad={self.grad:.4f})"

    # --- 2. Each operation: compute forward, store the local chain-rule step -------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad               # d(a+b)/da = 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad  # d(ab)/da = b
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __radd__, __rmul__ = __add__, __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    # --- 3. Nonlinearities and the loss need their own local derivatives ----------
    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,))
        def _backward():
            self.grad += s * (1 - s) * out.grad     # sigma' = sigma (1 - sigma)
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,))
        def _backward():
            self.grad += out.grad / self.data       # d ln a / da = 1/a
        out._backward = _backward
        return out

    # --- 4. Backward: order nodes so each is visited after everything that uses it --
    def backward(self):
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v.parents:
                    visit(p)
                order.append(v)                     # parents first, then v
        visit(self)
        self.grad = 1.0                             # dL/dL = 1
        for v in reversed(order):
            v._backward()


# --- 5. A neuron and an MLP built from Values -----------------------------------
class Neuron:
    def __init__(self, n_in, rng):
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x, act):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return getattr(z, act)()


def bce(y_hat, y):
    return -(y * y_hat.log() + (1 - y) * (1 - y_hat).log())


def loss_at(w1, w2, b, x=(2.0, 1.0), y=1.0):
    z = w1 * x[0] + w2 * x[1] + b
    return -(y * math.log(1 / (1 + math.exp(-z))) + (1 - y) * math.log(1 - 1 / (1 + math.exp(-z))))


if __name__ == "__main__":
    print("== Course nine-step example: x = (2, 1), w = (0.5, -1), b = 0, y = 1 ==")
    w1, w2, b = Value(0.5), Value(-1.0), Value(0.0)
    z = w1 * 2.0 + w2 * 1.0 + b
    y_hat = z.sigmoid()
    L = bce(y_hat, 1.0)
    L.backward()
    print(f"z = {z.data:.4f}, y_hat = {y_hat.data:.4f}, L = {L.data:.4f}")
    print(f"dL/dz = {z.grad:.4f}, dL/dw1 = {w1.grad:.4f}, dL/dw2 = {w2.grad:.4f}, dL/db = {b.grad:.4f}")

    h = 1e-6
    fd = [(loss_at(0.5 + h, -1, 0) - loss_at(0.5 - h, -1, 0)) / (2 * h),
          (loss_at(0.5, -1 + h, 0) - loss_at(0.5, -1 - h, 0)) / (2 * h),
          (loss_at(0.5, -1, h) - loss_at(0.5, -1, -h)) / (2 * h)]
    print("finite differences:", [round(g, 6) for g in fd])

    print("\n== Fan-out: a node used twice gets both gradients added ==")
    a = Value(3.0)
    out = a * a + a                                 # d/da = 2a + 1 = 7
    out.backward()
    print(f"d(a*a + a)/da at a = 3: {a.grad}")

    print("\n== Train a 2-4-1 MLP on XOR with the engine (plain gradient descent) ==")
    rng = random.Random(0)
    hidden = [Neuron(2, rng) for _ in range(4)]
    head = Neuron(4, rng)
    params = [p for n in hidden + [head] for p in n.w + [n.b]]
    data = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    for step in range(501):
        loss = sum((bce(head([n(x, "tanh") for n in hidden], "sigmoid"), y) for x, y in data), Value(0.0))
        for p in params:
            p.grad = 0.0                            # zero_grad: grads accumulate otherwise
        loss.backward()
        for p in params:
            p.data -= 0.5 * p.grad
        if step % 100 == 0:
            print(f"step {step:3d}: loss {loss.data:.4f}")
    preds = [round(head([n(x, "tanh") for n in hidden], "sigmoid").data, 3) for x, _ in data]
    print("predictions:", preds, " targets: [0, 1, 1, 0]", " parameters:", len(params))
