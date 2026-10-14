# Adapted from AI Engineering from Scratch (MIT), phases/03-deep-learning-core/11-intro-to-pytorch
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "numpy>=1.26", "matplotlib>=3.8"]
# ///
"""A micro autograd engine from scratch, then the same ideas in PyTorch: tensors, autograd,
nn.Module, the training loop, eval mode, and saving a state_dict. CPU only.

Data is a SYNTHETIC two-moons-style dataset generated with a fixed seed.
Run from guide/:  uv run code/10-neural-networks/intro_to_pytorch.py
(The first run downloads PyTorch, about 1 to 2 minutes.)
"""
import io
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402


# --- From scratch: a scalar autograd engine ------------------------------------------
class Value:
    """A number that remembers how it was computed, so it can send gradients back."""

    def __init__(self, data, parents=(), backward=lambda: None):
        self.data, self.grad = data, 0.0
        self._parents, self._backward = parents, backward

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def backward():                       # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def backward():                       # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def __sub__(self, other):
        return self + (other * -1 if isinstance(other, Value) else -other)

    def __pow__(self, k):
        out = Value(self.data**k, (self,))
        def backward():                       # d(a^k)/da = k a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = backward
        return out

    def backward(self):
        order, seen = [], set()
        def visit(v):                         # topological order: parents before children
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0                       # dL/dL = 1
        for v in reversed(order):             # walk from the loss back to the leaves
            v._backward()


# --- PyTorch: a model and a training loop ----------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in=2, hidden=32, d_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_out),           # raw logits: no softmax here
        )

    def forward(self, x):                       # x: [batch, d_in] -> logits: [batch, d_out]
        return self.net(x)


def make_moons(n=600, noise=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * math.pi
    y = torch.randint(0, 2, (n,), generator=g)
    x = torch.stack([torch.cos(t), torch.sin(t)], 1)
    x[y == 1] = torch.stack([1 - torch.cos(t[y == 1]), 0.5 - torch.sin(t[y == 1])], 1)
    return x + noise * torch.randn(n, 2, generator=g), y   # SYNTHETIC two interleaved half-circles


def train(model, loader, epochs, lr=1e-2, zero_grad=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        model.train()                           # dropout on
        total = 0.0
        for xb, yb in loader:
            logits = model(xb)                  # 1. forward
            loss = F.cross_entropy(logits, yb)  # 2. loss
            if zero_grad:
                opt.zero_grad()                 # 3. clear old gradients
            loss.backward()                     # 4. backward: fill .grad
            opt.step()                          # 5. update parameters
            total += loss.item() * len(xb)
        losses.append(total / len(loader.dataset))
    return losses


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()                                # dropout off
    return (model(x).argmax(1) == y).float().mean().item()


def main():
    torch.manual_seed(0)

    print("=== From scratch: L = (w*x + b - y)^2 with x=2, w=3, b=1, y=10 ===")
    w, b = Value(3.0), Value(1.0)
    L = (w * 2.0 + b - 10.0) ** 2
    L.backward()
    print(f"L = {L.data}, dL/dw = {w.grad}, dL/db = {b.grad}")

    print("\n=== Same graph in PyTorch ===")
    wt = torch.tensor(3.0, requires_grad=True)
    bt = torch.tensor(1.0, requires_grad=True)
    Lt = (wt * 2.0 + bt - 10.0) ** 2
    Lt.backward()
    print(f"L = {Lt.item()}, dL/dw = {wt.grad.item()}, dL/db = {bt.grad.item()}")
    Lt = (wt * 2.0 + bt - 10.0) ** 2
    Lt.backward()
    print(f"after a second backward without zeroing: dL/dw = {wt.grad.item()} (gradients accumulate)")
    with torch.no_grad():
        z = wt * 2.0
    print(f"inside no_grad: requires_grad = {z.requires_grad}")

    print("\n=== Tensors: shape, dtype, broadcasting, views ===")
    X = torch.arange(6, dtype=torch.float32).reshape(2, 3)     # [2, 3]
    bias = torch.tensor([10.0, 20.0, 30.0])                      # [3] broadcasts over rows
    print(f"X {tuple(X.shape)} {X.dtype}\n{X}")
    print(f"X + bias =\n{X + bias}")
    print(f"X @ X.T shape {tuple((X @ X.T).shape)};  X.T is a view: shares storage = {X.T.data_ptr() == X.data_ptr()}")
    print(f"device: {X.device}  (move with .to(\"cuda\") or .to(\"mps\") when available)")

    print("\n=== nn.Linear(2, 3) ===")
    lin = nn.Linear(2, 3)
    print(f"weight {tuple(lin.weight.shape)}, bias {tuple(lin.bias.shape)}  (out_features, in_features)")

    print("\n=== Training an MLP on synthetic moons (CPU) ===")
    x, y = make_moons()
    xtr, ytr, xva, yva = x[:480], y[:480], x[480:], y[480:]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xtr, ytr), batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    model = MLP()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params}  (2*32+32 + 32*32+32 + 32*2+2)")
    with torch.no_grad():
        model.eval()
        print(f"step-0 loss {F.cross_entropy(model(xtr), ytr).item():.4f} (ln 2 = {math.log(2):.4f})")
    good = train(model, loader, epochs=40)
    print(f"correct loop:      final train loss {good[-1]:.4f}, val accuracy {accuracy(model, xva, yva):.3f}")

    torch.manual_seed(0)
    buggy_model = MLP()
    buggy = train(buggy_model, loader, epochs=40, zero_grad=False)
    print(f"no zero_grad():    final train loss {buggy[-1]:.4f}, val accuracy {accuracy(buggy_model, xva, yva):.3f}")

    model.train()
    a, b2 = model(xva[:4]), model(xva[:4])
    model.eval()
    c, d = model(xva[:4]), model(xva[:4])
    print(f"train mode, same input twice -> identical? {torch.equal(a, b2)};  eval mode -> identical? {torch.equal(c, d)}")

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    restored = MLP()
    restored.load_state_dict(torch.load(buf, weights_only=True))
    restored.eval()
    print(f"state_dict keys: {list(model.state_dict())[:2]} ...;  reloaded model matches: {torch.equal(restored(xva), model(xva))}")

    # --- Figure ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    axes[0].plot(good, color=BLUE, label="zero_grad() every step")
    axes[0].plot(buggy, color=ORANGE, label="zero_grad() forgotten")
    axes[0].set(title="Training loss, same model and seed", xlabel="epoch", ylabel="cross-entropy (nats)")
    axes[0].legend()

    gx, gy = torch.meshgrid(torch.linspace(-1.6, 2.6, 200), torch.linspace(-1.1, 1.6, 200), indexing="xy")
    with torch.no_grad():
        model.eval()
        p = F.softmax(model(torch.stack([gx.ravel(), gy.ravel()], 1)), 1)[:, 1].reshape(gx.shape)
    axes[1].contourf(gx, gy, p, levels=[0, 0.5, 1], colors=["#dbeafe", "#ccfbf1"])
    axes[1].contour(gx, gy, p, levels=[0.5], colors=[GRAY])
    for cls, col in ((0, BLUE), (1, TEAL)):
        axes[1].scatter(*x[y == cls].T, s=8, color=col, label=f"class {cls}")
    axes[1].set(title=f"Decision boundary (val acc {accuracy(model, xva, yva):.2f})", xlabel="x1", ylabel="x2")
    axes[1].legend(loc="lower left", fontsize=8)
    save(fig, "intro-to-pytorch")


if __name__ == "__main__":
    main()
