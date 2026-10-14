# Adapted from AI Engineering from Scratch (MIT), phases/03-deep-learning-core/01-the-perceptron, phases/03-deep-learning-core/02-multi-layer-networks, phases/03-deep-learning-core/04-activation-functions
"""From one perceptron to a multilayer network: the perceptron learning rule (solves AND, fails
XOR), a reusable Layer, the course's 2-3-1 worked example, a hand-wired XOR network, and the
common activation functions side by side.

Run from guide/:  uv run code/10-neural-networks/mlp_scratch.py
"""
import numpy as np

AND = (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1]))
XOR = (AND[0], np.array([0, 1, 1, 0]))


# --- 1. The perceptron: a weighted sum, a hard threshold, an error-driven update ---
def train_perceptron(X, y, lr=0.1, epochs=50):
    w, b = np.zeros(X.shape[1]), 0.0
    for epoch in range(1, epochs + 1):
        mistakes = 0
        for xi, yi in zip(X, y):
            pred = int(w @ xi + b >= 0)          # step function: fire or not
            if pred != yi:
                w += lr * (yi - pred) * xi       # nudge the boundary toward the missed point
                b += lr * (yi - pred)
                mistakes += 1
        if mistakes == 0:
            return w, b, epoch                   # a full pass with no mistakes: converged
    return w, b, None                            # never converged


# --- 2. Activation functions (the nonlinearity between layers) -------------------
ACTIVATIONS = {
    "relu": lambda z: np.maximum(0, z),
    "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
    "tanh": np.tanh,
    "gelu": lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3))),
}


# --- 3. A dense layer: affine map then activation; a network is a list of layers ---
class Layer:
    def __init__(self, W, b, act):
        self.W, self.b, self.act = np.asarray(W, float), np.asarray(b, float), ACTIVATIONS[act]

    def __call__(self, X):                       # X: [B, d_in]
        return self.act(X @ self.W.T + self.b)   # [B, d_in] @ [d_in, d_out] -> [B, d_out]

    def n_params(self):
        return self.W.size + self.b.size


def forward(layers, X):
    for layer in layers:
        X = layer(X)
    return X


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("== Perceptron learning rule ==")
    for name, (X, y) in [("AND", AND), ("XOR", XOR)]:
        w, b, epoch = train_perceptron(X, y)
        preds = (X @ w + b >= 0).astype(int)
        status = f"converged at epoch {epoch}" if epoch else "did NOT converge in 50 epochs"
        print(f"{name}: {status}; predictions {preds}, targets {y}")

    print("\n== Course worked example: 2 -> 3 (ReLU) -> 1 (sigmoid) ==")
    net = [Layer([[0.5, -1], [1, 1], [-0.5, 0.25]], [0, -1, 0.5], "relu"),
           Layer([[1, -0.5, 2]], [0.25], "sigmoid")]
    x = np.array([[1.0, 2.0]])                   # one example, shape [1, 2]
    print("hidden a1 =", net[0](x)[0], " y_hat =", forward(net, x)[0])
    print("parameters:", sum(layer.n_params() for layer in net))

    print("\n== Hand-wired XOR with two ReLU units ==")
    xor_net = [Layer([[1, 1], [1, 1]], [0, -1], "relu"),     # h1 = x1+x2, h2 = x1+x2-1
               Layer([[1, -2]], [0], "relu")]                # y = h1 - 2*h2
    print("inputs ", XOR[0].tolist())
    print("outputs", forward(xor_net, XOR[0]).ravel(), " targets", XOR[1])

    print("\n== Without a nonlinearity, two layers collapse into one ==")
    W1, b1 = net[0].W, net[0].b
    W2, b2 = net[1].W, net[1].b
    Wc, bc = W2 @ W1, W2 @ b1 + b2
    print("two linear layers:", (W2 @ (W1 @ x[0] + b1) + b2), " one layer Wc x + bc:", Wc @ x[0] + bc)

    print("\n== Activations at z = -2, -0.5, 0, 0.5, 2 ==")
    z = np.array([-2, -0.5, 0, 0.5, 2])
    for name, f in ACTIVATIONS.items():
        print(f"{name:>7}: {f(z)}")

    print("\n== Dead ReLU units (synthetic: random inputs, badly shifted biases) ==")
    rng = np.random.default_rng(0)
    Xr = rng.normal(size=(1000, 5))
    for shift in (0.0, -3.0):
        H = ACTIVATIONS["relu"](Xr @ rng.normal(size=(5, 20)) * 0.3 + shift)
        dead = np.mean((H > 0).sum(axis=0) == 0)
        print(f"bias {shift:+.0f}: {dead:.0%} of 20 units never fire on 1,000 inputs")
