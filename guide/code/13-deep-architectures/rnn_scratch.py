# Adapted from AI Engineering from Scratch (MIT), phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text, phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence
"""Recurrent cells from scratch: a vanilla RNN (the course's scalar unroll, then vectors), an LSTM
cell and a GRU cell with every gate written out, a measurement of how the gradient to the
first step shrinks with sequence length, and a 1-D text CNN for contrast.

Synthetic: random weights and random embeddings.
Run from guide/:  uv run code/13-deep-architectures/rnn_scratch.py
"""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --- 1. Vanilla RNN: one tanh layer applied at every step, carrying h forward -----
def rnn_forward(xs, Wx, Wh, b, h0):
    h, hs = h0, []
    for x in xs:                                   # sequential: step t needs step t-1
        h = np.tanh(Wx @ x + Wh @ h + b)
        hs.append(h)
    return np.stack(hs)                            # [T, d_h]


# --- 2. LSTM cell: a separate cell state c, edited by three gates -------------------
def lstm_step(x, h, c, W, b):
    z = W @ np.concatenate([x, h]) + b             # all four blocks in one matmul: [4*d_h]
    f, i, o, g = np.split(z, 4)
    f, i, o, g = sigmoid(f), sigmoid(i), sigmoid(o), np.tanh(g)
    c = f * c + i * g                              # forget some old, write some new
    h = o * np.tanh(c)                             # expose a filtered view of c
    return h, c


# --- 3. GRU cell: two gates, no separate cell state ----------------------------------
def gru_step(x, h, W, U, b):
    zr = sigmoid(W[:2] @ x + U[:2] @ h + b[:2])    # update gate z, reset gate r (stacked)
    z, r = zr[0], zr[1]
    h_new = np.tanh(W[2] @ x + U[2] @ (r * h) + b[2])
    return (1 - z) * h + z * h_new                 # interpolate old and candidate


# --- 4. Gradient to the first hidden state: a product of T Jacobians -----------------
def grad_norm_to_start(Wh, T, rng, d_x=4):
    d_h = Wh.shape[0]
    Wx = rng.normal(scale=0.5, size=(d_h, d_x))
    h, J = np.zeros(d_h), np.eye(d_h)
    for _ in range(T):
        h = np.tanh(Wx @ rng.normal(size=d_x) + Wh @ h)
        J = np.diag(1 - h ** 2) @ Wh @ J           # dh_t/dh_0 = diag(tanh') Wh ... chain rule
    return np.linalg.norm(J, 2)


# --- 5. Text CNN: slide a filter over word embeddings, max-pool over positions -------
def text_cnn_feature(E, F, bias):                  # E: [T, d], F: [width, d]
    w = F.shape[0]
    acts = [max(0.0, float(np.sum(E[t:t + w] * F)) + bias) for t in range(len(E) - w + 1)]
    return max(acts), acts                         # one feature per filter, any length


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("== Course worked example: scalar RNN, wx=0.5, wh=0.8, x=[1, 0, 2] ==")
    hs = rnn_forward([np.array([1.0]), np.array([0.0]), np.array([2.0])],
                     np.array([[0.5]]), np.array([[0.8]]), np.zeros(1), np.zeros(1))
    print("h_1..h_3:", hs.ravel())
    print("linear gradient factors: 0.8^3 =", round(0.8 ** 3, 4), " 0.8^30 =", round(0.8 ** 30, 4),
          " 1.1^30 =", round(1.1 ** 30, 1))

    rng = np.random.default_rng(0)
    d_x, d_h, T = 4, 8, 6
    xs = rng.normal(size=(T, d_x))
    print(f"\n== Shapes: T={T}, d_x={d_x}, d_h={d_h} ==")
    print("RNN states:", rnn_forward(xs, rng.normal(size=(d_h, d_x)), rng.normal(size=(d_h, d_h)) * 0.3,
                                     np.zeros(d_h), np.zeros(d_h)).shape)
    W = rng.normal(scale=0.3, size=(4 * d_h, d_x + d_h))
    b = np.zeros(4 * d_h); b[:d_h] = 1.0           # forget-gate bias 1: start by remembering
    h, c = np.zeros(d_h), np.zeros(d_h)
    for x in xs:
        h, c = lstm_step(x, h, c, W, b)
    print("LSTM h, c:", h.shape, c.shape, " params:", W.size + b.size, "(4 x d_h x (d_x + d_h + 1))")
    Wg, Ug = rng.normal(scale=0.3, size=(3, d_h, d_x)), rng.normal(scale=0.3, size=(3, d_h, d_h))
    hg = np.zeros(d_h)
    for x in xs:
        hg = gru_step(x, hg, Wg, Ug, np.zeros((3, d_h)))
    print("GRU h:", hg.shape, " params:", Wg.size + Ug.size + 3 * d_h, "(3 x d_h x (d_x + d_h + 1))")

    print("\n== ||dh_T / dh_0|| for a tanh RNN, d_h = 8 (synthetic random inputs) ==")
    Q, _ = np.linalg.qr(rng.normal(size=(d_h, d_h)))
    for scale in (0.5, 1.0, 1.5):
        norms = [grad_norm_to_start(scale * Q, T, np.random.default_rng(1)) for T in (5, 20, 50)]
        print(f"Wh = {scale} x orthogonal: T=5 {norms[0]:.2e}  T=20 {norms[1]:.2e}  T=50 {norms[2]:.2e}")
    print("LSTM cell path with forget gate 0.95: 0.95^50 =", f"{0.95 ** 50:.3f}")

    print("\n== Text CNN: width-2 filter over 5 synthetic 3-d embeddings ==")
    E = np.array([[1.0, 0.2, 0.5], [0.8, 0.9, 0.1], [0.3, 0.4, 0.7], [0.6, 0.5, 0.5], [0.1, 0.8, 0.2]])
    feat, acts = text_cnn_feature(E, np.array([[0.5, 0.0, 0.5], [0.2, 0.3, 0.1]]), -0.5)
    print("activations per position:", [round(a, 2) for a in acts], " max-pooled feature:", round(feat, 2))
