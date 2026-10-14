# Adapted from AI Engineering from Scratch (MIT), phases/07-transformers-deep-dive/02-self-attention-from-scratch, phases/07-transformers-deep-dive/03-multi-head-attention, phases/19-capstone-projects/33-multihead-self-attention
"""Self-attention built up from nothing: softmax, one head, a causal mask, then multi-head
attention done two ways (a loop over heads and one batched reshape) that must agree.

Synthetic: random embeddings and random projection weights, so the weights show mechanics only.
Run from guide/:  uv run code/14-transformers/attention_scratch.py
"""
import numpy as np


# --- 1. Softmax that cannot overflow -----------------------------------------
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)    # shift so the largest entry is 0
    e = np.exp(z)                              # every value now in (0, 1]
    return e / e.sum(axis=axis, keepdims=True)


# --- 2. Scaled dot-product attention -----------------------------------------
def attention(Q, K, V, causal=False):
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)   # [..., n, n]
    if causal:
        n = scores.shape[-1]
        future = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)       # exp(-inf) = 0
    A = softmax(scores)                                  # each row sums to 1
    return A @ V, A                                      # [..., n, d_v], [..., n, n]


# --- 3. Multi-head attention, the readable way: a loop over heads -------------
def mha_loop(X, Wq, Wk, Wv, Wo, h, causal=True):
    d_head = X.shape[-1] // h
    outs = []
    for i in range(h):
        cols = slice(i * d_head, (i + 1) * d_head)       # this head's slice of the projections
        out, _ = attention(X @ Wq[:, cols], X @ Wk[:, cols], X @ Wv[:, cols], causal)
        outs.append(out)                                 # [n, d_head]
    return np.concatenate(outs, axis=-1) @ Wo            # [n, d_model]


# --- 4. Multi-head attention, the fast way: reshape so heads are a batch axis --
def split_heads(M, h):                                   # [n, d] -> [h, n, d/h]
    n, d = M.shape
    return M.reshape(n, h, d // h).transpose(1, 0, 2)


def mha_batched(X, Wq, Wk, Wv, Wo, h, causal=True):
    Q, K, V = (split_heads(X @ W, h) for W in (Wq, Wk, Wv))
    out, A = attention(Q, K, V, causal)                  # all heads in one call
    merged = out.transpose(1, 0, 2).reshape(X.shape[0], -1)   # [n, d_model]
    return merged @ Wo, A


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("== Course worked example (one query, two keys, d_k = 2) ==")
    q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 6.0]])
    out, A = attention(q, K, V)
    print("weights:", A[0], " output:", out[0])

    print("\n== Softmax stability ==")
    print("softmax([1000, 1001, 1002]) =", softmax(np.array([1000.0, 1001.0, 1002.0])))

    rng = np.random.default_rng(0)
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    n, d_model, h = len(tokens), 8, 2
    X = rng.normal(size=(n, d_model))                    # synthetic embeddings [6, 8]
    Wq, Wk, Wv, Wo = (rng.normal(scale=d_model ** -0.5, size=(d_model, d_model)) for _ in range(4))

    print("\n== Causal single head (d_k = 8) ==")
    _, A = attention(X @ Wq, X @ Wk, X @ Wv, causal=True)
    print("row sums:", A.sum(axis=-1))
    print("row 0 (first token can only see itself):", A[0])
    print("upper triangle all zero:", bool(np.allclose(np.triu(A, k=1), 0)))

    print(f"\n== Multi-head: d_model = {d_model}, h = {h}, d_head = {d_model // h} ==")
    y_loop = mha_loop(X, Wq, Wk, Wv, Wo, h)
    y_fast, A_heads = mha_batched(X, Wq, Wk, Wv, Wo, h)
    print("output shape:", y_fast.shape, " per-head weights shape:", A_heads.shape)
    print("loop and batched agree:", bool(np.allclose(y_loop, y_fast)))
    print("head 0 vs head 1 attention for 'mat':")
    print("  head 0:", A_heads[0, -1])
    print("  head 1:", A_heads[1, -1])
    print("projection parameters (4 * d_model^2):", 4 * d_model * d_model)
