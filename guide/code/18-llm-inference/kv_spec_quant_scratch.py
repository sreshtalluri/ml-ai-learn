# Adapted from AI Engineering from Scratch (MIT), phases/07-transformers-deep-dive/12-kv-cache-flash-attention, phases/07-transformers-deep-dive/16-speculative-decoding, phases/10-llms-from-scratch/11-quantization
"""Three inference tricks built from scratch with NumPy: a KV cache, int8 weight quantization, and
speculative sampling. All weights, inputs, and distributions are SYNTHETIC (seeded random numbers).
Run from guide/:  uv run code/18-llm-inference/kv_spec_quant_scratch.py
"""
import numpy as np

rng = np.random.default_rng(0)
D, T = 16, 12  # model width, sequence length (synthetic)
Wq, Wk, Wv = (rng.normal(0, D**-0.5, (D, D)) for _ in range(3))
X = rng.normal(0, 1, (T, D))  # one synthetic token embedding per row


# ---------- 1. KV cache: same outputs, far fewer projections ----------
def attend(q, K, V):
    s = K @ q / np.sqrt(D)  # [t] one score per cached position
    w = np.exp(s - s.max())
    return (w / w.sum()) @ V  # [D]


def decode_uncached(X):
    outs, proj = [], 0
    for t in range(1, len(X) + 1):
        K, V = X[:t] @ Wk, X[:t] @ Wv  # recompute K and V for the whole prefix
        proj += 2 * t
        outs.append(attend(X[t - 1] @ Wq, K, V))
    return np.array(outs), proj


def decode_cached(X):
    outs, K, V, proj = [], np.empty((0, D)), np.empty((0, D)), 0
    for x in X:
        K, V = np.vstack([K, x @ Wk]), np.vstack([V, x @ Wv])  # append one row each
        proj += 2
        outs.append(attend(x @ Wq, K, V))
    return np.array(outs), proj


# ---------- 2. int8 quantization: per-tensor versus per-channel scales ----------
def quant_int8(W, axis=None):
    scale = np.abs(W).max(axis=axis, keepdims=axis is not None) / 127  # absmax maps to 127
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale


def rel_error(W, W_hat):
    return np.linalg.norm(W - W_hat) / np.linalg.norm(W)


# ---------- 3. Speculative sampling: accept with min(1, q/p), else sample the residual ----------
def spec_one_token(q, p, rng):
    d = rng.choice(len(p), p=p)  # draft proposes from p
    if rng.random() < min(1.0, q[d] / p[d]):
        return d, True  # target agrees often enough: keep the draft token
    resid = np.maximum(q - p, 0)  # where the target wants more mass than the draft gave
    return rng.choice(len(q), p=resid / resid.sum()), False


def expected_tokens(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


if __name__ == "__main__":
    a, pa = decode_uncached(X)
    b, pb = decode_cached(X)
    print(f"[kv] max |uncached - cached| = {np.abs(a - b).max():.1e}")
    print(f"[kv] K/V projections: uncached {pa}, cached {pb} ({pa / pb:.1f}x fewer with cache)")
    assert np.allclose(a, b) and pa == T * (T + 1) and pb == 2 * T

    W = rng.normal(0, 0.02, (64, 256))  # synthetic weight matrix [out, in]
    W[:3] *= 40  # three outlier output channels (rows); one shared scale must stretch to cover them
    for name, axis in [("per-tensor", None), ("per-channel", 1)]:
        q, s = quant_int8(W, axis)
        print(f"[quant] int8 {name:11s}: relative error {rel_error(W, q * s):.4f}")
    print(f"[quant] bytes: fp16 {W.size * 2}, int8 {W.size} (+ scales)")

    q = np.array([0.35, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02])  # target (synthetic)
    p = np.array([0.25, 0.25, 0.10, 0.15, 0.10, 0.05, 0.05, 0.05])  # draft (synthetic)
    srng = np.random.default_rng(7)
    n = 50_000
    toks, acc = zip(*(spec_one_token(q, p, srng) for _ in range(n)))
    freq = np.bincount(toks, minlength=len(q)) / n
    alpha = np.minimum(p, q).sum()  # exact acceptance probability
    print(f"[spec] acceptance: theory {alpha:.3f}, simulated {np.mean(acc):.3f}")
    print(f"[spec] max |empirical - target| = {np.abs(freq - q).max():.4f}  (output follows q, not p)")
    print(f"[spec] expected tokens per target pass, gamma=4: {expected_tokens(alpha, 4):.2f}")
    assert np.abs(freq - q).max() < 0.01
