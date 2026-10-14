# Adapted from AI Engineering from Scratch (MIT), phases/07-transformers-deep-dive/04-positional-encoding, phases/07-transformers-deep-dive/05-full-transformer, phases/07-transformers-deep-dive/14-build-a-transformer-capstone, phases/19-capstone-projects/32-token-positional-embeddings, phases/19-capstone-projects/34-transformer-block, phases/19-capstone-projects/35-gpt-model-assembly
"""A GPT-style decoder forward pass in NumPy: embeddings, pre-norm blocks (attention + MLP,
each with a residual), final LayerNorm, and a weight-tied output head. Counts GPT-2-small's
parameters from the same shape table used to build the tiny model.

Synthetic: random weights and a toy 12-token vocabulary, so the generated text is gibberish.
Run from guide/:  uv run code/14-transformers/gpt_scratch.py
"""
import sys
from math import prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_scratch import attention, softmax, split_heads  # noqa: E402


# --- 1. Shapes of every parameter (one table drives both init and counting) ----
def param_shapes(V, ctx, d, L, d_ff):
    block = {"ln1_g": (d,), "ln1_b": (d,), "W_qkv": (d, 3 * d), "b_qkv": (3 * d,),
             "W_o": (d, d), "b_o": (d,), "ln2_g": (d,), "ln2_b": (d,),
             "W_1": (d, d_ff), "b_1": (d_ff,), "W_2": (d_ff, d), "b_2": (d,)}
    top = {"tok_emb": (V, d), "pos_emb": (ctx, d), "lnf_g": (d,), "lnf_b": (d,)}
    return top, [block] * L          # no separate output head: it reuses tok_emb


def count_params(V, ctx, d, L, d_ff):
    top, blocks = param_shapes(V, ctx, d, L, d_ff)
    return sum(prod(s) for s in top.values()) + sum(prod(s) for b in blocks for s in b.values())


# --- 2. The two per-token operations: LayerNorm and the GELU MLP ---------------
def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)                  # per token, over the d features
    var = x.var(axis=-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b


def gelu(x):                                             # tanh approximation used by GPT-2
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def mlp(x, p):
    return gelu(x @ p["W_1"] + p["b_1"]) @ p["W_2"] + p["b_2"]   # [n, d] -> [n, d_ff] -> [n, d]


# --- 3. Causal multi-head attention with one fused QKV projection --------------
def causal_mha(x, p, h):
    q, k, v = np.split(x @ p["W_qkv"] + p["b_qkv"], 3, axis=-1)   # three [n, d]
    out, _ = attention(split_heads(q, h), split_heads(k, h), split_heads(v, h), causal=True)
    merged = out.transpose(1, 0, 2).reshape(x.shape[0], -1)       # [n, d]
    return merged @ p["W_o"] + p["b_o"]


# --- 4. One pre-norm block: normalize, sublayer, add back ----------------------
def block(x, p, h):
    x = x + causal_mha(layer_norm(x, p["ln1_g"], p["ln1_b"]), p, h)   # communicate
    x = x + mlp(layer_norm(x, p["ln2_g"], p["ln2_b"]), p)             # compute
    return x                                                          # still [n, d]


# --- 5. The whole model: embed, stack blocks, normalize, project to the vocab ---
def gpt_forward(ids, P, h):
    n = len(ids)
    x = P["tok_emb"][ids] + P["pos_emb"][:n]           # [n, d]: what + where
    for p in P["blocks"]:
        x = block(x, p, h)
    x = layer_norm(x, P["lnf_g"], P["lnf_b"])
    return x @ P["tok_emb"].T                          # tied head: logits [n, V]


def init(V, ctx, d, L, d_ff, rng):
    top, blocks = param_shapes(V, ctx, d, L, d_ff)
    def make(name, s):
        if name.endswith("_g"):
            return np.ones(s)                          # LayerNorm scale starts at 1
        if name.startswith("b_") or name.endswith("_b"):
            return np.zeros(s)
        return rng.normal(scale=0.02, size=s)          # GPT-2's init scale
    P = {k: make(k, s) for k, s in top.items()}
    P["blocks"] = [{k: make(k, s) for k, s in b.items()} for b in blocks]
    return P


def sinusoidal_positions(n, d):
    pos, i = np.arange(n)[:, None], np.arange(0, d, 2)[None, :]
    angle = pos / 10000 ** (i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("== GPT-2-small parameter count from the shape table ==")
    print(f"total: {count_params(50257, 1024, 768, 12, 3072):,}")
    _, blocks = param_shapes(50257, 1024, 768, 12, 3072)
    print(f"one block: {sum(prod(s) for s in blocks[0].values()):,}")

    print("\n== Tiny model: V=12, ctx=16, d=16, L=2, h=4, d_ff=64 (synthetic weights) ==")
    rng = np.random.default_rng(0)
    V, ctx, d, L, h, d_ff = 12, 16, 16, 2, 4, 64
    P = init(V, ctx, d, L, d_ff, rng)
    ids = np.array([3, 7, 1, 4, 9])
    logits = gpt_forward(ids, P, h)
    print("ids shape:", ids.shape, "-> logits shape:", logits.shape)
    print(f"parameters: {count_params(V, ctx, d, L, d_ff):,}")

    print("\n== Causality check: change the last token, earlier logits must not move ==")
    ids2 = ids.copy(); ids2[-1] = 0
    logits2 = gpt_forward(ids2, P, h)
    print("positions 0-3 unchanged:", bool(np.allclose(logits[:4], logits2[:4])))
    print("position 4 changed:     ", not np.allclose(logits[4], logits2[4]))

    print("\n== Greedy generation, 5 new tokens (random weights, so meaningless) ==")
    seq = ids.tolist()
    for _ in range(5):
        nxt = int(np.argmax(gpt_forward(np.array(seq[-ctx:]), P, h)[-1]))   # last position only
        seq.append(nxt)
    print("sequence:", seq)
    print("next-token probs sum:", round(float(softmax(logits[-1]).sum()), 6))

    print("\n== Sinusoidal positions (the parameter-free alternative to pos_emb) ==")
    pe = sinusoidal_positions(ctx, d)
    print("pos 0, first 4 dims:", pe[0, :4], " every row has norm sqrt(d/2):",
          bool(np.allclose(np.linalg.norm(pe, axis=1), np.sqrt(d / 2))))
