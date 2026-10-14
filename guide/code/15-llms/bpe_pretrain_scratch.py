# Adapted from AI Engineering from Scratch (MIT), phases/19-capstone-projects/30-bpe-tokenizer-from-scratch, phases/19-capstone-projects/31-tokenized-dataset-sliding-window, phases/19-capstone-projects/36-training-loop-eval
"""Byte-level BPE from scratch, a sliding-window next-token dataset, and a tiny pretraining
loop on CPU (a neural bigram model trained by hand-written gradient descent).

Synthetic: the training text is a short paragraph written for this script.
Run from guide/:  uv run code/15-llms/bpe_pretrain_scratch.py
"""
import math
import re
from collections import Counter

import numpy as np

CORPUS = (
    "the model reads tokens and predicts the next token. the next token is a guess. "
    "a good guess has a high probability and a low loss. the loss is the negative log "
    "probability of the true next token. training lowers the loss on the training text, "
    "and we check the loss on held out text to see if the model learned patterns or "
    "only memorized the text. tokens are pieces of words. common words become one token "
    "and rare words split into pieces. the tokenizer learns the pieces from the text. "
    "a model with more context can make a better guess, because the earlier tokens narrow "
    "down what can come next. a model with one token of context only knows the last token. "
    "the training text is small, so the model sees each pattern only a few times. "
)


# --- 1. Byte-level BPE: train ------------------------------------------------
def pretokenize(text):
    return re.findall(r"\S+|\s+", text)          # merges never cross a word/space boundary


def count_pairs(units):                           # units: {tuple of ids: count}
    pairs = Counter()
    for sym, c in units.items():
        for a, b in zip(sym, sym[1:]):
            pairs[(a, b)] += c
    return pairs


def apply_merge(sym, pair, new_id):
    out, i = [], 0
    while i < len(sym):
        if i < len(sym) - 1 and (sym[i], sym[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(sym[i]); i += 1
    return tuple(out)


def train_bpe(text, vocab_size):
    vocab = {i: bytes([i]) for i in range(256)}   # ids 0..255 are raw bytes: nothing is ever unknown
    merges = {}                                   # (a, b) -> new id, in the order learned
    units = Counter(tuple(w.encode("utf-8")) for w in pretokenize(text))
    while len(vocab) < vocab_size:
        pairs = count_pairs(units)
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))   # most frequent; ties -> smallest pair
        if pairs[best] < 2:
            break
        new_id = len(vocab)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        merges[best] = new_id
        units = Counter({apply_merge(s, best, new_id): c for s, c in units.items()})
    return vocab, merges


# --- 2. Byte-level BPE: encode and decode -------------------------------------
def encode(text, merges):
    rank = {p: r for r, p in enumerate(merges)}   # earlier merge = higher priority
    ids = []
    for w in pretokenize(text):
        sym = tuple(w.encode("utf-8"))
        while len(sym) > 1:
            pair = min(zip(sym, sym[1:]), key=lambda p: rank.get(p, math.inf))
            if pair not in rank:
                break                             # no learned merge applies any more
            sym = apply_merge(sym, pair, merges[pair])
        ids.extend(sym)
    return ids


def show(ids, vocab):                             # readable pieces; a lone byte of a multi-byte char shows as <hex>
    return [vocab[i].decode() if vocab[i].isascii() else f"<{vocab[i].hex()}>" for i in ids]


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


# --- 3. Sliding windows: inputs and targets shifted by one --------------------
def windows(ids, T, stride):
    ids = np.asarray(ids)
    starts = range(0, len(ids) - T, stride)       # each window needs T + 1 ids
    X = np.stack([ids[s:s + T] for s in starts])
    Y = np.stack([ids[s + 1:s + T + 1] for s in starts])
    return X, Y                                   # both [n_windows, T]


# --- 4. A tiny language model and its loss ------------------------------------
def init_model(V, d, rng):
    return {"E": rng.normal(0, 0.1, (V, d)), "W": rng.normal(0, 0.1, (d, V))}


def forward(params, X, Y):
    h = params["E"][X]                            # [B, T, d]   token embeddings
    logits = h @ params["W"]                      # [B, T, V]   scores for the next token
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits); p /= p.sum(-1, keepdims=True)
    loss = -np.log(np.take_along_axis(p, Y[..., None], -1)).mean()
    return loss, h, p


# --- 5. Backward pass and one training step -----------------------------------
def train_step(params, X, Y, lr):
    loss, h, p = forward(params, X, Y)
    d_logits = p.copy()
    d_logits[np.arange(Y.shape[0])[:, None], np.arange(Y.shape[1]), Y] -= 1   # softmax + CE gradient: p - onehot
    d_logits /= Y.size                                                          # mean over all positions
    dW = h.reshape(-1, h.shape[-1]).T @ d_logits.reshape(-1, d_logits.shape[-1])
    dE = np.zeros_like(params["E"])
    np.add.at(dE, X, d_logits @ params["W"].T)    # scatter-add: a token used twice gets both grads
    params["W"] -= lr * dW
    params["E"] -= lr * dE
    return loss


if __name__ == "__main__":
    print("== BPE on the lesson's toy corpus ==")
    toy = " ".join(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3)
    vocab, merges = train_bpe(toy, 256 + 5)
    print("first 5 merges:", [vocab[i].decode() for i in merges.values()])

    print("\n== BPE on the training paragraph ==")
    text, held_out = CORPUS, "the tokenizer predicts rare tokens. naïve café!"
    vocab, merges = train_bpe(text, 256 + 60)
    ids = encode(held_out, merges)
    print("held-out text:", held_out)
    print("tokens:", show(ids, vocab))
    print(f"{len(held_out.encode())} bytes -> {len(ids)} tokens; round trip ok: {decode(ids, vocab) == held_out}")

    print("\n== Sliding windows ==")
    corpus_ids = encode(text, merges)
    X, Y = windows(corpus_ids, T=8, stride=8)
    print(f"{len(corpus_ids)} tokens -> X {X.shape}, Y {Y.shape}")
    print("X[0]:", X[0].tolist())
    print("Y[0]:", Y[0].tolist(), "(X shifted left by one)")

    print("\n== Pretraining loop (neural bigram, NumPy, CPU) ==")
    rng = np.random.default_rng(0)
    split = int(0.9 * len(X))
    Xtr, Ytr, Xva, Yva = X[:split], Y[:split], X[split:], Y[split:]
    V = len(vocab)
    params = init_model(V, d=32, rng=rng)
    unigram = np.bincount(Ytr.ravel(), minlength=V) + 1e-9
    uni_loss = -np.log(unigram[Yva] / unigram.sum()).mean()
    print(f"baselines on validation: uniform ln V = {math.log(V):.3f}, unigram = {uni_loss:.3f}")
    for step in range(301):
        b = rng.integers(0, len(Xtr), 16)              # random batch of 16 windows
        loss = train_step(params, Xtr[b], Ytr[b], lr=2.0)
        if step % 100 == 0:
            val, _, _ = forward(params, Xva, Yva)
            print(f"step {step:3d}  train loss {loss:.3f}  val loss {val:.3f}  val perplexity {math.exp(val):6.1f}")

    tok, out = encode("the", merges)[-1], []
    for _ in range(8):                                  # greedy generation, one token at a time
        _, _, p = forward(params, np.array([[tok]]), np.array([[0]]))
        tok = int(p[0, 0].argmax()); out.append(tok)
    print("greedy continuation of 'the':", repr(decode(out, vocab)))
