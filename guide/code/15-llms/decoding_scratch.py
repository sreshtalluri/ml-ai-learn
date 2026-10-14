# Adapted from AI Engineering from Scratch (MIT), phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding, phases/11-llm-engineering/03-structured-outputs
"""Decoding from scratch: temperature, top-k, top-p, sampling, greedy vs beam search on a toy
language model, and constrained decoding (a logit mask that forces one of a fixed set of labels).

Synthetic: the toy model's probabilities are made up to show the mechanics.
Run from guide/:  uv run code/15-llms/decoding_scratch.py
"""
import math

import numpy as np


# --- 1. Temperature -----------------------------------------------------------
def softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    e = np.exp(z - z.max())                       # -inf logits become exactly 0
    return e / e.sum()


# --- 2. Top-k and top-p filters -----------------------------------------------
def top_k(p, k):
    keep = np.zeros_like(p, dtype=bool)
    keep[np.argsort(-p)[:k]] = True
    q = np.where(keep, p, 0.0)
    return q / q.sum()                            # renormalize what is left


def top_p(p, threshold):
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    n_keep = np.searchsorted(cum, threshold - 1e-12) + 1   # smallest prefix with mass >= threshold
    q = np.zeros_like(p)
    q[order[:n_keep]] = p[order[:n_keep]]
    return q / q.sum()


# --- 3. One sampling step -------------------------------------------------------
def sample_next(logits, rng, T=1.0, k=None, p=None):
    if T == 0:
        return int(np.argmax(logits))             # greedy: temperature -> 0
    probs = softmax(logits, T)
    if k is not None:
        probs = top_k(probs, k)
    if p is not None:
        probs = top_p(probs, p)
    return int(rng.choice(len(probs), p=probs))


# --- 4. Greedy vs beam search on a toy model -----------------------------------
TOY = {                                           # P(next | previous word); synthetic
    "<s>": {"the": 0.5, "a": 0.4, "one": 0.1},
    "the": {"cat": 0.4, "dog": 0.35, "end": 0.25},
    "a": {"cat": 0.9, "dog": 0.1},
    "one": {"cat": 1.0},
    "cat": {"</s>": 1.0}, "dog": {"</s>": 1.0}, "end": {"</s>": 1.0},
}


def greedy(model, start="<s>"):
    seq, logp = [start], 0.0
    while seq[-1] != "</s>":
        word, prob = max(model[seq[-1]].items(), key=lambda kv: kv[1])
        seq.append(word); logp += math.log(prob)
    return seq, logp


def beam_search(model, B, start="<s>"):
    beams = [([start], 0.0)]                      # (sequence, total log-probability)
    while any(s[-1] != "</s>" for s, _ in beams):
        cand = []
        for seq, lp in beams:
            if seq[-1] == "</s>":
                cand.append((seq, lp)); continue  # finished beams carry over
            for w, pr in model[seq[-1]].items():
                cand.append((seq + [w], lp + math.log(pr)))
        beams = sorted(cand, key=lambda c: -c[1])[:B]   # keep the B best partial sequences
    return beams[0]


# --- 5. Constrained decoding: mask logits to valid continuations ---------------
def allowed_next(prefix, choices):
    """Characters that keep `prefix` on the way to one of `choices` (a prefix trie, flattened)."""
    nxt = {c[len(prefix)] for c in choices if c.startswith(prefix) and len(c) > len(prefix)}
    if prefix in choices:
        nxt.add("<eos>")
    return nxt


def constrained_generate(logit_fn, vocab, choices):
    out = ""
    while True:
        logits = logit_fn(out)
        ok = allowed_next(out, choices)
        masked = np.where([v in ok for v in vocab], logits, -np.inf)   # invalid tokens get -inf
        tok = vocab[int(np.argmax(masked))]                              # greedy over valid tokens only
        if tok == "<eos>":
            return out
        out += tok


if __name__ == "__main__":
    print("== Worked example 1: temperature on logits [2, 1, 0] ==")
    for T in (1.0, 0.5, 2.0):
        print(f"T={T}: {softmax([2, 1, 0], T).round(3)}")

    print("\n== Worked example 2: filters on p = [0.5, 0.3, 0.15, 0.05] ==")
    p = np.array([0.5, 0.3, 0.15, 0.05])
    print("top-k k=2  :", top_k(p, 2).round(3))
    print("top-p p=0.9:", top_p(p, 0.9).round(3))
    print("top-p p=0.8:", top_p(p, 0.8).round(3))

    print("\n== Sampling 1,000 times (synthetic logits for 'The capital of France is') ==")
    tokens = ["Paris", "a", "the", "located", "known", "Lyon", "not", "beautiful", "France", "Marseille"]
    logits = np.array([6.1, 3.2, 2.9, 2.4, 2.1, 1.6, 1.2, 1.0, 0.6, 0.4])
    for T, pp in [(0, None), (0.5, None), (1.0, None), (1.5, None), (1.5, 0.9)]:
        rng = np.random.default_rng(0)
        picks = [sample_next(logits, rng, T=T, p=pp) for _ in range(1000)]
        print(f"T={T:<3} top_p={pp}: Paris {picks.count(0) / 10:5.1f}%, distinct tokens {len(set(picks))}")

    print("\n== Greedy vs beam search (toy model) ==")
    for name, (seq, lp) in [("greedy", greedy(TOY)), ("beam B=2", beam_search(TOY, 2))]:
        print(f"{name:9s}: {' '.join(seq[1:-1]):8s} log-prob {lp:.3f}  P = {math.exp(lp):.2f}")

    print("\n== Constrained decoding: force one of three labels ==")
    vocab = list("abcdefghijklmnopqrstuvwxyz .") + ["<eos>"]
    prose = "the sentiment is positive."
    def chatty(prefix):                           # a toy model that wants to write a sentence
        z = np.zeros(len(vocab))
        target = prose[len(prefix)] if len(prefix) < len(prose) else "<eos>"
        z[vocab.index(target)] = 5.0
        z[vocab.index("p")] += 1.0                # it does lean toward "positive" over the others
        return z
    free = "".join(vocab[int(np.argmax(chatty(prose[:i])))] for i in range(len(prose)))
    print("unconstrained:", repr(free))
    print("constrained  :", repr(constrained_generate(chatty, vocab, ["positive", "negative", "neutral"])))
