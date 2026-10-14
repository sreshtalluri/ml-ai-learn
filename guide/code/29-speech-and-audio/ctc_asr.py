# Adapted from AI Engineering from Scratch (MIT), phases/06-speech-and-audio/04-speech-recognition-asr
"""CTC from scratch: greedy decoding, prefix beam search, the forward algorithm (CTC loss), and WER.

Frame posteriors are SYNTHETIC (what an acoustic encoder might output for "hello world"),
built with a fixed seed. No audio, no model weights, no downloads.
Unlike the upstream sketch, the beam search here tracks blank / non-blank prefix
probabilities, so it handles double letters ("ll") correctly.

Run from guide/:  uv run code/29-speech-and-audio/ctc_asr.py
"""
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import plt, save  # noqa: E402

VOCAB = ["_"] + list("abcdefghijklmnopqrstuvwxyz ")   # index 0 is the CTC blank
BLANK = 0


# ---------- synthetic encoder output ----------
def synthetic_posteriors(text, frames_per_char=3, peak=0.8, seed=0):
    rng = np.random.default_rng(seed)
    path = [BLANK]
    for i, ch in enumerate(text):
        if i and text[i - 1] == ch:
            path.append(BLANK)                       # CTC needs a blank between repeated letters
        path += [VOCAB.index(ch)] * frames_per_char
        path.append(BLANK)
    probs = rng.dirichlet(np.ones(len(VOCAB)) * 0.3, size=len(path)) * (1 - peak)
    probs[np.arange(len(path)), path] += peak        # each row sums to 1
    return probs                                     # [T, V]


# ---------- 1. greedy decoding ----------
def collapse(ids):
    out, prev = [], None
    for i in ids:
        if i != prev and i != BLANK:
            out.append(i)
        prev = i
    return out


def greedy(probs):
    return "".join(VOCAB[i] for i in collapse(probs.argmax(1)))


# ---------- 2. prefix beam search ----------
def prefix_beam(probs, beam=8):
    beams = {(): (1.0, 0.0)}                        # prefix -> (P ending in blank, P ending in non-blank)
    for p in probs:
        nxt = defaultdict(lambda: [0.0, 0.0])
        for prefix, (pb, pnb) in beams.items():
            nxt[prefix][0] += (pb + pnb) * p[BLANK]                    # emit blank: prefix unchanged
            for c in range(1, len(p)):
                if prefix and prefix[-1] == c:
                    nxt[prefix][1] += pnb * p[c]                       # repeat without blank: merges
                    nxt[prefix + (c,)][1] += pb * p[c]                 # blank in between: new letter
                else:
                    nxt[prefix + (c,)][1] += (pb + pnb) * p[c]
        beams = dict(sorted(nxt.items(), key=lambda kv: -sum(kv[1]))[:beam])
    best = max(beams.items(), key=lambda kv: sum(kv[1]))
    return "".join(VOCAB[i] for i in best[0]), sum(best[1])


# ---------- 3. CTC forward algorithm: P(y | x) summed over all alignments ----------
def ctc_prob(probs, target):
    ext = [BLANK]
    for c in target:
        ext += [c, BLANK]                             # "ab" -> _ a _ b _ ; length 2U + 1
    S, T = len(ext), len(probs)
    alpha = np.zeros((T, S))
    alpha[0, 0] = probs[0, ext[0]]
    alpha[0, 1] = probs[0, ext[1]] if S > 1 else 0.0
    for t in range(1, T):
        for s in range(S):
            a = alpha[t - 1, s] + (alpha[t - 1, s - 1] if s >= 1 else 0.0)
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                a += alpha[t - 1, s - 2]              # skip the blank between different letters
            alpha[t, s] = a * probs[t, ext[s]]
    return alpha[-1, -1] + (alpha[-1, -2] if S > 1 else 0.0), alpha


def brute_force(probs, target):
    T, V = probs.shape
    total = 0.0
    for path in itertools.product(range(V), repeat=T):
        if collapse(path) == list(target):
            total += math.prod(probs[t, k] for t, k in enumerate(path))
    return total


# ---------- 4. word error rate ----------
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(len(r) + 1), np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    i, j, S, D, I = len(r), len(h), 0, 0, 0       # backtrace to count each error type
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            S += r[i - 1] != h[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            D, i = D + 1, i - 1
        else:
            I, j = I + 1, j - 1
    return d[-1, -1] / len(r), S, D, I


def main():
    print("=== Tiny CTC example: T = 3 frames, vocab {_, a}, target 'a' ===")
    tiny = np.array([[0.4, 0.6], [0.5, 0.5], [0.7, 0.3]])
    for path in itertools.product([0, 1], repeat=3):
        s = "".join("_a"[k] for k in path)
        pr = math.prod(tiny[t, k] for t, k in enumerate(path))
        print(f"  {s}  p = {pr:.3f}  collapses to {''.join('_a'[k] for k in collapse(path))!r}")
    p_tiny, _ = ctc_prob(tiny, [1])
    print(f"  forward algorithm P('a') = {p_tiny:.3f}; brute force = {brute_force(tiny, [1]):.3f}; "
          f"CTC loss = -ln P = {-math.log(p_tiny):.4f}")

    rng = np.random.default_rng(3)
    rand = rng.dirichlet(np.ones(4), size=6)
    fwd, brute = ctc_prob(rand, [1, 2, 2])[0], brute_force(rand, [1, 2, 2])
    print(f"  random check, T = 6, V = 4, target 'abb': forward {fwd:.6f} vs brute force {brute:.6f}")
    assert abs(fwd - brute) < 1e-12

    print("\n=== Greedy vs beam: two frames, P(_) = 0.6, P(a) = 0.4 each ===")
    two = np.array([[0.6, 0.4], [0.6, 0.4]])
    print(f"  greedy best path '__' -> '' with p = {0.6 * 0.6:.2f}")
    print(f"  P('a') = aa + a_ + _a = {0.4 * 0.4:.2f} + {0.4 * 0.6:.2f} + {0.6 * 0.4:.2f} = {ctc_prob(two, [1])[0]:.2f}")
    lab, pr = prefix_beam(two)
    print(f"  greedy -> {greedy(two)!r}; prefix beam -> {VOCAB[1] if lab else ''!r} with p = {pr:.2f}")

    # beam (wide) must find the most probable labeling that brute force finds
    for seed in range(20):
        r = np.random.default_rng(seed).dirichlet(np.ones(3) * 0.7, size=5)
        totals = defaultdict(float)
        for path in itertools.product(range(3), repeat=5):
            totals[tuple(collapse(path))] += math.prod(r[t, k] for t, k in enumerate(path))
        best = max(totals, key=totals.get)
        assert prefix_beam(r, beam=50)[0] == "".join(VOCAB[i] for i in best)
    print("  check: prefix beam (width 50) matches brute-force best labeling on 20 random 5-frame cases")

    print("\n=== 'hello world' from synthetic posteriors ===")
    probs = synthetic_posteriors("hello world")
    print(f"  posteriors shape {probs.shape} (frames x vocab)")
    print(f"  greedy: {greedy(probs)!r}")
    print(f"  beam:   {prefix_beam(probs)[0]!r}")
    p_true, alpha = ctc_prob(probs, [VOCAB.index(c) for c in "hello world"])
    print(f"  CTC loss for the true transcript: {-math.log(p_true):.3f}")

    print("\n=== WER ===")
    ref = "turn on the kitchen lights"
    for hyp in ["turn on the kitchen lights", "turn on the kitchen light", "turn the kitchen light on",
                "turn on on the big kitchen lights", "um so turn uh on the the kitchen lights please"]:
        w, S, D, I = wer(ref, hyp)
        print(f"  WER {w:.2f}  S={S} D={D} I={I}  {hyp!r}")

    # ---------- figure ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    used = [0] + sorted({VOCAB.index(c) for c in "hello world"})
    ax1.imshow(probs[:, used].T, aspect="auto", cmap="Blues", origin="lower")
    ax1.plot(np.arange(len(probs)), [used.index(i) if i in used else np.nan for i in probs.argmax(1)],
             "o", ms=3, color="#ea580c", label="greedy argmax per frame")
    ax1.set_yticks(range(len(used)), ["blank" if i == 0 else repr(VOCAB[i]) for i in used], fontsize=8)
    ax1.set_xlabel("frame t")
    ax1.set_title("Synthetic frame posteriors for 'hello world'")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(False)
    tgt = [VOCAB.index(c) for c in "hello"]
    _, a = ctc_prob(probs[:20], tgt)
    labels = ["_"]
    for c in "hello":
        labels += [c, "_"]
    ax2.imshow(np.log10(a.T + 1e-30), aspect="auto", cmap="magma", origin="lower", vmin=-12)
    ax2.set_yticks(range(len(labels)), labels, fontsize=8)
    ax2.set_xlabel("frame t")
    ax2.set_ylabel("position in _h_e_l_l_o_")
    ax2.set_title("Forward variables (log10 alpha), first 20 frames")
    ax2.grid(False)
    fig.tight_layout()
    save(fig, "speech-recognition-asr")


if __name__ == "__main__":
    main()
