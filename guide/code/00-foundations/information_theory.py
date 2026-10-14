# Adapted from AI Engineering from Scratch (MIT), phases/01-math-foundations/09-information-theory
"""Surprise, entropy, cross-entropy, KL divergence, perplexity, and mutual information from scratch.

Every distribution here is a small hand-picked example (synthetic). Natural log (nats) unless a
function is called with base=2 (bits).
Run from guide/:  uv run code/00-foundations/information_theory.py
"""
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402


# --- Surprise and entropy -------------------------------------------------
def surprise(p, base=math.e):
    """Information content of an event with probability p: -log p."""
    return math.log(1 / p) / math.log(base)   # = -log p, written so p = 1 gives +0.0


def entropy(p, base=math.e):
    """Average surprise. Terms with p = 0 contribute 0 (the limit of p log p)."""
    return sum(pi * surprise(pi, base) for pi in p if pi > 0)


# --- Cross-entropy and KL ---------------------------------------------------
def cross_entropy(p, q, base=math.e):
    """Average surprise when events come from p but you score them with q."""
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi == 0:
                return math.inf          # q says "impossible" for something that happens
            total += pi * surprise(qi, base)
    return total


def kl_divergence(p, q, base=math.e):
    """Extra surprise from using q instead of p: H(p, q) - H(p). Never negative."""
    return cross_entropy(p, q, base) - entropy(p, base)


# --- Classification loss -----------------------------------------------------
def softmax(logits):
    m = max(logits)                      # subtract the max so exp never overflows
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def cross_entropy_loss(logits, target):
    """One-hot target: H(p, q) collapses to -log q[target]."""
    return -math.log(softmax(logits)[target])


def perplexity(true_token_probs):
    """exp of the average negative log-probability of the observed tokens."""
    nll = sum(-math.log(p) for p in true_token_probs) / len(true_token_probs)
    return nll, math.exp(nll)


# --- Mutual information --------------------------------------------------------
def mutual_information(joint, base=math.e):
    """I(X;Y) = sum p(x,y) log[p(x,y) / (p(x) p(y))] for a joint table."""
    px = [sum(row) for row in joint]
    py = [sum(col) for col in zip(*joint)]
    mi = 0.0
    for i, row in enumerate(joint):
        for j, pxy in enumerate(row):
            if pxy > 0:
                mi += pxy * math.log(pxy / (px[i] * py[j])) / math.log(base)
    return mi


def main():
    print("=== Surprise (bits) ===")
    for name, p in [("fair coin heads", 0.5), ("roll a 6", 1 / 6), ("1-in-1000", 0.001), ("certain", 1.0)]:
        print(f"{name:16s} p={p:.3f}  surprise={surprise(p, 2):.3f} bits")

    print("\n=== Entropy (bits) ===")
    print(f"fair coin   {entropy([0.5, 0.5], 2):.4f}")
    print(f"0.99 coin   {entropy([0.99, 0.01], 2):.4f}")
    print(f"fair die    {entropy([1/6] * 6, 2):.4f}")

    P = [0.7, 0.2, 0.1]
    good, bad = [0.6, 0.25, 0.15], [0.1, 0.1, 0.8]
    print("\n=== Worked example: P = [0.7, 0.2, 0.1] (nats) ===")
    print(f"H(P)            = {entropy(P):.4f}")
    for name, Q in [("good", good), ("bad", bad)]:
        print(f"H(P, {name:4s})    = {cross_entropy(P, Q):.4f}   KL(P||{name}) = {kl_divergence(P, Q):.4f}")
    print(f"KL(good||P)     = {kl_divergence(good, P):.4f}   (not equal to KL(P||good): KL is asymmetric)")

    logits, target = [2.0, 1.0, 0.1], 0
    probs = softmax(logits)
    loss = cross_entropy_loss(logits, target)
    print("\n=== Classification loss ===")
    print(f"logits {logits} -> softmax {[round(p, 4) for p in probs]}")
    print(f"loss = -ln {probs[target]:.4f} = {loss:.4f} nats = {loss / math.log(2):.4f} bits")

    random.seed(0)
    labels = [random.randrange(3) for _ in range(1000)]
    rows = [[random.gauss(0, 1) for _ in range(3)] for _ in range(1000)]
    ce = sum(cross_entropy_loss(z, y) for z, y in zip(rows, labels)) / 1000
    likelihood_log = sum(math.log(softmax(z)[y]) for z, y in zip(rows, labels))
    print(f"\nmean CE over 1000 random examples = {ce:.6f}; -log-likelihood / n = {-likelihood_log / 1000:.6f}")

    nll, ppl = perplexity([0.5, 0.25, 0.125, 0.5])
    print("\n=== Perplexity ===")
    print(f"true-token probs [0.5, 0.25, 0.125, 0.5]: mean NLL = {nll:.4f} nats, perplexity = {ppl:.4f}")
    print(f"uniform over 50 tokens: perplexity = {perplexity([1/50] * 4)[1]:.1f}")

    eps, k = 0.1, 4
    soft = [eps / k] * k
    soft[2] += 1 - eps
    print(f"\nlabel smoothing eps=0.1, 4 classes: target {soft}, entropy {entropy(soft):.4f} nats (one-hot: 0)")

    print("\n=== Mutual information (bits) ===")
    print(f"independent [[.25,.25],[.25,.25]]: {mutual_information([[.25, .25], [.25, .25]], 2):.4f}")
    print(f"dependent   [[.45,.05],[.05,.45]]: {mutual_information([[.45, .05], [.05, .45]], 2):.4f}")

    # Library check: SciPy-free NumPy versions match the loops above.
    p, q = np.array(P), np.array(good)
    assert abs(-(p * np.log(q)).sum() - cross_entropy(P, good)) < 1e-12
    assert abs((p * np.log(p / q)).sum() - kl_divergence(P, good)) < 1e-12
    assert kl_divergence(P, bad) > kl_divergence(P, good) > 0

    # --- Figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.linspace(1e-4, 1 - 1e-4, 400)
    axes[0].plot(x, -(x * np.log2(x) + (1 - x) * np.log2(1 - x)), color=BLUE)
    axes[0].scatter([0.5, 0.99], [1.0, entropy([0.99, 0.01], 2)], color=ORANGE, zorder=3)
    axes[0].annotate("fair coin: 1 bit", (0.5, 1.0), (0.55, 0.82))
    axes[0].annotate("0.99 coin: 0.08 bits", (0.99, 0.08), (0.42, 0.18))
    axes[0].set(title="Entropy of a coin", xlabel="P(heads)", ylabel="H (bits)")

    axes[1].plot(x, -np.log(x), color=TEAL)
    for qv in (0.9, 0.5, 0.1):
        axes[1].scatter(qv, -math.log(qv), color=ORANGE, zorder=3)
        dx = -0.32 if qv > 0.8 else 0.03
        axes[1].annotate(f"q={qv}: {-math.log(qv):.3f}", (qv, -math.log(qv)), (qv + dx, -math.log(qv) + 0.4))
    axes[1].set(title="Loss = -ln(prob. of true class)", xlabel="q (true class)", ylabel="loss (nats)", ylim=(0, 5))

    names = ["P itself", "good Q", "bad Q"]
    hs = [entropy(P)] * 3
    kls = [0.0, kl_divergence(P, good), kl_divergence(P, bad)]
    axes[2].bar(names, hs, color=GRAY, label="H(P), fixed")
    axes[2].bar(names, kls, bottom=hs, color=ORANGE, label="KL(P||Q), extra")
    for i in range(3):
        axes[2].text(i, hs[i] + kls[i] + 0.04, f"{hs[i] + kls[i]:.3f}", ha="center")
    axes[2].set(title="Cross-entropy = entropy + KL", ylabel="nats")
    axes[2].legend(loc="upper left")
    save(fig, "information-theory")


if __name__ == "__main__":
    main()
