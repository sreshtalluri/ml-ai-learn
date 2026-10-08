"""Transformer block parameter counts, tokenization, next-token cross-entropy/perplexity, decoding, and LoRA arithmetic.

Run from guide/:  uv run code/15-llms/llms.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Transformer block parameters (GPT-2 small sizes)
# ---------------------------------------------------------------------------
d, ff, layers, vocab, ctx = 768, 3072, 12, 50257, 1024
attn = 4 * d * d + 4 * d
mlp = 2 * d * ff + ff + d
ln = 2 * 2 * d
block = attn + mlp + ln
emb = vocab * d + ctx * d
total = layers * block + emb + 2 * d
print(f"per block: attention {attn:,}, MLP {mlp:,}, layer norms {ln:,} -> {block:,}")
print(f"12 blocks {layers * block:,} + embeddings {emb:,} + final LN = {total:,} (~124M, GPT-2 small)")

# ---------------------------------------------------------------------------
# 2. Byte-pair encoding: a few merges on a toy corpus
# ---------------------------------------------------------------------------
corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
words = Counter(tuple(w) + ("</w>",) for w in corpus)


def merge_step(words):
    pairs = Counter()
    for w, c in words.items():
        for a, b in zip(w, w[1:]):
            pairs[(a, b)] += c
    best = max(pairs, key=pairs.get)
    merged = Counter()
    for w, c in words.items():
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                out.append(w[i] + w[i + 1])
                i += 2
            else:
                out.append(w[i])
                i += 1
        merged[tuple(out)] += c
    return best, pairs[best], merged


print("\nBPE merges:")
for _ in range(5):
    best, count, words = merge_step(words)
    print(f"  merge {best} (count {count}) -> {sorted(words, key=lambda w: -words[w])[:2]}")

text = "Tokenization splits text into subword units; unbelievable!"
print("\nword-level tokens:", re.findall(r"\w+|[^\w\s]", text))

# ---------------------------------------------------------------------------
# 3. Next-token cross-entropy and perplexity
# ---------------------------------------------------------------------------
p_true = np.array([0.6, 0.25, 0.9, 0.05])
ce = -np.log(p_true)
print(f"\nper-token losses {ce.round(4)}, mean {ce.mean():.4f}, perplexity e^mean = {np.exp(ce.mean()):.3f}")
print(f"uniform over 50,000 tokens: loss ln(50000) = {np.log(50000):.3f}, perplexity 50000")

# ---------------------------------------------------------------------------
# 4. Decoding: temperature, top-k, top-p on the same logits as the web lab
# ---------------------------------------------------------------------------
tokens = ["Paris", "a", "the", "located", "known", "Lyon", "not", "beautiful", "France", "Marseille"]
logits = np.array([6.1, 3.2, 2.9, 2.4, 2.1, 1.6, 1.2, 1.0, 0.6, 0.4])


def softmax(z, T=1.0):
    e = np.exp((z - z.max()) / T)
    return e / e.sum()


for T in (0.5, 1.0, 1.5):
    p = softmax(logits, T)
    print(f"T={T}: P(Paris)={p[0]:.3f}, entropy={-(p * np.log2(p)).sum():.2f} bits")
p = softmax(logits)
order = np.argsort(-p)
cum = np.cumsum(p[order])
nucleus = order[: np.searchsorted(cum, 0.9) + 1]
print(f"top-p 0.9 keeps {[tokens[i] for i in nucleus]} (cumulative {cum[len(nucleus) - 1]:.3f})")
z2 = np.array([2.0, 1.0, 0.0])
print(f"softmax([2,1,0]) at T=1: {softmax(z2).round(4)}, T=0.5: {softmax(z2, 0.5).round(4)}, T=2: {softmax(z2, 2).round(4)}")

# ---------------------------------------------------------------------------
# 5. LoRA parameter arithmetic
# ---------------------------------------------------------------------------
d_in = d_out = 4096
r = 8
print(f"\nLoRA on a {d_in}x{d_out} matrix: full {d_in * d_out:,} params vs rank {r}: {r * (d_in + d_out):,} ({r * (d_in + d_out) / (d_in * d_out):.2%})")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
ax = axes[0]
for T, col in [(0.5, BLUE), (1.0, TEAL), (1.5, ORANGE)]:
    ax.plot(range(10), softmax(logits, T), "-o", color=col, ms=4, label=f"T = {T}")
ax.set_xticks(range(10), tokens, rotation=45, ha="right")
ax.set(title="Temperature reshapes the next-token distribution", ylabel="probability")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
ax.bar(range(10), p[order], color=[PURPLE if i < len(nucleus) else GRAY for i in range(10)])
ax.plot(range(10), cum, "-o", color=ORANGE, ms=3, label="cumulative")
ax.axhline(0.9, color=ORANGE, ls="--", lw=1)
ax.set_xticks(range(10), [tokens[i] for i in order], rotation=45, ha="right")
ax.set(title=f"Top-p = 0.9 keeps {len(nucleus)} tokens (purple)", ylabel="probability")
ax.legend(frameon=False, fontsize=8)
ax = axes[2]
probs = np.linspace(0.01, 1, 200)
ax.plot(probs, -np.log(probs), color=BLUE, lw=2)
for pt in p_true:
    ax.plot(pt, -np.log(pt), "o", color=ORANGE)
ax.set(title="Token loss = −ln P(actual next token)", xlabel="probability assigned to the true token", ylabel="cross-entropy")
for a in axes:
    a.title.set_fontsize(10)
save(fig, "decoding")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
ax = axes[0]
names = ["attention\n(Q,K,V,O)", "feed-forward\n(MLP)", "layer norms"]
vals = [attn, mlp, ln]
ax.bar(names, [v / 1e6 for v in vals], color=[BLUE, PURPLE, GRAY])
for i, v in enumerate(vals):
    ax.text(i, v / 1e6 + 0.1, f"{v / 1e6:.2f}M", ha="center", fontsize=9)
ax.set(title="Parameters in one GPT-2-small block (d = 768)", ylabel="millions")
ax = axes[1]
seq = np.array([128, 256, 512, 1024, 2048, 4096])
ax.plot(seq, seq**2 / seq[0] ** 2, "-o", color=ORANGE, label="attention scores (∝ n²)")
ax.plot(seq, seq / seq[0], "-o", color=TEAL, label="feed-forward compute (∝ n)")
ax.set(xscale="log", yscale="log", title="Cost growth with context length (relative to n = 128)", xlabel="sequence length n", ylabel="relative cost")
ax.legend(frameon=False, fontsize=8)
for a in axes:
    a.title.set_fontsize(10)
save(fig, "transformer-architecture")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
ax = axes[0]
samples = ["hello world", "Hello, World!", "naïve café", "internationalization", "ML 2026 → ship", "def f(x): return x**2"]
chars = [len(s) for s in samples]
utf8 = [len(s.encode("utf-8")) for s in samples]
y = np.arange(len(samples))
ax.barh(y - 0.2, chars, height=0.4, color=BLUE, label="characters")
ax.barh(y + 0.2, utf8, height=0.4, color=ORANGE, label="UTF-8 bytes (byte-level BPE starts here)")
ax.set_yticks(y, samples)
ax.set(title="The same text has different lengths in different units", xlabel="count")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
ctx_tokens = np.arange(1, 9)
ax.bar(ctx_tokens, -np.log(np.array([0.05, 0.2, 0.35, 0.5, 0.62, 0.7, 0.8, 0.85])), color=PURPLE)
ax.set(title="Illustrative: loss on a token falls as context grows", xlabel="tokens of context", ylabel="−ln P(next token)")
for a in axes:
    a.title.set_fontsize(10)
save(fig, "tokenization-and-pretraining")

fig, ax = plt.subplots(figsize=(8, 3.6))
methods = ["Prompting", "RAG", "LoRA / PEFT", "Full fine-tuning"]
trained = [0, 0, r * (d_in + d_out) * 4 * 32, 7e9]
ax.barh(methods, [max(t, 1) for t in trained], color=[GRAY, TEAL, PURPLE, ORANGE])
ax.set(xscale="log", title="Parameters updated to adapt a ~7B model (illustrative)", xlabel="trainable parameters (log scale; 1 = none)")
for i, t in enumerate(trained):
    ax.text(max(t, 1) * 1.3, i, "none (changes context only)" if t == 0 else f"{t:,.0f}", va="center", fontsize=9)
save(fig, "adapting-llms")
