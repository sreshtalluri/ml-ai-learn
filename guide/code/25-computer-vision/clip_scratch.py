# Adapted from AI Engineering from Scratch (MIT), phases/12-multimodal-ai/02-clip-contrastive-pretraining
"""CLIP's contrastive objective in NumPy: cosine similarity matrix, temperature, row softmax (image -> text),
column softmax (text -> image), the symmetric InfoNCE loss, its gradient, a few training steps that pull
matched pairs together, zero-shot classification, and the SigLIP sigmoid loss for comparison.

The "embeddings" are SYNTHETIC hand-made 5-D vectors standing in for encoder outputs (the same numbers as
the clip-similarity lab on the website). Deterministic.
Run from guide/:  uv run code/25-computer-vision/clip_scratch.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import plt, save  # noqa: E402

# feature axes (for reading the numbers only): [animal, feline, vehicle, food, plant]
NAMES = ["cat", "dog", "car", "pizza", "tree"]
IMAGES = np.array([[0.8, 0.2, 0, 0, 0.1], [0.9, 0.1, 0, 0.1, 0.1], [0, 0, 1, 0, 0.2], [0.1, 0, 0, 1, 0.1], [0, 0, 0.2, 0.1, 1]])
CAPTIONS = np.array([[0.7, 0.5, 0, 0, 0], [0.9, 0, 0, 0.1, 0.2], [0, 0, 0.9, 0.1, 0.3], [0, 0, 0.1, 0.9, 0], [0.1, 0, 0, 0, 0.9]])


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def softmax(z, axis):
    z = z - z.max(axis=axis, keepdims=True)           # stable: largest logit becomes 0
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ---------- 1. Similarity matrix and the two softmaxes ----------
def clip_logits(img, txt, tau):
    """img, txt: [N, D] raw embeddings. Returns [N, N] logits S[i, j] = cos(img_i, txt_j) / tau."""
    return normalize(img) @ normalize(txt).T / tau


def clip_loss(S):
    p_i2t = softmax(S, axis=1)                         # row i: which caption goes with image i?
    p_t2i = softmax(S, axis=0)                         # column j: which image goes with caption j?
    loss_i2t = -np.log(np.diag(p_i2t)).mean()         # correct answer is always the diagonal
    loss_t2i = -np.log(np.diag(p_t2i)).mean()
    return (loss_i2t + loss_t2i) / 2, loss_i2t, loss_t2i, p_i2t, p_t2i


# ---------- 2. One training step on the embeddings themselves ----------
def train_step(img, txt, tau=0.1, lr=0.05):
    """Gradient step on unit vectors, then re-normalise (projected gradient on the sphere)."""
    I, T = normalize(img), normalize(txt)
    n = len(I)
    _, _, _, p, q = clip_loss(I @ T.T / tau)
    dS = ((p - np.eye(n)) + (q - np.eye(n))) / (2 * n)  # dLoss/dS: softmax minus one-hot, both directions
    dI = dS @ T / tau                                  # chain rule through S = I T^T / tau
    dT = dS.T @ I / tau
    return normalize(I - lr * dI), normalize(T - lr * dT)


# ---------- 3. SigLIP: an independent sigmoid per pair, no softmax over the batch ----------
def siglip_loss(cos, t=10.0, b=-10.0):
    """cos: [N, N] cosines. Positive pairs on the diagonal, negatives everywhere else."""
    z = t * cos + b
    sign = 2 * np.eye(len(cos)) - 1                    # +1 on the diagonal, -1 elsewhere
    return float(np.log1p(np.exp(-sign * z)).sum() / len(cos))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("== worked example: 3 pairs, 2-D unit vectors, tau = 0.1 ==")
    wi = np.array([[1, 0], [0.6, 0.8], [0, 1]])
    wt = np.array([[0.8, 0.6], [0.6, 0.8], [0, 1]])
    cos = normalize(wi) @ normalize(wt).T
    print("cosine matrix C (rows = images, cols = captions):\n", cos)
    loss, li, lt, p, q = clip_loss(cos / 0.1)
    print("row softmax (image -> text):\n", p)
    print("column softmax (text -> image):\n", q)
    print(f"loss_i2t = {li:.4f}   loss_t2i = {lt:.4f}   symmetric loss = {loss:.4f}")
    print(f"chance level ln 3 = {np.log(3):.4f}")

    print("\n== temperature on the worked example ==")
    for tau in (1.0, 0.1, 0.01):
        print(f"tau {tau:<4}: image 2 row = {softmax(cos / tau, 1)[1].round(4).tolist()}  loss = {clip_loss(cos / tau)[0]:.4f}")

    print("\n== lab data: 5 SYNTHETIC image/caption pairs, tau = 0.1 ==")
    S = clip_logits(IMAGES, CAPTIONS, 0.1)
    print("cosine matrix:\n", S * 0.1)
    loss, li, lt, p, q = clip_loss(S)
    print(f"loss = {loss:.4f} (i2t {li:.4f}, t2i {lt:.4f})")
    pred = p.argmax(1)
    print("zero-shot (best caption per image):", [f"{NAMES[i]}->{NAMES[j]}" for i, j in enumerate(pred)],
          f"accuracy {np.mean(pred == np.arange(5)):.0%}")

    print("\n== training: nudge matched pairs together ==")
    I, T = IMAGES, CAPTIONS
    history = [loss]
    for step in range(1, 31):
        I, T = train_step(I, T)
        history.append(clip_loss(normalize(I) @ normalize(T).T / 0.1)[0])
        if step in (1, 5, 10, 30):
            P = softmax(normalize(I) @ normalize(T).T / 0.1, 1)
            print(f"step {step:>2}: loss {history[-1]:.4f}  P(cat image -> 'cat' caption) = {P[0, 0]:.3f}  "
                  f"accuracy {np.mean(P.argmax(1) == np.arange(5)):.0%}")

    print("\n== SigLIP sigmoid loss on the same cosines (t = 10, b = -10) ==")
    print(f"before training {siglip_loss(normalize(IMAGES) @ normalize(CAPTIONS).T):.4f}   "
          f"after 30 steps {siglip_loss(normalize(I) @ normalize(T).T):.4f}")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    labels = [f"'{n}'" for n in NAMES]
    for ax, m, title in [(axes[0], normalize(IMAGES) @ normalize(CAPTIONS).T, "cosine similarity (before)"),
                         (axes[1], softmax(clip_logits(IMAGES, CAPTIONS, 0.1), 1), "row softmax, τ = 0.1 (image → text)")]:
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else "black")
        ax.set(xticks=range(5), yticks=range(5), title=title)
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.set_yticklabels([f"{n} image" for n in NAMES], fontsize=8)
        ax.grid(False)
    axes[1].set_yticklabels([])
    fig.subplots_adjust(wspace=0.35)
    axes[2].plot(history, marker=".")
    axes[2].set(xlabel="training step", ylabel="symmetric contrastive loss", title="Training pulls matched pairs together")
    save(fig, "clip-contrastive-pretraining")
