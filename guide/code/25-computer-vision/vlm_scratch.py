# Adapted from AI Engineering from Scratch (MIT), phases/04-computer-vision/25-vision-language-models
"""The ViT-projector-LLM pattern in NumPy: image token counts, a 2-layer MLP projector and its size,
splicing projected image tokens into the text sequence at <image> placeholders, token compression
and high-resolution tiling arithmetic, and a stage-1 "alignment" run that trains only the projector.

Vision features and word embeddings are SYNTHETIC random vectors (frozen), standing in for a frozen
vision encoder and a frozen LLM. Deterministic.
Run from guide/:  uv run code/25-computer-vision/vlm_scratch.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, plt, save  # noqa: E402

rng = np.random.default_rng(0)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


# ---------- 1. The projector: vision width -> LLM width ----------
def projector_params(d_vit, d_llm, hidden=None):
    hidden = hidden or d_llm
    return d_vit * hidden + hidden + hidden * d_llm + d_llm


def project(v, P):
    """v: [N_img, d_vit] patch features -> [N_img, d_llm] 'soft tokens' the LLM reads like words."""
    return gelu(v @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]


# ---------- 2. Splice image tokens into the text sequence ----------
def merge(text_ids, embed_table, image_tokens, image_id):
    """Replace each <image> placeholder id with the next projected image token."""
    out = embed_table[text_ids].copy()                         # [M, d_llm] ordinary word embeddings
    slots = np.flatnonzero(text_ids == image_id)
    if len(slots) != len(image_tokens):                        # most common VLM porting bug: say so loudly
        raise ValueError(f"{len(slots)} <image> placeholders but {len(image_tokens)} image tokens")
    out[slots] = image_tokens
    return out


# ---------- 3. Stage-1 alignment: train ONLY the projector ----------
def align(n_classes=5, d_vit=32, d_llm=48, hidden=64, steps=300, lr=0.01):
    words = rng.standard_normal((n_classes, d_llm))            # frozen LLM embeddings of the class words
    words /= np.linalg.norm(words, axis=1, keepdims=True)
    protos = rng.standard_normal((n_classes, d_vit))           # frozen vision encoder: one prototype per class
    y = np.repeat(np.arange(n_classes), 40)
    feats = protos[y] + 0.5 * rng.standard_normal((len(y), d_vit))   # SYNTHETIC pooled image features
    P = {"W1": rng.standard_normal((d_vit, hidden)) / np.sqrt(d_vit), "b1": np.zeros(hidden),
         "W2": rng.standard_normal((hidden, d_llm)) / np.sqrt(hidden), "b2": np.zeros(d_llm)}

    def accuracy():
        out = project(feats, P)
        return np.mean((out @ words.T).argmax(1) == y)         # nearest word embedding = "what the LLM sees"

    hist = [(0, accuracy())]
    for step in range(1, steps + 1):                           # MSE to the caption word embedding, plain GD
        h_pre = feats @ P["W1"] + P["b1"]
        h = gelu(h_pre)
        err = (h @ P["W2"] + P["b2"]) - words[y]                # [n, d_llm]
        g_out = 2 * err / len(y)
        dW2, db2 = h.T @ g_out, g_out.sum(0)
        t = np.tanh(np.sqrt(2 / np.pi) * (h_pre + 0.044715 * h_pre ** 3))
        dgelu = 0.5 * (1 + t) + 0.5 * h_pre * (1 - t ** 2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * h_pre ** 2)
        g_h = (g_out @ P["W2"].T) * dgelu
        dW1, db1 = feats.T @ g_h, g_h.sum(0)
        for k, g in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
            P[k] -= lr * g
        if step % 10 == 0:
            hist.append((step, accuracy()))
    return hist


if __name__ == "__main__":
    print("== image tokens ==")
    for name, img, patch in [("ViT-L/14 at 224", 224, 14), ("ViT-L/14 at 336", 336, 14), ("ViT-B/16 at 224", 224, 16)]:
        print(f"{name}: ({img}/{patch})^2 = {(img // patch) ** 2} image tokens")

    print("\n== projector size (2-layer MLP, 1024 -> 4096 -> 4096) ==")
    print(f"params = 1024*4096 + 4096 + 4096*4096 + 4096 = {projector_params(1024, 4096):,}")

    print("\n== merging a prompt ==")
    d_vit, d_llm, IMG = 8, 12, 99
    P = {"W1": rng.standard_normal((d_vit, d_llm)), "b1": np.zeros(d_llm),
         "W2": rng.standard_normal((d_llm, d_llm)), "b2": np.zeros(d_llm)}
    vision = rng.standard_normal((4, d_vit))                   # 4 patch features from a tiny "ViT"
    table = rng.standard_normal((100, d_llm))                  # LLM embedding table, vocab of 100
    ids = np.array([1, 7, IMG, IMG, IMG, IMG, 23, 42, 5])      # "<s> USER: <image>x4 what is this ?"
    seq = merge(ids, table, project(vision, P), IMG)
    print(f"text ids {ids.tolist()} -> sequence {seq.shape}; rows 2..5 are projected image tokens")
    try:
        merge(ids[:-1][:5], table, project(vision, P), IMG)
    except ValueError as e:
        print(f"mismatch caught: {e}")

    print("\n== token budget ==")
    base = (336 // 14) ** 2
    print(f"one 336px view: {base} tokens; 2x2 token merge (pixel shuffle) -> {base // 4}")
    print(f"672x672 as 4 tiles of 336 + 1 thumbnail: 5 x {base} = {5 * base} tokens")
    print(f"a 10-image prompt at 576 tokens each: {10 * base:,} tokens before any text")

    print("\n== stage-1 alignment: frozen vision features, frozen word embeddings, train the projector ==")
    hist = align()
    for s, a in hist[:6] + [hist[-1]]:
        print(f"step {s:>3}: nearest-word accuracy {a:.0%}")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    labels = ["224 / p14", "336 / p14", "336 + 2x2\nmerge", "672 tiled\n(4 + thumb)"]
    vals = [(224 // 14) ** 2, base, base // 4, 5 * base]
    axes[0].bar(labels, vals, color=BLUE)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 40, str(v), ha="center", fontsize=9)
    axes[0].set(title="Image tokens one picture costs the LLM", ylabel="tokens")
    xs, ys = zip(*hist)
    axes[1].plot(xs, ys, "o-", color=ORANGE)
    axes[1].set(title="Stage 1: training only the projector (SYNTHETIC)", xlabel="step", ylabel="nearest-word accuracy", ylim=(0, 1.05))
    save(fig, "vision-language-models")
