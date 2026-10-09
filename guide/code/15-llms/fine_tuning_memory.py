"""Fine-tuning memory arithmetic: parameter counts, LoRA trainable parameters, and GPU memory
for full fine-tuning, LoRA, and QLoRA on a SYNTHETIC 7B-style decoder config.

Pure arithmetic, no model download. Numbers are estimates of training-state memory, not measurements.
Run from guide/:  uv run code/15-llms/fine_tuning_memory.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

GB = 1e9  # decimal gigabytes, like GPU spec sheets

# SYNTHETIC 7B-style config (Llama-like shape: gated MLP, untied output head, full multi-head attention).
CFG = dict(d=4096, layers=32, d_ff=11008, vocab=32000, kv_dim=4096)


def matrices(c):
    """(name, d_in, d_out) for every linear layer in one transformer block."""
    d, f, kv = c["d"], c["d_ff"], c["kv_dim"]
    return [("q", d, d), ("k", d, kv), ("v", d, kv), ("o", d, d), ("gate", d, f), ("up", d, f), ("down", f, d)]


def param_counts(c):
    block = sum(i * o for _, i, o in matrices(c)) * c["layers"]
    embed = 2 * c["vocab"] * c["d"]  # input embedding + untied output head (norm weights ignored, ~0.004%)
    return block, embed


def lora_params(c, r, targets):
    return r * sum(i + o for n, i, o in matrices(c) if n in targets) * c["layers"]


# Bytes per parameter, mixed precision with AdamW:
#   weights bf16 = 2, grads bf16 = 2, optimizer = fp32 master copy 4 + Adam m 4 + Adam v 4 = 12.
NF4_BYTES = 4.127 / 8  # 4-bit NF4 plus ~0.127 bits/param of (double-quantized) block constants


def activations(c, tokens, checkpointing):
    """Rough rule of thumb: ~34 bytes per token per hidden unit per layer in 16-bit (flash attention, no s^2 term).
    With checkpointing keep only each layer's 2-byte input plus one layer's full activations for recompute."""
    per_layer = 34 * tokens * c["d"]
    if checkpointing:
        return c["layers"] * 2 * tokens * c["d"] + per_layer
    return c["layers"] * per_layer


def memory(c, method, trainable, tokens=2048, checkpointing=True):
    block, embed = param_counts(c)
    n = block + embed
    if method == "full":
        w, g, o = 2 * n, 2 * n, 12 * n
    else:
        base = 2 * n if method == "lora" else NF4_BYTES * block + 2 * embed  # QLoRA keeps embeddings/head in 16-bit
        w, g, o = base + 2 * trainable, 2 * trainable, 12 * trainable
    return dict(weights=w, grads=g, optimizer=o, activations=activations(c, tokens, checkpointing))


if __name__ == "__main__":
    c = CFG
    block, embed = param_counts(c)
    n = block + embed
    print("SYNTHETIC 7B-style config:", c)
    for name, i, o in matrices(c):
        print(f"  {name:5s} {i:>6} x {o:<6} = {i * o:>12,}")
    print(f"block params     = {block:,}")
    print(f"embed + head     = {embed:,}")
    print(f"total params     = {n:,}  ({n / 1e9:.3f} B)")

    r, targets = 16, {"q", "k", "v", "o", "gate", "up", "down"}
    t = lora_params(c, r, targets)
    print(f"\nLoRA r={r} on all 7 linear layers")
    for name, i, o in matrices(c):
        print(f"  {name:5s} {r} x ({i} + {o}) = {r * (i + o):,}")
    print(f"  per layer = {t // c['layers']:,}; x {c['layers']} layers = {t:,} ({100 * t / n:.3f}% of total)")
    t_attn = lora_params(c, 8, {"q", "k", "v", "o"})
    print(f"LoRA r=8 on q,k,v,o only = {t_attn:,} ({100 * t_attn / n:.3f}%)  [the earlier lesson's example]")
    print(f"adapter file in bf16 = {2 * t / 1e6:.1f} MB")

    print("\nMemory (decimal GB), 2,048 tokens per micro-batch, gradient checkpointing on:")
    results = {}
    for m in ["full", "lora", "qlora"]:
        mem = memory(c, m, 0 if m == "full" else t)
        results[m] = mem
        parts = "  ".join(f"{k}={v / GB:6.2f}" for k, v in mem.items())
        print(f"  {m:6s} {parts}  total={sum(mem.values()) / GB:6.2f}")
    print(f"  activations without checkpointing = {activations(c, 2048, False) / GB:.2f} GB")

    print("\nDPO loss, beta = 0.1 (log-ratios are log pi_theta - log pi_ref, summed over response tokens):")
    beta = 0.1
    for lw, ll in [(0.0, 0.0), (2.0, -1.0)]:
        z = beta * (lw - ll)
        sig = 1 / (1 + np.exp(-z))
        print(f"  chosen {lw:+.1f}, rejected {ll:+.1f}: z = {z:.2f}, sigmoid = {sig:.6f}, loss = {-np.log(sig):.6f}")

    rewards = np.array([1.0, 0.0, 0.0, 1.0])
    adv = (rewards - rewards.mean()) / rewards.std()
    print(f"\nGRPO group of 4, verifier rewards {rewards.tolist()}: mean {rewards.mean()}, std {rewards.std()}, advantages {adv.tolist()}")

    # ---- figure ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    labels = {"full": "Full fine-tune", "lora": "LoRA (bf16 base)", "qlora": "QLoRA (NF4 base)"}
    colors = {"weights": BLUE, "grads": ORANGE, "optimizer": PURPLE, "activations": TEAL}
    for yi, m in enumerate(["qlora", "lora", "full"]):
        left = 0.0
        for k, v in results[m].items():
            a1.barh(yi, v / GB, left=left, color=colors[k], label=k if yi == 0 else None, height=0.55)
            left += v / GB
        a1.text(left + 1.5, yi, f"{left:.1f} GB", va="center", fontsize=9)
    for g in [24, 48, 80]:
        a1.axvline(g, color=GRAY, ls="--", lw=1)
        a1.text(g, 2.45, f"{g} GB GPU", ha="center", fontsize=8, color=GRAY)
    a1.set_yticks([0, 1, 2], [labels["qlora"], labels["lora"], labels["full"]])
    a1.set_xlabel("training memory (GB, estimate)")
    a1.set_xlim(0, 125)
    a1.set_ylim(-0.5, 2.7)
    a1.legend(loc="lower right", fontsize=8)
    a1.set_title("Training memory (7B-style, r = 16)")

    ranks = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    for tg, lab, col in [({"q", "k", "v", "o"}, "q, k, v, o", BLUE), (targets, "all 7 linear layers", ORANGE)]:
        a2.plot(ranks, [100 * lora_params(c, int(rr), tg) / n for rr in ranks], "o-", color=col, label=lab)
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xlabel("LoRA rank r")
    a2.set_ylabel("trainable params (share of model)")
    a2.set_title("LoRA trainable fraction vs rank")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
    a2.legend(fontsize=8)
    save(fig, "fine-tuning-in-practice")
