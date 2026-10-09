"""LLM inference arithmetic: KV-cache size, weight memory, decode speed limits, arithmetic intensity,
speculative-decoding speedup, mixture-of-experts memory vs compute, and latency metrics.

All model shapes and accelerator specs are SYNTHETIC ("8B-style", "80 GB / 3.0 TB/s"), chosen to be in the
range of current hardware, not measurements of any product. GB means 10^9 bytes.
Run from guide/:  uv run code/18-llm-inference/llm_inference.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

GB = 1e9

# Synthetic 8B-style model with grouped-query attention.
L, N_HEADS, N_KV, D_HEAD, PARAMS = 32, 32, 8, 128, 8e9
# Synthetic accelerator: 80 GB memory, 3.0 TB/s bandwidth, 1000 TFLOP/s dense 16-bit peak.
MEM, BW, PEAK = 80 * GB, 3.0e12, 1000e12


def kv_bytes_per_token(layers, n_kv, d_head, nbytes):
    """K and V (the 2), for every layer, every KV head, every head dimension."""
    return 2 * layers * n_kv * d_head * nbytes


def kv_cache_bytes(layers, n_kv, d_head, nbytes, tokens, batch):
    return kv_bytes_per_token(layers, n_kv, d_head, nbytes) * tokens * batch


def decode_tokens_per_s(weight_bytes, kv_bytes, bandwidth, batch):
    """Upper bound: every decode step reads all weights once plus every sequence's KV cache.
    Returns (per-sequence tokens/s, aggregate tokens/s)."""
    steps = bandwidth / (weight_bytes + kv_bytes)
    return steps, steps * batch


def spec_expected_tokens(alpha, gamma):
    """Expected tokens produced per target-model forward pass, i.i.d. acceptance rate alpha, draft length gamma."""
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


# ---------------------------------------------------------------------------
# 1. KV cache size
# ---------------------------------------------------------------------------
per_tok = kv_bytes_per_token(L, N_KV, D_HEAD, 2)
print("1. KV cache, synthetic 8B-style GQA model (L=32, n_kv=8, d_head=128, BF16 = 2 bytes)")
print(f"   per token: 2 x 32 x 8 x 128 x 2 = {per_tok:,} bytes ({per_tok / 1024:.0f} KiB)")
kv1 = kv_cache_bytes(L, N_KV, D_HEAD, 2, 8192, 1)
kv16 = kv_cache_bytes(L, N_KV, D_HEAD, 2, 8192, 16)
print(f"   T=8192, B=1 : {per_tok:,} x 8192 = {kv1:,} bytes = {kv1 / GB:.3f} GB")
print(f"   T=8192, B=16: {kv1:,} x 16 = {kv16:,} bytes = {kv16 / GB:.2f} GB")
for name, nkv in [("MHA (n_kv=32)", 32), ("GQA (n_kv=8) ", 8), ("MQA (n_kv=1) ", 1)]:
    b = kv_cache_bytes(L, nkv, D_HEAD, 2, 8192, 16)
    print(f"   {name}: {kv_bytes_per_token(L, nkv, D_HEAD, 2):>7,} B/token -> {b / GB:6.2f} GB at T=8192, B=16")
print(f"   FP8 KV cache (1 byte): {kv_cache_bytes(L, N_KV, D_HEAD, 1, 8192, 16) / GB:.2f} GB at T=8192, B=16")

# ---------------------------------------------------------------------------
# 2. Weights and the memory budget
# ---------------------------------------------------------------------------
w16 = PARAMS * 2
print("\n2. Weights: 8e9 params x bytes/param")
for name, nb in [("BF16", 2), ("FP8/INT8", 1), ("INT4", 0.5)]:
    print(f"   {name:8s}: 8e9 x {nb} = {PARAMS * nb / GB:.0f} GB")
total = w16 + kv16
print(f"   BF16 weights + KV (T=8192, B=16) = 16 + {kv16 / GB:.2f} = {total / GB:.2f} GB of {MEM / GB:.0f} GB")
max_batch = int((MEM - w16) // kv1)
print(f"   max batch at T=8192 (no activation headroom): floor((80 - 16) / {kv1 / GB:.3f}) = {max_batch}")

# ---------------------------------------------------------------------------
# 3. Decode is memory-bandwidth bound
# ---------------------------------------------------------------------------
print("\n3. Decode upper bound = bandwidth / bytes read per step")
s0, _ = decode_tokens_per_s(w16, 0, BW, 1)
print(f"   B=1, ignore KV:  3.0e12 / 16e9 = {s0:.1f} tokens/s")
s1, _ = decode_tokens_per_s(w16, kv1, BW, 1)
print(f"   B=1, T=8192:     3.0e12 / ({w16 / GB:.0f} + {kv1 / GB:.3f}) GB = {s1:.1f} tokens/s")
s16, a16 = decode_tokens_per_s(w16, kv16, BW, 16)
print(f"   B=16, T=8192:    3.0e12 / ({w16 / GB:.0f} + {kv16 / GB:.2f}) GB = {s16:.1f} steps/s per sequence, x16 = {a16:.0f} tokens/s")
print(f"   ms per token, B=1 no KV: 1000 / {s0:.1f} = {1000 / s0:.2f} ms")

print("\n   Arithmetic intensity (FLOPs per byte read), weights only")
print("   decode B=1: 2 FLOPs per param / 2 bytes per param = 1 FLOP/byte")
print(f"   accelerator ridge point: 1000e12 / 3.0e12 = {PEAK / BW:.1f} FLOP/byte -> decode B=1 uses ~{1 / (PEAK / BW):.1%} of peak compute")
print("   decode B=64: ~64 FLOP/byte (weights are read once, used 64 times)")
prefill_s = 2 * PARAMS * 2000 / PEAK
print(f"   prefill of 2000 tokens: 2 x 8e9 x 2000 = {2 * PARAMS * 2000:.1e} FLOPs / 1000e12 = {prefill_s * 1000:.0f} ms at peak (intensity ~2000)")

# ---------------------------------------------------------------------------
# 4. FlashAttention: the score matrix it never writes to HBM
# ---------------------------------------------------------------------------
n = 8192
per_head = n * n * 2
print(f"\n4. Naive attention scores, n=8192, 16-bit: 8192^2 x 2 = {per_head:,} bytes = {per_head / 1e6:.0f} MB per head")
print(f"   x 32 heads = {per_head * 32 / GB:.2f} GB per layer (FlashAttention keeps tiles in on-chip SRAM instead)")

# ---------------------------------------------------------------------------
# 5. Speculative decoding
# ---------------------------------------------------------------------------
alpha, gamma, c = 0.8, 4, 0.05
e = spec_expected_tokens(alpha, gamma)
cost = 1 + gamma * c
print(f"\n5. Speculative decoding, alpha={alpha}, gamma={gamma}, draft cost c={c} of a target pass")
print(f"   0.8^5 = {alpha ** 5:.5f};  E[tokens/pass] = (1 - {alpha ** 5:.5f}) / (1 - 0.8) = {1 - alpha ** 5:.5f} / 0.2 = {e:.4f}")
print(f"   cost per round = 1 + 4 x 0.05 = {cost:.2f} target passes;  speedup = {e:.4f} / {cost:.2f} = {e / cost:.3f}x")
for a in (0.5, 0.6, 0.9):
    ea = spec_expected_tokens(a, gamma)
    print(f"   alpha={a}: E={ea:.3f}, speedup={ea / cost:.2f}x")

# ---------------------------------------------------------------------------
# 6. Mixture of experts
# ---------------------------------------------------------------------------
shared, n_exp, per_exp, top_k = 4e9, 16, 2e9, 2
tot, act = shared + n_exp * per_exp, shared + top_k * per_exp
print("\n6. Synthetic MoE: 4B shared + 16 experts x 2B, top-2 routing")
print(f"   total = 4 + 16 x 2 = {tot / 1e9:.0f}B params -> {tot * 2 / GB:.0f} GB in BF16 (memory)")
print(f"   active = 4 + 2 x 2 = {act / 1e9:.0f}B params per token -> {2 * act:.1e} FLOPs per token (compute)")

# ---------------------------------------------------------------------------
# 7. Latency metrics
# ---------------------------------------------------------------------------
ttft, tpot, n_out = 0.200, 0.025, 300
e2e = ttft + tpot * (n_out - 1)
print(f"\n7. E2E = TTFT + TPOT x (n_out - 1) = 0.200 + 0.025 x 299 = 0.200 + {tpot * 299:.3f} = {e2e:.3f} s")
print(f"   per-user decode speed = 1 / 0.025 = {1 / tpot:.0f} tokens/s")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ctx = np.array([1024, 2048, 4096, 8192, 16384, 32768])
for name, nkv, col in [("MHA, n_kv = 32", 32, ORANGE), ("GQA, n_kv = 8", 8, BLUE), ("MQA, n_kv = 1", 1, TEAL)]:
    ax1.plot(ctx / 1024, (w16 + kv_cache_bytes(L, nkv, D_HEAD, 2, ctx, 16)) / GB, "o-", color=col, label=name)
ax1.axhline(w16 / GB, color=GRAY, ls=":", label="BF16 weights alone (16 GB)")
ax1.axhline(MEM / GB, color=PURPLE, ls="--", label="80 GB accelerator")
ax1.set_xscale("log", base=2)
ax1.set_xticks(ctx / 1024, [f"{c:g}k" for c in ctx / 1024])
ax1.set_ylim(0, 200)
ax1.set(xlabel="context length T (tokens)", ylabel="weights + KV cache (GB)", title="Memory, synthetic 8B-style model, batch 16")
ax1.legend(fontsize=8)

batches = np.array([1, 2, 4, 8, 16, 32, 48])
per_seq, agg = zip(*(decode_tokens_per_s(w16, kv_cache_bytes(L, N_KV, D_HEAD, 2, 8192, b), BW, b) for b in batches))
ax2.plot(batches, agg, "o-", color=BLUE, label="aggregate tokens/s")
ax2.set(xlabel="batch size B (T = 8192)", ylabel="aggregate decode tokens/s (upper bound)", title="Decode speed limit at 3.0 TB/s")
ax2b = ax2.twinx()
ax2b.plot(batches, per_seq, "s--", color=ORANGE, label="per-sequence tokens/s")
ax2b.set_ylabel("per-sequence tokens/s", color=ORANGE)
ax2b.grid(False)
ax2b.set_ylim(0, 200)
ax2.legend(loc="upper left", fontsize=8)
ax2b.legend(loc="center right", fontsize=8)
fig.tight_layout()
save(fig, "llm-inference")
