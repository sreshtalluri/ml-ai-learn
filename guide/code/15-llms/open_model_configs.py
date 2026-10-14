# Adapted from AI Engineering from Scratch (MIT), phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs
"""Read open-model configs: parameter counts (total and active), KV cache per token and at full
context, for GPT-2 small, Llama 3 8B, Mistral 7B, Mixtral 8x7B, Gemma 2 9B, Qwen2.5 72B, DeepSeek-V3.
Configs use Hugging Face field names, copied from each model's public config.json.

Run from guide/:  uv run code/15-llms/open_model_configs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402

LLAMA = dict(hidden_size=4096, intermediate_size=14336, num_hidden_layers=32, num_attention_heads=32,
             num_key_value_heads=8, mlp="gated", norm="rms")
CONFIGS = {
    "GPT-2 small": dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12,
                        num_key_value_heads=12, vocab_size=50257, max_position_embeddings=1024,
                        mlp="plain", norm="layer", bias=True, tie_word_embeddings=True),
    "Llama 3 8B": dict(LLAMA, vocab_size=128256, max_position_embeddings=131072),
    "Mistral 7B": dict(LLAMA, vocab_size=32000, max_position_embeddings=32768),
    "Mixtral 8x7B": dict(LLAMA, vocab_size=32000, max_position_embeddings=32768,
                         num_local_experts=8, num_experts_per_tok=2),
    "Gemma 2 9B": dict(hidden_size=3584, intermediate_size=14336, num_hidden_layers=42, num_attention_heads=16,
                       num_key_value_heads=8, head_dim=256, vocab_size=256000, max_position_embeddings=8192,
                       mlp="gated", norm="rms", norms_per_layer=4, tie_word_embeddings=True,
                       layer_types=["sliding_attention", "full_attention"] * 21, sliding_window=4096),
    "Qwen2.5 72B": dict(hidden_size=8192, intermediate_size=29568, num_hidden_layers=80, num_attention_heads=64,
                        num_key_value_heads=8, vocab_size=152064, max_position_embeddings=131072,
                        mlp="gated", norm="rms", qkv_bias=True),
    "DeepSeek-V3": dict(hidden_size=7168, intermediate_size=18432, num_hidden_layers=61, num_attention_heads=128,
                        vocab_size=129280, max_position_embeddings=163840, mlp="gated", norm="rms",
                        q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
                        v_head_dim=128, first_k_dense_replace=3, moe_intermediate_size=2048,
                        n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8),
}


# ---------------------------------------------------------------------------
# 1. Attention parameters: GQA shrinks K and V; MLA compresses them through a latent
# ---------------------------------------------------------------------------
def attention_params(c):
    h, nq = c["hidden_size"], c["num_attention_heads"]
    if "kv_lora_rank" in c:                                   # MLA (DeepSeek)
        qk = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
        q = h * c["q_lora_rank"] + c["q_lora_rank"] + c["q_lora_rank"] * nq * qk
        kv = h * (c["kv_lora_rank"] + c["qk_rope_head_dim"]) + c["kv_lora_rank"] \
            + c["kv_lora_rank"] * nq * (c["qk_nope_head_dim"] + c["v_head_dim"])
        return q + kv + nq * c["v_head_dim"] * h
    d = c.get("head_dim", h // nq)
    nkv = c["num_key_value_heads"]
    p = h * nq * d + 2 * h * nkv * d + nq * d * h             # Q, K and V, output
    if c.get("bias"):
        p += nq * d + 2 * nkv * d + h                         # GPT-2: bias on every projection
    elif c.get("qkv_bias"):
        p += nq * d + 2 * nkv * d                             # Qwen: bias on Q, K, V only
    return p


# ---------------------------------------------------------------------------
# 2. MLP parameters: SwiGLU/GeGLU has three matrices, GELU MLP has two
# ---------------------------------------------------------------------------
def mlp_params(c, ff):
    h = c["hidden_size"]
    if c["mlp"] == "gated":
        return 3 * h * ff                                     # gate, up, down; no biases
    return 2 * h * ff + ff + h                                # GPT-2: up and down, with biases


def count(c):
    """Return (total, active) parameters."""
    h, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    norm = h if c["norm"] == "rms" else 2 * h                 # RMSNorm: scale only
    per_layer_fixed = attention_params(c) + c.get("norms_per_layer", 2) * norm
    dense_mlp = mlp_params(c, c["intermediate_size"])
    experts = c.get("num_local_experts") or c.get("n_routed_experts")
    if not experts:                                           # dense model
        blocks = L * (per_layer_fixed + dense_mlp)
        blocks_active = blocks
    else:                                                     # MoE: some layers may stay dense
        k, shared = c["num_experts_per_tok"], c.get("n_shared_experts", 0)
        expert = mlp_params(c, c.get("moe_intermediate_size", c["intermediate_size"]))
        n_dense = c.get("first_k_dense_replace", 0)
        router = h * experts
        moe_total = per_layer_fixed + router + (experts + shared) * expert
        moe_active = per_layer_fixed + router + (k + shared) * expert
        dense = per_layer_fixed + dense_mlp
        blocks = n_dense * dense + (L - n_dense) * moe_total
        blocks_active = n_dense * dense + (L - n_dense) * moe_active
    pos = c["max_position_embeddings"] * h if c["norm"] == "layer" else 0   # GPT-2 learned positions
    emb = V * h * (1 if c.get("tie_word_embeddings") else 2) + pos           # embedding (+ LM head)
    return emb + blocks + norm, emb + blocks_active + norm


# ---------------------------------------------------------------------------
# 3. KV cache: elements stored per token per layer, times the tokens each layer keeps
# ---------------------------------------------------------------------------
def kv_elems_per_token_layer(c):
    if "kv_lora_rank" in c:
        return c["kv_lora_rank"] + c["qk_rope_head_dim"]      # MLA caches the latent + RoPE key
    d = c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])
    return 2 * c["num_key_value_heads"] * d                   # K and V for each KV head


def kv_cache_bytes(c, tokens, bytes_per=2):
    kinds = c.get("layer_types", ["full_attention"] * c["num_hidden_layers"])
    kept = [min(tokens, c["sliding_window"]) if k == "sliding_attention" else tokens for k in kinds]
    return kv_elems_per_token_layer(c) * sum(kept) * bytes_per


print(f"{'model':<13} {'total':>15} {'active':>15} {'KV/token':>10} {'KV @ 32k, BF16':>15}")
rows = {}
for name, c in CONFIGS.items():
    total, active = count(c)
    per_tok = kv_cache_bytes(c, 1)
    at_32k = kv_cache_bytes(c, min(32768, c["max_position_embeddings"]))
    rows[name] = (total, active, at_32k)
    print(f"{name:<13} {total:>15,} {active:>15,} {per_tok / 1024:>7.0f} KiB {at_32k / 1e9:>12.2f} GB")

assert count(CONFIGS["Llama 3 8B"])[0] == 8_030_261_248      # matches the published count exactly
assert count(CONFIGS["GPT-2 small"])[0] == 124_439_808

g = CONFIGS["Gemma 2 9B"]
print(f"\nGemma 2 9B at 8,192 tokens: {kv_cache_bytes(g, 8192) / 1e9:.2f} GB with sliding layers, "
      f"{kv_elems_per_token_layer(g) * 42 * 8192 * 2 / 1e9:.2f} GB if every layer were global")
ds = CONFIGS["DeepSeek-V3"]
mha_equiv = 128 * ((128 + 64) + 128) * 61 * 2                # full per-head K (192) and V (128)
print(f"DeepSeek-V3 MLA: {kv_cache_bytes(ds, 1):,} bytes/token vs {mha_equiv:,} for uncompressed "
      f"128-head K and V ({mha_equiv / kv_cache_bytes(ds, 1):.0f}x smaller)")


# ---------------------------------------------------------------------------
# 4. Figure: total vs active parameters, and KV cache at 32k tokens
# ---------------------------------------------------------------------------
names = [n for n in rows if n != "GPT-2 small"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
y = range(len(names))
a1.barh(y, [rows[n][0] / 1e9 for n in names], color=GRAY, label="total (memory)")
a1.barh(y, [rows[n][1] / 1e9 for n in names], color=BLUE, label="active per token (compute)")
a1.set_yticks(list(y), names); a1.set_xscale("log"); a1.invert_yaxis()
a1.set_xlabel("parameters (billions, log scale)"); a1.set_title("Total vs active parameters"); a1.legend()
a2.barh(y, [rows[n][2] / 1e9 for n in names], color=[ORANGE if n in ("Gemma 2 9B", "DeepSeek-V3") else TEAL for n in names])
a2.set_yticks(list(y), names); a2.invert_yaxis()
a2.set_xlabel("GB per sequence (BF16)")
a2.set_title("KV cache at 32,768 tokens (Gemma 2: its 8,192 max)\norange = sliding-window or MLA cache")
for i, n in enumerate(names):
    a2.text(rows[n][2] / 1e9, i, f" {rows[n][2] / 1e9:.2f}", va="center", fontsize=8)
fig.tight_layout()
save(fig, "open-models-architecture-walkthroughs")
