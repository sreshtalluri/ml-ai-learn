# Adapted from AI Engineering from Scratch (MIT), phases/07-transformers-deep-dive/11-mixture-of-experts
"""Mixture of experts: top-k routing, gate renormalization, expert capacity and dropped tokens,
the Switch load-balancing loss, auxiliary-loss-free bias balancing, and total vs active parameters.

Run from guide/:  uv run code/15-llms/mixture_of_experts.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, plt, save  # noqa: E402


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. Worked example: one token, four experts, top-2
# ---------------------------------------------------------------------------
logits = np.array([2.0, 1.0, 0.5, -1.0])
p = softmax(logits)
top = np.argsort(-logits)[:2]
gates = p[top] / p[top].sum()
print("=== one token, 4 experts, top-2 ===")
print("router probs:", np.round(p, 4), " chosen:", top, " renormalized gates:", np.round(gates, 4))


# ---------------------------------------------------------------------------
# 2. A tiny MoE layer: router + top-k + weighted sum of expert outputs
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
d, E, k, T = 16, 8, 2, 512                                   # model width, experts, top-k, tokens
W_router = rng.normal(0, 0.3, (d, E))
W_router[:, 0] += 0.15                                       # synthetic skew: expert 0 is "popular"
experts = rng.normal(0, d ** -0.5, (E, d, d))                # one linear map per expert, for illustration
X = rng.normal(0, 1, (T, d)) + 0.3                           # synthetic token states, [T, d]


def route(X, W_router, k, bias=None):
    logits = X @ W_router                                    # [T, E]
    probs = softmax(logits)
    pick = logits if bias is None else logits + bias         # bias steers selection only
    top = np.argsort(-pick, axis=1)[:, :k]                   # [T, k] chosen experts
    g = np.take_along_axis(probs, top, axis=1)
    return top, g / g.sum(axis=1, keepdims=True), probs      # gates renormalized over the k chosen


def moe_forward(X, top, gates):
    out = np.zeros_like(X)
    for slot in range(top.shape[1]):
        for e in range(len(experts)):
            rows = top[:, slot] == e                         # tokens sent to expert e in this slot
            out[rows] += gates[rows, slot, None] * (X[rows] @ experts[e])
    return out                                               # [T, d]


top, gates, probs = route(X, W_router, k)
Y = moe_forward(X, top, gates)
load = np.bincount(top.ravel(), minlength=E)
print(f"\n=== {T} synthetic tokens, {E} experts, top-{k} ===")
print("output shape:", Y.shape, " requests per expert:", load, f" (fair share {T * k // E})")


# ---------------------------------------------------------------------------
# 3. Capacity: each expert has a fixed number of slots; overflow is dropped
# ---------------------------------------------------------------------------
def dropped(top, E, cf):
    cap = int(np.ceil(cf * top.size / E))                    # slots per expert
    used = np.zeros(E, dtype=int)
    lost = 0
    for t in range(top.shape[0]):                            # tokens claim slots in order
        for e in top[t]:
            if used[e] < cap:
                used[e] += 1
            else:
                lost += 1                                    # this token skips expert e
    return cap, lost


for cf in (1.0, 1.25, 2.0):
    cap, lost = dropped(top, E, cf)
    print(f"  capacity factor {cf:<4}: {cap:>3} slots/expert, {lost:>3} of {top.size} assignments dropped")


# ---------------------------------------------------------------------------
# 4. Two ways to balance: the Switch auxiliary loss, and a selection bias
# ---------------------------------------------------------------------------
def switch_aux_loss(top, probs, E):
    f = np.bincount(top[:, 0], minlength=E) / len(top)       # fraction routed (first choice)
    P = probs.mean(axis=0)                                   # mean router probability
    return E * float(f @ P)                                  # 1.0 when perfectly uniform


print(f"\nSwitch aux loss (1.0 = balanced): {switch_aux_loss(top, probs, E):.3f}")

bias, gamma, history = np.zeros(E), 0.02, []
for step in range(100):                                      # aux-loss-free balancing (DeepSeek-V3)
    t_b, _, _ = route(X, W_router, k, bias)
    load_b = np.bincount(t_b.ravel(), minlength=E)
    history.append(load_b.max() / load_b.mean())
    bias -= gamma * np.sign(load_b - load_b.mean())          # overloaded down, underloaded up
print("max/mean load before, after 100 bias updates:", f"{history[0]:.2f} -> {history[-1]:.2f}")
print("requests per expert after:", load_b)
print("dropped at capacity factor 1.25 after balancing:", dropped(t_b, E, 1.25)[1], "of", t_b.size)


# ---------------------------------------------------------------------------
# 5. Total vs active parameters
# ---------------------------------------------------------------------------
def params(shared, per_expert, E, k):
    return shared + E * per_expert, shared + k * per_expert   # (total, active)


tot, act = params(4e9, 2e9, 16, 2)
print(f"\nsynthetic 4B shared + 16 x 2B experts, top-2: total {tot / 1e9:.0f}B "
      f"({tot * 2 / 1e9:.0f} GB in BF16), active {act / 1e9:.0f}B, {2 * act / 1e9:.0f} GFLOPs per token")


# ---------------------------------------------------------------------------
# 6. Figure
# ---------------------------------------------------------------------------
cap = int(np.ceil(1.25 * top.size / E))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
x = np.arange(E)
a1.bar(x - 0.2, load, 0.4, color=ORANGE, label="no balancing")
a1.bar(x + 0.2, load_b, 0.4, color=BLUE, label="bias balancing")
a1.axhline(cap, color=GRAY, ls="--", label=f"capacity (factor 1.25) = {cap}")
a1.set_xlabel("expert"); a1.set_ylabel("token assignments")
a1.set_title(f"Requests per expert ({T} synthetic tokens, top-{k})")
a1.legend(fontsize=8)
a2.plot(history, color=BLUE)
a2.axhline(1, color=GRAY, ls=":")
a2.set_xlabel("bias update step"); a2.set_ylabel("max load / mean load")
a2.set_title("Auxiliary-loss-free balancing")
fig.tight_layout()
save(fig, "mixture-of-experts")
