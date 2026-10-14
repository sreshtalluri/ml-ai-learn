# Adapted from AI Engineering from Scratch (MIT), phases/07-transformers-deep-dive/13-scaling-laws
"""Chinchilla scaling law: predicted loss, the compute-optimal (N, D) split, and the cost of over-training.

Run from guide/:  uv run code/15-llms/scaling_laws.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402

# Hoffmann et al. (2022), Approach 3 fit. N = parameters, D = training tokens.
A, B, ALPHA, BETA, E = 406.4, 410.7, 0.34, 0.28, 1.69


# ---------------------------------------------------------------------------
# 1. The loss law and the compute budget
# ---------------------------------------------------------------------------
def loss(N, D):
    return E + A / N**ALPHA + B / D**BETA      # floor + model-size term + data term


def flops(N, D):
    return 6 * N * D                           # ~2ND forward + ~4ND backward


print("=== predicted loss for three public (N, D) pairs ===")
for name, N, D in [("GPT-3", 175e9, 300e9), ("Chinchilla", 70e9, 1.4e12), ("Llama 3 8B", 8e9, 15e12)]:
    print(f"  {name:<11} N={N / 1e9:>5.0f}B  D={D / 1e12:>5.2f}T  D/N={D / N:>6.0f}  "
          f"C={flops(N, D):.2e}  loss={loss(N, D):.3f}")


# ---------------------------------------------------------------------------
# 2. Compute-optimal split: minimise loss(N, C / 6N) over N
# ---------------------------------------------------------------------------
def optimal_closed_form(C):
    G = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))
    N = G * (C / 6) ** (BETA / (ALPHA + BETA))
    return N, C / (6 * N)


def optimal_grid(C):
    Ns = np.logspace(6, 14, 20_001)            # brute force check of the calculus
    return Ns[np.argmin(loss(Ns, C / (6 * Ns)))]


print("\n=== compute-optimal split under the Approach 3 fit ===")
for C in [1e20, 1e22, 1e24, 1e26]:
    N, D = optimal_closed_form(C)
    assert abs(optimal_grid(C) / N - 1) < 1e-3
    print(f"  C={C:.0e}  N*={N / 1e9:8.2f}B  D*={D / 1e9:10.1f}B  D/N={D / N:6.1f}  loss={loss(N, D):.3f}")


# ---------------------------------------------------------------------------
# 3. Over-training: same compute, a smaller model on more tokens
# ---------------------------------------------------------------------------
C = 1e24
N_opt, D_opt = optimal_closed_form(C)
print(f"\n=== spending C={C:.0e} FLOPs ===")
for shrink in [1, 2, 4, 10]:
    N, D = N_opt / shrink, D_opt * shrink
    print(f"  N = N*/{shrink:<2}  N={N / 1e9:6.1f}B  D={D / 1e12:6.2f}T  loss={loss(N, D):.4f}  "
          f"(+{loss(N, D) - loss(N_opt, D_opt):.4f})  inference FLOPs/token {2 * N:.2e}")

# ---------------------------------------------------------------------------
# 4. Figure: loss along fixed-compute lines, and the optimum's D/N
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for C, color in zip([1e20, 1e22, 1e24], [GRAY, TEAL, BLUE]):
    Ns = np.logspace(7, 12.5, 400)
    a1.plot(Ns, loss(Ns, C / (6 * Ns)), color=color, label=f"C = {C:.0e} FLOPs")
    N, D = optimal_closed_form(C)
    a1.plot(N, loss(N, D), "o", color=ORANGE)
a1.set_xscale("log"); a1.set_ylim(1.8, 3.6)
a1.set_xlabel("parameters N (tokens D = C / 6N)"); a1.set_ylabel("predicted loss")
a1.set_title("Fixed compute: too small or too big both lose"); a1.legend()
Cs = np.logspace(18, 26, 50)
ratios = [optimal_closed_form(c)[1] / optimal_closed_form(c)[0] for c in Cs]
a2.plot(Cs, ratios, color=BLUE, label="Approach 3 fit (this script)")
a2.axhline(20, color=ORANGE, ls="--", label="~20 tokens/param (Approaches 1 and 2)")
a2.set_xscale("log"); a2.set_yscale("log")
a2.set_xlabel("training compute C (FLOPs)"); a2.set_ylabel("optimal D / N")
a2.set_title("Compute-optimal tokens per parameter"); a2.legend()
fig.tight_layout()
save(fig, "scaling-laws")
