# Adapted from AI Engineering from Scratch (MIT), phases/10-llms-from-scratch/06-instruction-tuning-sft, phases/11-llm-engineering/08-fine-tuning-lora, phases/19-capstone-projects/46-gradient-accumulation
"""Fine-tuning mechanics from scratch: SFT loss masking, a LoRA layer trained by hand and merged,
and gradient accumulation (including the token-count normalization bug).

Synthetic: random weights and a random low-rank "task" update; numbers show mechanics only.
Run from guide/:  uv run code/15-llms/finetune_scratch.py
"""
import numpy as np


# --- 1. SFT loss masking --------------------------------------------------------
def masked_nll(logp_true, mask):
    """logp_true: log-prob of each actual next token [T]; mask: 1 on response tokens, 0 on prompt."""
    return -(logp_true * mask).sum() / mask.sum()   # average over response tokens only


# --- 2. A LoRA linear layer ------------------------------------------------------
class LoRALinear:
    def __init__(self, W0, r, alpha, rng):
        d_out, d_in = W0.shape
        self.W0 = W0                                    # frozen
        self.A = rng.normal(0, 1 / np.sqrt(d_in), (r, d_in))   # random
        self.B = np.zeros((d_out, r))                   # zero, so B @ A = 0 at step 0
        self.s = alpha / r

    def forward(self, X):                               # X: [n, d_in]
        self.X, self.XA = X, X @ self.A.T               # [n, r]: the low-rank bottleneck
        return X @ self.W0.T + self.s * self.XA @ self.B.T

    def merged(self):
        return self.W0 + self.s * self.B @ self.A       # same shape as W0: no extra latency


# --- 3. LoRA backward: gradients for A and B only --------------------------------
def lora_step(layer, X, Y, lr):
    pred = layer.forward(X)
    dP = 2 * (pred - Y) / len(X)                        # mean squared error gradient, [n, d_out]
    dB = layer.s * dP.T @ layer.XA                      # [d_out, r]
    dA = layer.s * (dP @ layer.B).T @ layer.X           # [r, d_in]  (zero while B is zero)
    layer.B -= lr * dB
    layer.A -= lr * dA
    return ((pred - Y) ** 2).sum(1).mean()


# --- 4. Gradient accumulation ------------------------------------------------------
def grad_sum(w, X, y, mask):
    """Gradient of the SUM of masked squared errors for a linear model, and the token count."""
    err = (X @ w - y) * mask
    return 2 * X.T @ err, mask.sum()


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("== SFT loss masking: 300-token prompt, 100-token response ==")
    logp = np.concatenate([np.full(300, np.log(0.05)), np.full(100, np.log(0.5))])   # synthetic
    mask = np.concatenate([np.zeros(300), np.ones(100)])
    print(f"masked loss (response only): {masked_nll(logp, mask):.3f}")
    print(f"unmasked loss (all tokens) : {masked_nll(logp, np.ones(400)):.3f}  <- dominated by the prompt")

    print("\n== LoRA on a 64x64 layer, rank 4 ==")
    d, r = 64, 4
    W0 = rng.normal(0, 1 / np.sqrt(d), (d, d))
    delta = rng.normal(0, 1, (d, r)) @ rng.normal(0, 1, (r, d)) / d   # the task needs a rank-4 change
    X = rng.normal(0, 1, (512, d))
    Y = X @ (W0 + delta).T
    layer = LoRALinear(W0, r=r, alpha=8, rng=rng)
    print(f"step 0 output equals base model: {np.allclose(layer.forward(X), X @ W0.T)}")
    print(f"trainable {layer.A.size + layer.B.size:,} vs full {W0.size:,} ({(layer.A.size + layer.B.size) / W0.size:.1%})")
    for step in range(401):
        loss = lora_step(layer, X, Y, lr=0.02)
        if step in (0, 100, 400):
            print(f"step {step:3d}  loss {loss:.4f}")
    print(f"merged weights give the same outputs: {np.allclose(X @ layer.merged().T, layer.forward(X))}")

    print("\n== Gradient accumulation: 4 micro-batches vs one batch of 32 token positions ==")
    w = rng.normal(0, 1, 8)
    Xb, yb = rng.normal(0, 1, (32, 8)), rng.normal(0, 1, 32)
    mb = np.array([1] * 8 + [1] * 2 + [0] * 6 + [1] * 8 + [1] * 1 + [0] * 7, float)  # uneven response lengths
    g_full, n_full = grad_sum(w, Xb, yb, mb)
    full = g_full / n_full                                          # one big batch: mean over 19 tokens
    parts = [grad_sum(w, Xb[i:i + 8], yb[i:i + 8], mb[i:i + 8]) for i in range(0, 32, 8)]
    naive = sum(g / n for g, n in parts) / 4                        # mean of per-micro-batch means
    right = sum(g for g, _ in parts) / sum(n for _, n in parts)     # sum, then divide by total tokens
    print(f"tokens per micro-batch: {[int(n) for _, n in parts]}")
    print(f"naive  (average of means) matches full batch: {np.allclose(naive, full)}  max error {np.abs(naive - full).max():.3f}")
    print(f"correct (divide by total) matches full batch: {np.allclose(right, full)}")
