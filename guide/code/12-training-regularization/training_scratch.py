# Adapted from AI Engineering from Scratch (MIT), phases/03-deep-learning-core/07-regularization, phases/03-deep-learning-core/08-weight-initialization, phases/03-deep-learning-core/09-learning-rate-schedules, phases/19-capstone-projects/44-cosine-lr-warmup
"""The training toolkit from scratch: inverted dropout, layer norm, weight initialization
(why the scale must be 1/fan_in or 2/fan_in), a warmup-plus-cosine schedule, and early
stopping with patience.

Run from guide/:  uv run code/12-training-regularization/training_scratch.py
"""
import math

import numpy as np


# --- 1. Inverted dropout: zero with probability p, scale survivors by 1/(1-p) ------
def dropout(a, p, rng, training=True):
    if not training or p == 0:
        return a                                  # inference: identity, no rescaling
    mask = rng.random(a.shape) >= p               # keep with probability 1 - p
    return a * mask / (1 - p)


# --- 2. Layer norm: per example, across features ----------------------------------
def layer_norm(a, gamma=1.0, beta=0.0, eps=1e-5):
    mu = a.mean(axis=-1, keepdims=True)
    var = a.var(axis=-1, keepdims=True)           # population variance (divide by n)
    return gamma * (a - mu) / np.sqrt(var + eps) + beta


# --- 3. Initialization: keep the activation scale constant through depth -----------
def init(fan_in, fan_out, scheme, rng):
    std = {"small": 0.01,
           "xavier": math.sqrt(1 / fan_in),       # LeCun/Xavier-style, for tanh
           "he": math.sqrt(2 / fan_in)}[scheme]   # ReLU zeros half the inputs, so double it
    return rng.normal(0, std, (fan_in, fan_out))


def activation_std_by_depth(scheme, act, depth=20, width=256, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(512, width))             # synthetic unit-variance inputs
    stds = []
    for _ in range(depth):
        x = act(x @ init(width, width, scheme, rng))
        stds.append(x.std())
    return stds


# --- 4. Learning-rate schedule: linear warmup, then cosine decay to a floor ---------
def lr_at(t, max_lr, warmup, total, min_lr=0.0):
    if t < warmup:
        return max_lr * (t + 1) / warmup          # ramp up from max_lr/warmup to max_lr
    progress = (t - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# --- 5. Early stopping: remember the best epoch, stop after `patience` bad ones ------
def early_stop(val_losses, patience):
    best, best_epoch, bad = float("inf"), -1, 0
    for epoch, v in enumerate(val_losses):
        if v < best:
            best, best_epoch, bad = v, epoch, 0   # new best: this is the checkpoint to keep
        else:
            bad += 1
            if bad >= patience:
                return best_epoch, epoch          # stopped here, restore best_epoch
    return best_epoch, len(val_losses) - 1


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    rng = np.random.default_rng(0)

    print("== Dropout (p = 0.5) on the course example [2, 4, 6, 8] with mask [1, 0, 1, 0] ==")
    a = np.array([2.0, 4.0, 6.0, 8.0])
    print("scaled output:", a * np.array([1, 0, 1, 0]) / 0.5)
    many = np.stack([dropout(a, 0.5, rng) for _ in range(100_000)])
    print("mean over 100,000 random masks:", many.mean(axis=0), " (original:", a, ")")
    print("eval mode returns the input:", dropout(a, 0.5, rng, training=False))

    print("\n== Layer norm on [1, 2, 3, 6] ==")
    print(layer_norm(np.array([1.0, 2.0, 3.0, 6.0])))

    print("\n== Activation std after 1, 5, 10, 20 layers (width 256, synthetic inputs) ==")
    relu = lambda z: np.maximum(0, z)
    for scheme, act, name in [("small", np.tanh, "std 0.01 + tanh"), ("xavier", np.tanh, "xavier + tanh"),
                              ("small", relu, "std 0.01 + relu"), ("he", relu, "he + relu"),
                              ("xavier", relu, "xavier + relu")]:
        s = activation_std_by_depth(scheme, act)
        print(f"{name:>16}: " + "  ".join(f"{s[i]:.2e}" for i in (0, 4, 9, 19)))

    print("\n== Warmup + cosine: max_lr 3e-4, 100 warmup steps, 1,000 total, floor 3e-5 ==")
    for t in (0, 49, 99, 100, 550, 999):
        print(f"step {t:4d}: lr = {lr_at(t, 3e-4, 100, 1000, 3e-5):.3e}")

    print("\n== Early stopping on a synthetic validation curve, patience 5 ==")
    val = [1.0, 0.7, 0.55, 0.45, 0.40, 0.38, 0.364, 0.37, 0.39, 0.42, 0.45, 0.50, 0.55, 0.6]
    best_epoch, stopped = early_stop(val, patience=5)
    print(f"best epoch {best_epoch} (val {val[best_epoch]}), stopped at epoch {stopped}")
