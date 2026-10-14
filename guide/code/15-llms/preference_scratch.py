# Adapted from AI Engineering from Scratch (MIT), phases/09-reinforcement-learning/09-reward-modeling-rlhf, phases/10-llms-from-scratch/08-dpo, phases/19-capstone-projects/40-dpo-from-scratch
"""Preference tuning from scratch: LoRA parameter savings, a Bradley-Terry reward model trained on
preference pairs, the DPO loss, and DPO training of a toy policy, checked against the closed-form
optimum pi*(y) proportional to pi_ref(y) * exp(r(y) / beta).

Synthetic: four candidate responses with made-up quality scores; pairs are sampled from them.
Run from guide/:  uv run code/15-llms/preference_scratch.py
"""
import numpy as np


# --- 1. LoRA parameter savings -------------------------------------------------
def lora_params(d_in, d_out, r):
    return r * (d_in + d_out)                     # A: [r, d_in] plus B: [d_out, r]


# --- 2. Preference data: which of two responses did a rater prefer? ------------
RESPONSES = ["cites the policy and answers", "answers, no citation", "vague hedge", "confident but wrong"]
TRUE_SCORE = np.array([2.0, 1.0, 0.0, -1.0])     # hidden quality, synthetic


def sample_pairs(n, rng):
    pairs = []
    for _ in range(n):
        i, j = rng.choice(4, size=2, replace=False)
        p_i = 1 / (1 + np.exp(-(TRUE_SCORE[i] - TRUE_SCORE[j])))   # Bradley-Terry: noisy raters
        pairs.append((i, j) if rng.random() < p_i else (j, i))     # (chosen, rejected)
    return np.array(pairs)


# --- 3. Reward model: Bradley-Terry logistic loss --------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_reward_model(pairs, steps=2000, lr=0.5):
    r = np.zeros(4)                               # one scalar reward per response (a real RM is a network)
    w, l = pairs[:, 0], pairs[:, 1]
    for _ in range(steps):
        g = sigmoid(r[w] - r[l]) - 1              # d/dmargin of -log sigmoid(r_w - r_l)
        grad = np.zeros(4)
        np.add.at(grad, w, g); np.add.at(grad, l, -g)
        r -= lr * grad / len(pairs)
        r -= r.mean()                             # rewards are only defined up to a constant
    return r


# --- 4. The DPO loss ----------------------------------------------------------
def dpo_loss(chosen_logratio, rejected_logratio, beta):
    """Each log-ratio is log pi_theta(y|x) - log pi_ref(y|x), summed over the response's tokens."""
    return -np.log(sigmoid(beta * (chosen_logratio - rejected_logratio)))


# --- 5. DPO training on a toy policy (a softmax over the 4 responses) ----------
def log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def train_dpo(pairs, beta, steps=3000, lr=1.0):
    theta = np.zeros(4)                           # policy logits; start equal to the reference
    ref = log_softmax(theta)                      # frozen reference log-probs
    w, l = pairs[:, 0], pairs[:, 1]
    for _ in range(steps):
        lp = log_softmax(theta)
        h = beta * ((lp[w] - ref[w]) - (lp[l] - ref[l]))
        g = beta * (sigmoid(h) - 1)               # d loss / d (log pi_w - log pi_l)
        grad = np.zeros(4)                        # d log pi_w / d theta = onehot_w - pi; the pi terms cancel
        np.add.at(grad, w, g); np.add.at(grad, l, -g)
        theta -= lr * grad / len(pairs)
    return np.exp(log_softmax(theta))


if __name__ == "__main__":
    print("== LoRA savings (adapting-llms worked example) ==")
    one = lora_params(4096, 4096, 8)
    print(f"one 4096x4096 matrix: {4096 * 4096:,} full vs {one:,} LoRA r=8 ({one / 4096**2:.2%})")
    print(f"Q,K,V,O x 32 layers: {one * 4 * 32:,} trainable; bf16 adapter {one * 4 * 32 * 2 / 1e6:.1f} MB")

    print("\n== DPO loss (fine-tuning-in-practice worked numbers, beta = 0.1) ==")
    print(f"step 0 (policy = reference): {dpo_loss(0.0, 0.0, 0.1):.3f}")
    print(f"chosen +2.0, rejected -1.0 : {dpo_loss(2.0, -1.0, 0.1):.3f}")

    rng = np.random.default_rng(0)
    pairs = sample_pairs(2000, rng)
    print(f"\n== Reward model on {len(pairs)} synthetic preference pairs ==")
    r = train_reward_model(pairs)
    agree = np.mean(r[pairs[:, 0]] > r[pairs[:, 1]])
    print("learned rewards:", r.round(2), " (true scores minus their mean:", (TRUE_SCORE - TRUE_SCORE.mean()).round(2), ")")
    print(f"reward model agrees with {agree:.1%} of rater labels (raters are noisy by design)")

    print("\n== DPO: trained policy vs closed form pi_ref * exp(r / beta) ==")
    for beta in (0.5, 2.0):
        pi = train_dpo(pairs, beta)
        closed = np.exp(r / beta) / np.exp(r / beta).sum()   # pi_ref is uniform here
        print(f"beta={beta}: DPO {pi.round(3)}  closed form {closed.round(3)}")
    print("responses:", RESPONSES)
