# Adapted from AI Engineering from Scratch (MIT), phases/09-reinforcement-learning/08-ppo
"""PPO from scratch: an actor-critic on a 4x4 GridWorld with GAE, the clipped surrogate objective,
K epochs of minibatch updates per rollout, and the two numbers every PPO run logs (approx KL, clip fraction).

The world is synthetic: start (0, 0), goal (3, 3), every move costs -1, walls leave you in place.
NumPy only, fixed seeds.
Run from guide/:  uv run code/26-reinforcement-learning/ppo.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, plt, save  # noqa: E402

N = 4
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
START, GOAL = (0, 0), (3, 3)


def step(s, a):
    dr, dc = MOVES[a]
    s2 = (min(max(s[0] + dr, 0), N - 1), min(max(s[1] + dc, 0), N - 1))
    return s2, -1.0, s2 == GOAL


def features(s):
    x = np.zeros(N * N)
    x[s[0] * N + s[1]] = 1.0
    return x


def probs(theta, X):
    """Softmax policy for a batch: X [B, 16], theta [4, 16] -> [B, 4]."""
    z = X @ theta.T
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# --- 1. Collect a rollout with the current (soon to be "old") policy ------------------
def collect(theta, w, rng, n_episodes=8, horizon=50):
    eps = []
    for _ in range(n_episodes):
        s, X, A, R = START, [], [], []
        done = False
        for _ in range(horizon):
            x = features(s)
            a = rng.choice(4, p=probs(theta, x[None])[0])
            s, r, done = step(s, a)
            X.append(x), A.append(a), R.append(r)
            if done:
                break
        last_v = 0.0 if done else w @ features(s)  # truncated by the horizon: bootstrap, don't pretend it ended
        eps.append((np.array(X), np.array(A), np.array(R), last_v))
    return eps


# --- 2. Generalized advantage estimation ---------------------------------------
def gae(rewards, values, last_v, gamma=0.99, lam=0.95):
    """delta_t = r_t + gamma V(s_{t+1}) - V(s_t);  A_t = delta_t + gamma*lam*A_{t+1}."""
    adv, running = np.zeros(len(rewards)), 0.0
    next_v = last_v
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_v - values[t]
        running = delta + gamma * lam * running
        adv[t], next_v = running, values[t]
    return adv, adv + values  # advantages, value targets (returns)


# --- 3. The clipped surrogate for one sample ----------------------------------------
def clipped_objective(ratio, adv, eps=0.2):
    return np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)


# --- 4. One PPO update: K epochs over shuffled minibatches ----------------------------
def ppo_update(theta, w, X, A, adv, ret, logp_old, rng, eps=0.2, epochs=10, batch=64, lr=4.0, lr_v=0.2):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # per-batch advantage normalization
    n, clipped, seen = len(A), 0, 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            j = order[i:i + batch]
            p = probs(theta, X[j])
            logp = np.log(p[np.arange(len(j)), A[j]])
            ratio = np.exp(logp - logp_old[j])  # new / old, computed in log space
            # The gradient flows only where the min picks the unclipped term.
            active = ~(((adv[j] > 0) & (ratio > 1 + eps)) | ((adv[j] < 0) & (ratio < 1 - eps)))
            clipped += (~active).sum()
            seen += len(j)
            g = -p
            g[np.arange(len(j)), A[j]] += 1.0  # d log pi(a) / d logits
            coef = (active * ratio * adv[j])[:, None]  # d(ratio * A) / d logits = ratio * A * (onehot - p)
            theta += lr * (coef * g).T @ X[j] / len(j)  # ascend the surrogate
            w += lr_v * ((ret[j] - X[j] @ w)[:, None] * X[j]).mean(axis=0)  # critic: regress to targets
    p = probs(theta, X)
    logp_new = np.log(p[np.arange(n), A])
    approx_kl = np.mean(logp_old - logp_new)  # estimate of KL(old || new) on the rollout's samples
    return approx_kl, clipped / seen


def train(seed, updates=40, eps=0.2, epochs=10, lr=4.0):
    rng = np.random.default_rng(seed)
    theta, w = np.zeros((4, N * N)), np.zeros(N * N)
    log = []
    for _ in range(updates):
        episodes = collect(theta, w, rng)
        X = np.concatenate([e[0] for e in episodes])
        A = np.concatenate([e[1] for e in episodes])
        advs, rets = zip(*(gae(R, Xe @ w, lv) for Xe, _, R, lv in episodes))
        adv, ret = np.concatenate(advs), np.concatenate(rets)
        logp_old = np.log(probs(theta, X)[np.arange(len(A)), A])  # frozen snapshot of pi_old
        kl, clip_frac = ppo_update(theta, w, X, A, adv, ret, logp_old, rng, eps=eps, epochs=epochs, lr=lr)
        mean_return = np.mean([e[2].sum() for e in episodes])
        log.append((mean_return, kl, clip_frac))
    return theta, w, np.array(log)


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=== Clipped objective, epsilon = 0.2 ===")
    for ratio, adv in [(1.1, 2.0), (1.5, 2.0), (0.7, 2.0), (0.7, -1.0), (1.5, -1.0)]:
        un, cl = ratio * adv, np.clip(ratio, 0.8, 1.2) * adv
        pick = "clipped, gradient 0" if cl < un else "unclipped, gradient flows"
        print(f"r = {ratio}, A = {adv:+}: min({un:+.2f}, {cl:+.2f}) = {min(un, cl):+.2f}  ({pick})")

    print("\n=== GAE on a 3-step episode that ends at the goal (gamma = 0.99, lambda = 0.95) ===")
    adv, ret = gae(np.array([-1.0, -1.0, -1.0]), np.array([-2.5, -1.8, -0.9]), last_v=0.0)
    print("advantages =", np.round(adv, 6), " value targets =", np.round(ret, 6))

    print("\n=== PPO: 40 updates x (8 episodes, horizon 50), K = 10 epochs, minibatch 64, lr 4.0, 8 seeds ===")
    runs = [train(s) for s in range(8)]
    for k in (0, 4, 9, 19, 39):
        r = np.mean([lg[k, 0] for _, _, lg in runs])
        kl = np.mean([lg[k, 1] for _, _, lg in runs])
        cf = np.mean([lg[k, 2] for _, _, lg in runs])
        print(f"update {k + 1:2d}: mean return = {r:7.2f}  approx KL = {kl:.4f}  clip fraction = {cf:.2f}")
    theta, w, _ = runs[0]
    print(f"seed 0 critic V(start) = {w @ features(START):.2f} (true value of the optimal policy with gamma 0.99: "
          f"{-sum(0.99 ** k for k in range(6)):.2f})")

    print("\n=== Too aggressive: same run with K = 30 epochs and no clipping (epsilon = 100) ===")
    wild = [train(s, epochs=30, eps=100.0) for s in range(8)]
    for k in (0, 1, 4):
        print(f"update {k + 1}: mean approx KL = {np.mean([lg[k, 1] for _, _, lg in wild]):.3f}"
              f" (PPO with clipping: {np.mean([lg[k, 1] for _, _, lg in runs]):.3f})")
    for name, rs in [("clipped, K=10", runs), ("no clip, K=30", wild)]:
        print(f"{name}: final mean return per seed (last 5 updates) = "
              f"{np.round([lg[-5:, 0].mean() for _, _, lg in rs], 1)}")

    # --- figure ---------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    r = np.linspace(0, 2, 401)
    for adv, color in [(1.0, BLUE), (-1.0, ORANGE)]:
        ax1.plot(r, r * adv, color=color, ls=":", lw=1)
        ax1.plot(r, clipped_objective(r, adv), color=color, lw=2.5, label=f"A = {adv:+.0f}: min(rA, clip(r)A)")
    ax1.axvspan(0.8, 1.2, color=GRAY, alpha=0.15)
    ax1.text(1.0, 1.6, "trust region\n1 ± ε", ha="center", fontsize=8, color=INK_GRAY)
    ax1.set_xlabel("probability ratio r = π_new(a|s) / π_old(a|s)")
    ax1.set_ylabel("objective per sample")
    ax1.set_title("Clipped surrogate (ε = 0.2); dotted = unclipped")
    ax1.legend(fontsize=8, loc="lower left")
    for i, (_, _, lg) in enumerate(runs):
        ax2.plot(np.arange(1, len(lg) + 1), lg[:, 0], color=BLUE, lw=1, alpha=0.7, label="clipped, K = 10" if i == 0 else None)
    for i, (_, _, lg) in enumerate(wild):
        ax2.plot(np.arange(1, len(lg) + 1), lg[:, 0], color=PURPLE, lw=1, alpha=0.7, label="no clip, K = 30" if i == 0 else None)
    ax2.axhline(-6, color=GRAY, ls="--", lw=1)
    ax2.set_xlabel("PPO update")
    ax2.set_ylabel("mean episode return (horizon 50)")
    ax2.set_title("Training, one line per seed")
    ax2.legend(fontsize=8, loc="lower right")
    save(fig, "ppo")


INK_GRAY = "#52525b"

if __name__ == "__main__":
    main()
