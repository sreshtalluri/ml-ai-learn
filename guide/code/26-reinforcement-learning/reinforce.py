# Adapted from AI Engineering from Scratch (MIT), phases/09-reinforcement-learning/06-policy-gradients-reinforce
"""REINFORCE from scratch: a softmax policy on a 4x4 GridWorld, returns-to-go, and a learned baseline.

The world is synthetic: start (0, 0), goal (3, 3), every move costs -1, walls leave you in place.
The script prints one hand-checkable update, measures gradient variance with and without a baseline,
and trains both versions over several seeds. NumPy only, fixed seeds.
Run from guide/:  uv run code/26-reinforcement-learning/reinforce.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, plt, save  # noqa: E402

N = 4
ACTIONS = ["up", "down", "left", "right"]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
START, GOAL = (0, 0), (3, 3)


def step(s, a):
    dr, dc = MOVES[a]
    s2 = (min(max(s[0] + dr, 0), N - 1), min(max(s[1] + dc, 0), N - 1))
    return s2, -1.0, s2 == GOAL


def features(s):
    x = np.zeros(N * N)
    x[s[0] * N + s[1]] = 1.0  # one-hot: a linear policy on these is a table of logits
    return x


# --- 1. The policy: softmax over linear logits --------------------------------
def softmax(z):
    e = np.exp(z - z.max())  # shift for numerical safety
    return e / e.sum()


def policy(theta, x):
    return softmax(theta @ x)  # theta: [4 actions, 16 features] -> probs: [4]


# --- 2. Sample one episode and compute returns-to-go ------------------------------
def rollout(theta, rng, max_steps=100):
    s, traj = START, []
    for _ in range(max_steps):
        x = features(s)
        p = policy(theta, x)
        a = rng.choice(4, p=p)
        s, r, done = step(s, a)
        traj.append((x, a, r, p))
        if done:
            break
    return traj


def returns_to_go(rewards, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):  # G_t = r_t + gamma * G_{t+1}
        G = r + gamma * G
        out.append(G)
    return out[::-1]


# --- 3. The REINFORCE gradient: sum of (G_t - b(s_t)) * grad log pi(a_t | s_t) -------
def grad_log_pi(x, a, p):
    """For a softmax over theta @ x: d log pi(a) / d theta = (onehot(a) - p) outer x."""
    g = -p.copy()
    g[a] += 1.0
    return np.outer(g, x)


def episode_gradient(traj, gamma, w=None):
    G = returns_to_go([r for _, _, r, _ in traj], gamma)
    grad = np.zeros((4, N * N))
    for (x, a, _, p), g_t in zip(traj, G):
        adv = g_t - (w @ x if w is not None else 0.0)  # baseline b(s) = w . x does not depend on a
        grad += adv * grad_log_pi(x, a, p)
    return grad, G


def reinforce(episodes, seed, baseline, lr=0.02, lr_v=0.1, gamma=0.99):
    rng = np.random.default_rng(seed)
    theta, w = np.zeros((4, N * N)), np.zeros(N * N)
    curve = []
    for _ in range(episodes):
        traj = rollout(theta, rng)
        grad, G = episode_gradient(traj, gamma, w if baseline else None)
        theta += lr * grad  # gradient ASCENT on expected return
        if baseline:  # fit the baseline: V(s) toward the observed return-to-go
            for (x, _, _, _), g_t in zip(traj, G):
                w += lr_v * (g_t - w @ x) * x
        curve.append(sum(r for _, _, r, _ in traj))
    return theta, curve


def true_values(theta, gamma, sweeps=3000):
    """Exact V of the current policy (policy evaluation), used as a reference baseline."""
    V = np.zeros((N, N))
    for _ in range(sweeps):
        for r in range(N):
            for c in range(N):
                if (r, c) == GOAL:
                    continue
                p = policy(theta, features((r, c)))
                V[r, c] = sum(p[a] * (-1.0 + gamma * (0.0 if (s2 := step((r, c), a)[0]) == GOAL else V[s2]))
                              for a in range(4))
    return V


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=== Worked example: one 3-step episode, gamma = 0.9 ===")
    rewards = [-1.0, -1.0, -1.0]  # (2,1) -right-> (2,2) -right-> (2,3) -down-> goal
    print("returns-to-go G_0, G_1, G_2 =", [round(g, 4) for g in returns_to_go(rewards, 0.9)])
    logits = np.array([0.0, 0.5, -0.5, 1.0])  # logits at (2,1) for up, down, left, right
    p = softmax(logits)
    g = -p.copy()
    g[3] += 1.0
    print("logits at (2,1) =", logits, " softmax probs =", p)
    print("grad of log pi(right) w.r.t. the logits = onehot(right) - p =", g)
    G0, b = returns_to_go(rewards, 0.9)[0], -3.5
    print(f"vanilla:  step = lr * G_0 * grad = 0.1 * {G0:.2f} * grad =", 0.1 * G0 * g)
    print(f"baseline: A = G_0 - b = {G0:.2f} - ({b}) = {G0 - b:.2f};  step = 0.1 * {G0 - b:.2f} * grad =", 0.1 * (G0 - b) * g)

    print("\n=== Gradient variance at the uniform random policy (500 episodes, gamma = 0.99) ===")
    theta0 = np.zeros((4, N * N))
    V = true_values(theta0, 0.99)
    w_true = V.flatten()
    rng = np.random.default_rng(7)
    idx = 3, 0  # component: logit of "right" at the start cell
    plain, based = [], []
    for _ in range(500):
        traj = rollout(theta0, rng, max_steps=1000)
        plain.append(episode_gradient(traj, 0.99)[0][idx])
        based.append(episode_gradient(traj, 0.99, w_true)[0][idx])
    plain, based = np.array(plain), np.array(based)
    print(f"V(start) under the random policy = {V[0, 0]:.2f}")
    print(f"no baseline:   mean = {plain.mean():7.2f}  std = {plain.std():7.2f}")
    print(f"baseline V(s): mean = {based.mean():7.2f}  std = {based.std():7.2f}")

    print("\n=== Training: 800 episodes, lr = 0.02, gamma = 0.99, 8 seeds ===")
    seeds, episodes = range(8), 800
    curves = {name: [reinforce(episodes, s, bl)[1] for s in seeds] for name, bl in [("vanilla", False), ("baseline", True)]}
    for name, cs in curves.items():
        first = np.mean([np.mean(c[:100]) for c in cs])
        last = [np.mean(c[-100:]) for c in cs]
        print(f"{name:<9} mean return, first 100 episodes = {first:7.2f}; last 100 per seed = {np.round(last, 1)}"
              f"; seeds within 1 of optimal: {sum(v > -7 for v in last)}/{len(last)}")
    theta, _ = reinforce(episodes, 0, True)
    print("P(action) at the start after training (baseline, seed 0):",
          {a: round(float(v), 3) for a, v in zip(ACTIONS, policy(theta, features(START)))})

    # --- figure ---------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for (name, cs), color in zip(curves.items(), [ORANGE, BLUE]):
        for i, c in enumerate(cs):  # one thin line per seed: the failure is one seed, not a slow mean
            smooth = np.convolve(c, np.ones(20) / 20, mode="valid")
            ax1.plot(np.arange(len(smooth)) + 19, smooth, color=color, lw=0.9, alpha=0.7, label=name if i == 0 else None)
    ax1.axhline(-6, color=GRAY, ls="--", lw=1)
    ax1.text(episodes * 0.6, -5, "optimal −6", fontsize=8, color=GRAY, va="bottom")
    ax1.set_ylim(-105, 0)
    ax1.set_xlabel("episode")
    ax1.set_ylabel("return")
    ax1.set_title("REINFORCE return, one line per seed (8 seeds)")
    ax1.legend(loc="lower right")
    bins = np.linspace(min(plain.min(), based.min()), max(plain.max(), based.max()), 40)
    ax2.hist(plain, bins=bins, color=ORANGE, alpha=0.6, label=f"no baseline (std {plain.std():.1f})")
    ax2.hist(based, bins=bins, color=BLUE, alpha=0.6, label=f"baseline V(s) (std {based.std():.1f})")
    ax2.set_xlabel("per-episode gradient estimate (logit of 'right' at start)")
    ax2.set_ylabel("episodes")
    ax2.set_title("Same mean, far less spread")
    ax2.legend(fontsize=8)
    save(fig, "policy-gradients-reinforce")


if __name__ == "__main__":
    main()
