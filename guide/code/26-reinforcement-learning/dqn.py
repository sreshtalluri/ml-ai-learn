# Adapted from AI Engineering from Scratch (MIT), phases/09-reinforcement-learning/05-dqn
"""Deep Q-network from scratch in NumPy: a tiny MLP Q-function, an experience replay buffer,
a target network, epsilon decay, and the Huber TD loss, trained on a 4x4 GridWorld.

The environment is a toy (start (0,0), goal (3,3), reward -1 per step). The TD-target numbers
in the worked example are hand-picked "mid-training" estimates, not outputs of a trained net.
Deterministic: fixed seeds. Run from guide/:  uv run code/26-reinforcement-learning/dqn.py
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, plt, save  # noqa: E402

# --- 1. Environment: 4x4 GridWorld ------------------------------------------
GRID, GOAL = 4, (3, 3)
ACTIONS = ["up", "down", "left", "right"]
DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def step(s, a):
    r, c = s
    dr, dc = DELTAS[a]
    nxt = (min(max(r + dr, 0), GRID - 1), min(max(c + dc, 0), GRID - 1))  # walls: stay put
    return nxt, -1.0, nxt == GOAL


def features(s):  # one-hot state, shape [16]
    x = np.zeros(GRID * GRID)
    x[s[0] * GRID + s[1]] = 1.0
    return x


# --- 2. Q-network: 16 -> 32 (ReLU) -> 4 --------------------------------------
def init_net(rng, n_in=16, n_hidden=32, n_out=4):
    return {"W1": rng.normal(0, 0.3, (n_in, n_hidden)), "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, 0.3, (n_hidden, n_out)), "b2": np.zeros(n_out)}


def forward(net, X):  # X: [B, 16] -> Q: [B, 4], hidden: [B, 32]
    H = np.maximum(0.0, X @ net["W1"] + net["b1"])
    return H @ net["W2"] + net["b2"], H


# --- 3. Huber TD loss and one gradient step ----------------------------------
def huber(delta, k=1.0):
    """Quadratic for |delta| <= k, linear beyond: the gradient is clipped to [-k, k]."""
    a = np.abs(delta)
    return np.where(a <= k, 0.5 * delta**2, k * (a - 0.5 * k)), np.clip(delta, -k, k)


def train_step(online, target, batch, gamma, lr):
    S, A, R, S2, D = batch                                     # shapes [B,16], [B], [B], [B,16], [B]
    q_next, _ = forward(target, S2)                            # frozen network builds the target
    y = R + gamma * (1.0 - D) * q_next.max(axis=1)             # no bootstrap past a terminal state
    Q, H = forward(online, S)
    idx = np.arange(len(A))
    loss, g = huber(Q[idx, A] - y)                             # g = dLoss/dQ(s,a), clipped
    dQ = np.zeros_like(Q)
    dQ[idx, A] = g / len(A)                                    # only the taken action gets gradient
    dH = (dQ @ online["W2"].T) * (H > 0)
    grads = {"W2": H.T @ dQ, "b2": dQ.sum(0), "W1": S.T @ dH, "b1": dH.sum(0)}
    for k in online:
        online[k] -= lr * grads[k]
    return loss.mean()


# --- 4. Replay buffer, epsilon decay, and the outer loop ---------------------
def run_dqn(seed, episodes=300, sync_every=100, capacity=2000, batch=32, warmup=100,
            gamma=0.99, lr=0.05, eps_end=0.05, eps_decay_eps=150, max_steps=50):
    rng = np.random.default_rng(seed)
    online = init_net(rng)
    target = {k: v.copy() for k, v in online.items()}
    buffer = deque(maxlen=capacity)                            # oldest transitions fall off
    returns, q00, steps = [], [], 0
    for ep in range(episodes):
        eps = max(eps_end, 1.0 - ep / eps_decay_eps)           # linear decay 1.0 -> 0.05
        s, total = (0, 0), 0.0
        for _ in range(max_steps):
            if rng.random() < eps:
                a = int(rng.integers(4))
            else:
                a = int(forward(online, features(s)[None])[0][0].argmax())
            s2, r, done = step(s, a)
            buffer.append((features(s), a, r, features(s2), float(done)))
            total += r
            if len(buffer) >= warmup:
                picks = rng.choice(len(buffer), batch, replace=False)   # uniform, decorrelated
                S, A, R, S2, D = zip(*(buffer[i] for i in picks))
                train_step(online, target, (np.array(S), np.array(A), np.array(R), np.array(S2), np.array(D)), gamma, lr)
            steps += 1
            if steps % sync_every == 0:
                target = {k: v.copy() for k, v in online.items()}      # hard copy theta -> theta^-
            s = s2
            if done:
                break
        returns.append(total)
        q00.append(forward(online, features((0, 0))[None])[0][0].max())
    return np.array(returns), np.array(q00), online


def greedy_grid(net):
    arrows = "^v<>"
    rows = []
    for r in range(GRID):
        rows.append(" ".join("G" if (r, c) == GOAL else arrows[int(forward(net, features((r, c))[None])[0][0].argmax())]
                             for c in range(GRID)))
    return rows


def worked_example():
    gamma = 0.99
    q_target_next = np.array([-5.6, -4.9, -5.8, -5.0])   # target net, s' = (0,1)
    q_online_next = np.array([-5.5, -5.1, -5.7, -4.95])  # online net, s' = (0,1)
    q_sa = -5.2                                           # online Q((0,0), right)
    y = -1.0 + gamma * q_target_next.max()
    delta = q_sa - y
    loss, g = huber(np.array([delta]))
    a_star = int(q_online_next.argmax())
    y_double = -1.0 + gamma * q_target_next[a_star]
    print("=== Worked example: one transition (0,0) --right--> (0,1), r = -1 ===")
    print(f"DQN target   y = -1 + 0.99 * max{q_target_next.tolist()} = {y:.3f}")
    print(f"TD error     delta = Q(s,a) - y = {q_sa} - ({y:.3f}) = {delta:.3f}")
    print(f"Huber loss   0.5 * delta^2 = {loss[0]:.4f}, gradient = {g[0]:.3f}")
    big = np.array([3.0])
    print(f"if delta = 3: Huber = {huber(big)[0][0]:.1f} (grad {huber(big)[1][0]:.0f}), MSE 0.5*delta^2 = {0.5 * 9:.1f} (grad 3)")
    print(f"Double DQN   online picks {ACTIONS[a_star]}, target scores it {q_target_next[a_star]} -> y = {y_double:.3f}")
    print()


def main():
    worked_example()
    v_star = -sum(0.99**k for k in range(6))
    seeds = [0, 1, 2]
    runs = {"target net (sync every 100 steps)": 100, "no target net (sync every step)": 1}
    results = {}
    for name, C in runs.items():
        out = [run_dqn(seed, sync_every=C) for seed in seeds]
        results[name] = out
        last = np.mean([o[0][-50:].mean() for o in out])
        first_good = [int(np.argmax(np.convolve(o[0], np.ones(20) / 20, "valid") >= -8)) + 19 for o in out]
        print(f"=== DQN, {name}, 300 episodes, seeds {seeds} ===")
        print(f"mean return, last 50 episodes: {last:.2f}   (optimal -6)")
        print(f"max Q(0,0) at the end per seed: {[round(float(o[1][-1]), 2) for o in out]}   (true V*(0,0) = {v_star:.3f})")
        print(f"lowest max Q(0,0) during training per seed: {[round(float(o[1].min()), 1) for o in out]}")
        print(f"episode where the 20-episode mean first reaches -8, per seed: {first_good}")
        print()
    print("greedy policy, seed 0, with target net:")
    for row in greedy_grid(results["target net (sync every 100 steps)"][0][2]):
        print("  " + row)
    print()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for (name, out), color in zip(results.items(), [BLUE, ORANGE]):
        for i, (ret, q, _) in enumerate(out):
            smooth = np.convolve(ret, np.ones(20) / 20, "valid")
            ax1.plot(np.arange(19, len(ret)), smooth, color=color, alpha=0.85, label=name if i == 0 else None)
            ax2.plot(q, color=color, alpha=0.85, label=name if i == 0 else None)
    ax1.axhline(-6, color=GRAY, ls="--", lw=1, label="optimal return (-6)")
    ax1.set(title="Episode return (20-episode mean, 3 seeds)", xlabel="episode", ylabel="return", ylim=(-52, 0))
    ax2.axhline(v_star, color=GRAY, ls="--", lw=1, label=f"true V*(0,0) = {v_star:.2f}")
    ax2.set(title="Network's max Q(start) during training", xlabel="episode", ylabel="max_a Q((0,0), a)")
    ax1.legend(fontsize=8, loc="lower right")
    ax2.legend(fontsize=8, loc="lower right")
    fig.suptitle("DQN on a synthetic 4x4 GridWorld (reward -1 per step)", fontsize=10)
    save(fig, "dqn")


if __name__ == "__main__":
    main()
