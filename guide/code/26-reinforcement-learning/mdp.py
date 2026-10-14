# Adapted from AI Engineering from Scratch (MIT), phases/09-reinforcement-learning/01-mdps-states-actions-rewards
"""A Markov decision process from scratch: a 4x4 GridWorld, discounted returns, random rollouts,
iterative policy evaluation (Bellman expectation) and value iteration (Bellman optimality).

The world is synthetic: start top-left, goal bottom-right, every move costs -1, bumping a wall leaves you in place.
Run from guide/:  uv run code/26-reinforcement-learning/mdp.py
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PURPLE, plt, save  # noqa: E402

# --- 1. The five pieces: states, actions, transitions, rewards, discount ---------
N = 4
STATES = [(r, c) for r in range(N) for c in range(N)]
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
START, GOAL = (0, 0), (3, 3)


def step(s, a):
    """Transition and reward in one function. Deterministic, so P(s'|s,a) is 1 for one s'."""
    if s == GOAL:
        return s, 0.0, True  # absorbing terminal state
    dr, dc = ACTIONS[a]
    s2 = (min(max(s[0] + dr, 0), N - 1), min(max(s[1] + dc, 0), N - 1))
    return s2, -1.0, s2 == GOAL


# --- 2. Return: the discounted sum of rewards -----------------------------------
def discounted_return(rewards, gamma):
    g = 0.0
    for r in reversed(rewards):  # G_t = r_t + gamma * G_{t+1}
        g = r + gamma * g
    return g


# --- 3. A policy is a distribution over actions; roll it out ---------------------
def uniform(_s):
    return {a: 0.25 for a in ACTIONS}


def rollout(policy, rng, max_steps=500):
    s, rewards = START, []
    for _ in range(max_steps):
        dist = policy(s)
        a = rng.choices(list(dist), weights=list(dist.values()))[0]
        s, r, done = step(s, a)
        rewards.append(r)
        if done:
            break
    return rewards


# --- 4. Policy evaluation: apply the Bellman expectation equation until it stops changing
def policy_evaluation(policy, gamma, tol=1e-10):
    V = {s: 0.0 for s in STATES}
    while True:
        delta = 0.0
        for s in STATES:
            if s == GOAL:
                continue
            v = sum(p * (r + gamma * V[s2]) for a, p in policy(s).items() for s2, r, _ in [step(s, a)])
            delta, V[s] = max(delta, abs(v - V[s])), v
        if delta < tol:
            return V


# --- 5. Value iteration: the Bellman optimality equation (max instead of average) --
def value_iteration(gamma, tol=1e-10):
    V = {s: 0.0 for s in STATES}
    while True:
        delta = 0.0
        for s in STATES:
            if s == GOAL:
                continue
            v = max(r + gamma * V[s2] for a in ACTIONS for s2, r, _ in [step(s, a)])
            delta, V[s] = max(delta, abs(v - V[s])), v
        if delta < tol:
            return V


def grid(V):
    return np.array([[V[(r, c)] for c in range(N)] for r in range(N)])


def main():
    print("=== Discounted return of six -1 rewards ===")
    for gamma in (1.0, 0.9, 0.5):
        print(f"gamma = {gamma}: G = {discounted_return([-1.0] * 6, gamma):.4f}")

    rng = random.Random(42)
    rets = [sum(rollout(uniform, rng)) for _ in range(5000)]
    print(f"\n=== Uniform random policy, 5000 rollouts (undiscounted) ===")
    print(f"mean return = {np.mean(rets):.2f}, std = {np.std(rets):.2f}  (optimal = -6)")

    print("\n=== V for the uniform random policy (policy evaluation) ===")
    vals = {}
    for gamma in (0.5, 0.9, 0.99):
        vals[gamma] = policy_evaluation(uniform, gamma)
        print(f"gamma = {gamma}:")
        print(np.array2string(grid(vals[gamma]), precision=2, suppress_small=True))

    V = vals[0.9]
    s = (3, 2)
    nxt = {a: step(s, a)[0] for a in ACTIONS}
    check = sum(0.25 * (-1.0 + 0.9 * V[s2]) for s2 in nxt.values())
    print(f"\nBellman check at {s}, gamma=0.9: neighbours " +
          ", ".join(f"{a}->{s2} V={V[s2]:.3f}" for a, s2 in nxt.items()) +
          f"; 0.25 * sum(-1 + 0.9 V) = {check:.3f} = V{s} = {V[s]:.3f}")

    v_star = value_iteration(1.0)
    print("\n=== V* by value iteration (gamma = 1): minus the number of moves to the goal ===")
    print(grid(v_star))
    print("effective horizon 1/(1-gamma): " + ", ".join(f"gamma={g}: {1 / (1 - g):.0f} steps" for g in (0.5, 0.9, 0.99)))

    # --- figure: discount weights, V of the random policy, V* --------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    k = np.arange(0, 60)
    for g, color in zip((0.5, 0.9, 0.99), (ORANGE, BLUE, PURPLE)):
        axes[0].plot(k, g ** k, color=color, label=f"γ = {g} (horizon ≈ {1 / (1 - g):.0f})")
    axes[0].set_xlabel("steps into the future k")
    axes[0].set_ylabel("weight γᵏ on reward r(t+k)")
    axes[0].set_title("Discount sets the horizon")
    axes[0].legend(fontsize=8)
    for ax, M, title in [(axes[1], grid(vals[0.9]), "V of random policy (γ = 0.9)"),
                         (axes[2], grid(v_star), "V* of best policy (γ = 1)")]:
        ax.imshow(M, cmap="Blues")
        ax.grid(False)
        for r in range(N):
            for c in range(N):
                ax.text(c, r, "goal" if (r, c) == GOAL else f"{M[r, c]:.1f}", ha="center", va="center",
                        fontsize=9, color="black" if M[r, c] < M.min() / 3 else "white")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    save(fig, "mdps-states-actions-rewards")


if __name__ == "__main__":
    main()
