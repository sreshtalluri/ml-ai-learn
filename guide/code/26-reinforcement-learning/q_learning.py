# Adapted from AI Engineering from Scratch (MIT), phases/09-reinforcement-learning/04-q-learning-sarsa
"""Tabular Q-learning and SARSA on a 5x5 cliff grid (the same world as the website's Q-learning lab).

The agent starts bottom-left, the goal is bottom-right, and the three cells between them are pits.
Every move costs -1, falling into a pit costs -100 and ends the episode. Two walls block cells in row 1.
Everything is synthetic and seeded, so every run prints the same numbers.
Run from guide/:  uv run code/26-reinforcement-learning/q_learning.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, plt, save  # noqa: E402

# --- 1. The environment ---------------------------------------------------------
LAYOUT = [
    ".....",
    ".#.#.",
    ".....",
    ".....",
    "SPPPG",
]
ROWS, COLS = len(LAYOUT), len(LAYOUT[0])
ACTIONS = ["up", "right", "down", "left"]
MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
START, GOAL = (4, 0), (4, 4)
STEP_REWARD, PIT_REWARD = -1.0, -100.0


def step(state, action):
    """Deterministic move. Walls and edges leave you in place. Returns (next_state, reward, done)."""
    r, c = state
    dr, dc = MOVES[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS) or LAYOUT[nr][nc] == "#":
        nr, nc = r, c
    cell = LAYOUT[nr][nc]
    if cell == "P":
        return (nr, nc), PIT_REWARD, True
    return (nr, nc), STEP_REWARD, cell == "G"


# --- 2. Epsilon-greedy action choice --------------------------------------------
def greedy(q, rng):
    best = max(q.values())
    return rng.choice([a for a in ACTIONS if q[a] == best])  # break ties at random


def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return rng.choice(ACTIONS)  # explore
    return greedy(Q[s], rng)  # exploit


def new_table():
    return {(r, c): {a: 0.0 for a in ACTIONS} for r in range(ROWS) for c in range(COLS)}


# --- 3. Q-learning: the target uses the best next action ------------------------
def q_learning(episodes, alpha=0.1, gamma=1.0, eps=0.1, seed=0, max_steps=100):
    rng, Q, returns = random.Random(seed), new_table(), []
    for _ in range(episodes):
        s, total = START, 0.0
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, eps, rng)
            s2, r, done = step(s, a)
            target = r if done else r + gamma * max(Q[s2].values())
            Q[s][a] += alpha * (target - Q[s][a])
            s, total = s2, total + r
            if done:
                break
        returns.append(total)
    return Q, returns


# --- 4. SARSA: the target uses the action the agent will actually take -----------
def sarsa(episodes, alpha=0.1, gamma=1.0, eps=0.1, seed=0, max_steps=100):
    rng, Q, returns = random.Random(seed), new_table(), []
    for _ in range(episodes):
        s, total = START, 0.0
        a = epsilon_greedy(Q, s, eps, rng)
        for _ in range(max_steps):
            s2, r, done = step(s, a)
            a2 = epsilon_greedy(Q, s2, eps, rng)
            target = r if done else r + gamma * Q[s2][a2]
            Q[s][a] += alpha * (target - Q[s][a])
            s, a, total = s2, a2, total + r
            if done:
                break
        returns.append(total)
    return Q, returns


# --- 5. Ground truth by value iteration (we know the model here) -----------------
def value_iteration(gamma=1.0, sweeps=200):
    Q = new_table()
    for _ in range(sweeps):
        for s in Q:
            if LAYOUT[s[0]][s[1]] in "PG":
                continue
            for a in ACTIONS:
                s2, r, done = step(s, a)
                Q[s][a] = r if done else r + gamma * max(Q[s2].values())
    return Q


def greedy_path(Q, max_len=30):
    s, path = START, [START]
    while LAYOUT[s[0]][s[1]] not in "PG" and len(path) <= max_len:
        a = max(ACTIONS, key=lambda x: Q[s][x])  # first best action, deterministic
        s, _, _ = step(s, a)
        path.append(s)
    return path


def show_policy(Q, title):
    arrow = {"up": "^", "right": ">", "down": "v", "left": "<"}
    print(f"  {title}")
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            cell = LAYOUT[r][c]
            row.append(cell if cell in "#PG" else arrow[max(ACTIONS, key=lambda x: Q[(r, c)][x])])
        print("   " + " ".join(row))


def mean(xs):
    return sum(xs) / len(xs)


def main():
    print("=== One update by hand (alpha = 0.5, gamma = 0.9) ===")
    q_sa, r, gamma, alpha = -2.0, -1.0, 0.9, 0.5
    q_next = {"up": -3.0, "right": -1.5, "down": -4.0, "left": -2.0}
    t_q = r + gamma * max(q_next.values())
    t_s = r + gamma * q_next["down"]  # SARSA: the agent explored and will take "down"
    print(f"Q-learning target = -1 + 0.9 * {max(q_next.values())} = {t_q:.2f};"
          f" new Q = {q_sa} + 0.5 * ({t_q:.2f} - ({q_sa})) = {q_sa + alpha * (t_q - q_sa):.3f}")
    print(f"SARSA target      = -1 + 0.9 * {q_next['down']} = {t_s:.2f};"
          f" new Q = {q_sa} + 0.5 * ({t_s:.2f} - ({q_sa})) = {q_sa + alpha * (t_s - q_sa):.3f}")

    episodes, seeds = 500, range(10)
    runs = {"Q-learning": [q_learning(episodes, seed=s) for s in seeds],
            "SARSA": [sarsa(episodes, seed=s) for s in seeds]}
    q_star = value_iteration()
    print(f"\n=== 5x5 cliff grid: {episodes} episodes, alpha=0.1, gamma=1.0, epsilon=0.1, {len(seeds)} seeds ===")
    print(f"optimal return (6 moves along the pits) = {max(q_star[START].values()):.0f}")
    for name, rs in runs.items():
        Q0 = rs[0][0]
        lens = [len(greedy_path(Q)) - 1 for Q, _ in rs]
        online = mean([mean(ret[-100:]) for _, ret in rs])
        print(f"{name:<10}  greedy path length per seed = {lens}"
              f"  mean return while exploring (last 100 eps) = {online:7.2f}")
        show_policy(Q0, f"{name} greedy policy (seed 0)")
    Qq = runs["Q-learning"][0][0]
    on_path = max(abs(max(Qq[s].values()) - max(q_star[s].values())) for s in greedy_path(Qq)[:-1])
    anywhere = max(abs(Qq[s][a] - q_star[s][a]) for s in q_star for a in ACTIONS if LAYOUT[s[0]][s[1]] not in "PG#")
    print(f"Q-learning, seed 0: max Q(start) = {max(Qq[START].values()):.2f} (true {max(q_star[START].values()):.0f});"
          f" max |V - V*| on the greedy path = {on_path:.2f}; max |Q - Q*| over all pairs = {anywhere:.2f}")

    # --- figure: learning curves and the two greedy paths -----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for (name, rs), color in zip(runs.items(), [BLUE, ORANGE]):
        avg = [mean([ret[i] for _, ret in rs]) for i in range(episodes)]
        smooth = [mean(avg[max(0, i - 19): i + 1]) for i in range(episodes)]
        ax1.plot(smooth, color=color, label=name)
    ax1.axhline(-6, color=GRAY, ls="--", lw=1)
    ax1.text(5, -5, "optimal −6", va="bottom", fontsize=8, color=GRAY)
    ax1.set_ylim(-60, 0)
    ax1.set_xlabel("episode")
    ax1.set_ylabel("return while exploring (ε = 0.1)")
    ax1.set_title("Return per episode (10 seeds, 20-episode mean)")
    ax1.legend(loc="lower right")

    for r in range(ROWS):
        for c in range(COLS):
            cell = LAYOUT[r][c]
            face = {"#": "#52525b", "P": "#fecaca", "G": "#bbf7d0", "S": "#e0e7ff"}.get(cell, "white")
            ax2.add_patch(plt.Rectangle((c, ROWS - 1 - r), 1, 1, facecolor=face, edgecolor="#d4d4d8"))
            if cell in "SPG":
                ax2.text(c + 0.5, ROWS - 0.5 - r, {"S": "start", "P": "pit", "G": "goal"}[cell],
                         ha="center", va="center", fontsize=8)
    for (name, rs), color, off in zip(runs.items(), [BLUE, ORANGE], [-0.08, 0.08]):
        path = greedy_path(rs[0][0])
        ax2.plot([c + 0.5 + off for _, c in path], [ROWS - 0.5 - r + off for r, _ in path],
                 color=color, lw=2.5, marker="o", ms=3, label=f"{name}: {len(path) - 1} moves")
    ax2.set_xlim(0, COLS)
    ax2.set_ylim(0, ROWS)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Greedy path after training (seed 0)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=8, frameon=False)
    save(fig, "q-learning-sarsa")


if __name__ == "__main__":
    main()
