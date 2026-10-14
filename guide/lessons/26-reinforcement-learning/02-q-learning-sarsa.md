---
title: Q-learning and SARSA
summary: Learn action values from experience with one-step temporal-difference updates, compute Q-learning and SARSA targets by hand, and explain why the two take different paths along a cliff.
skill: deep-learning
minutes: 45
prerequisites: [mdps-states-actions-rewards]
related: [dqn, policy-gradients-reinforce, ppo]
adapted_from: [phases/09-reinforcement-learning/04-q-learning-sarsa]
---

# Q-learning and SARSA

> **Mental model.** Keep a scorecard with one number for every (place, move) pair: "how much total reward do I expect if I make this move here?" After every move, compare what the card predicted with what just happened (the reward, plus what the card says about where you landed) and nudge the entry a little toward that. Q-learning fills in "where you landed" assuming you'll play perfectly from there. SARSA fills it in with the move you will actually make, mistakes included.

**You will learn to**
- Write the temporal-difference (TD) update and compute a Q-learning and a SARSA update by hand.
- Explain ε-greedy exploration and why optimistic initial values also explore.
- Distinguish on-policy (SARSA) from off-policy (Q-learning) learning, and predict which takes the safer route on a cliff.
- Tune the learning rate α, discount γ, and exploration rate ε, and read a learning curve.
- Recognize the limits of a table: unvisited states, maximization bias, and state spaces too big to enumerate.

**Why it matters.** The [previous lesson](01-mdps-states-actions-rewards.md) solved an MDP when we knew the rules. Usually you don't: you can only act and observe. TD learning updates after every single step, with no model and no need to wait for the episode to end. The same one-line update, with a neural network in place of the table, is [DQN](03-dqn.md); the same "target minus estimate" error drives the critic in [PPO](05-ppo.md). Interviewers ask about the cliff-walking difference because it tests whether you understand on-policy versus off-policy, not just the formula.

## 1. Intuition

Our world is a 5×5 grid. The agent starts in the bottom-left corner and wants the bottom-right corner. The three cells between them are pits: falling in costs −100 and ends the episode. Every other move costs −1. Two walls sit in the second row.

The agent keeps a **Q-table**: 25 cells × 4 moves = 100 numbers, all starting at 0. Each number $Q(s, a)$ is a guess of the total reward from making move $a$ in cell $s$ and then carrying on.

After each move it sees one reward $r$ and the cell it landed in, $s'$. It builds a better guess for $Q(s, a)$, called the **target**:

> target = reward now + γ × (value of being in $s'$)

and moves $Q(s, a)$ a fraction α of the way toward it. This is a **temporal-difference** update: it learns from the difference between consecutive guesses, so values spread backward from the goal one step at a time. Nothing needs the episode to finish first.

The two algorithms disagree on one thing: the value of being in $s'$.
- **Q-learning** uses the best move in $s'$: $\max_{a'} Q(s', a')$. It learns how good things would be if the agent played greedily from here on.
- **SARSA** uses the move the agent will actually make next, $Q(s', a')$ with $a'$ already chosen. Its name lists what one update uses: $s, a, r, s', a'$.

**Exploration.** If the agent always took its best-looking move, it would never discover anything better. **ε-greedy** takes a random move with probability ε (say 10%) and the best-looking move otherwise.

**The cliff.** Here the difference becomes visible. The shortest route runs along the row just above the pits: 6 moves. But with ε = 0.1, one move in ten is random, and a random "down" next to a pit costs 100. Q-learning ignores its own randomness when it computes targets, so it learns that the edge path is best and keeps walking it, falling in now and then. SARSA's targets include those random moves, so the cells beside the pits look bad, and it learns a path one row further up: 8 moves, but fewer falls.

> [!IMPORTANT]
> **Off-policy vs on-policy.** Q-learning is *off-policy*: it learns the value of the greedy policy while behaving with a different, exploring one. SARSA is *on-policy*: it learns the value of the policy it is actually running. If exploration will continue when the agent is deployed, SARSA's answer is the more honest one. If you will switch exploration off, Q-learning's is the better plan.

## 2. Visualization

<!-- lab:q-learning -->
![Left: mean return per episode over 500 episodes for Q-learning and SARSA on the synthetic 5x5 cliff grid, averaged over 10 seeds. Both climb from below -60 toward -10 to -20 within about 150 episodes; afterward SARSA sits higher, around -13, than Q-learning, around -16, because Q-learning sometimes falls into a pit while exploring. Right: the greedy paths after training. Q-learning walks along the row directly above the three pits in 6 moves; SARSA goes one row higher and takes 8 moves.](../../figures/q-learning-sarsa.png)

*Synthetic grid, ε = 0.1, α = 0.1, γ = 1, 10 seeds. Q-learning finds the shortest path but earns less while exploring; SARSA takes a safer path and earns more while exploring.*

*Interactive version: [open the lab on the website](https://sreshtalluri.github.io/ml-ai-learn/labs/q-learning/).*
<!-- /lab -->

In the lab, each cell shows four arrows, one per move, sized by how good that move looks relative to the others; the teal arrow is the greedy choice. The dashed orange line is the greedy path from the start. The panel under the curve writes out the last update with real numbers.

**Try it** (predict first, then check):
1. Press **Step** a few times from a fresh start. Before each, predict which single Q-value will change and whether it goes up or down. Why is the first update always $0 + \alpha(-1 - 0)$ or $0 + \alpha(-100 - 0)$?
2. Run 300 episodes with Q-learning, then switch to SARSA and run 300 again. Predict the length of each greedy path, and which one has the higher mean return.
3. Set ε to 0 and train from scratch. Predict whether the agent still finds the goal. (Hint: every untried move still has its initial value of 0, while every tried move costs at least −1.)
4. Set α to 1. Predict what happens to the return curve when the agent occasionally falls into a pit.

## 3. The math

### Symbols

| Symbol | Meaning | Shape / type |
|---|---|---|
| $Q(s, a)$ | current estimate of the action value | table, 25 × 4 here |
| $s, a, r, s'$ | state, action, reward, next state of one transition | cell, move, scalar, cell |
| $a'$ | next action: argmax for Q-learning, the sampled action for SARSA | move |
| $\alpha$ | learning rate (step size) | scalar in $(0, 1]$ |
| $\gamma$ | discount | scalar in $[0, 1]$ |
| $\varepsilon$ | exploration probability in ε-greedy | scalar in $[0, 1]$ |
| $y$ | TD target | scalar |
| $\delta$ | TD error, $y - Q(s, a)$ | scalar |

### The updates

Both move the estimate toward a one-step target:

```math
Q(s, a) \leftarrow Q(s, a) + \alpha\,\big[\,y - Q(s, a)\,\big]
```

```math
y_{\text{Q-learning}} = r + \gamma \max_{a'} Q(s', a') \qquad\qquad y_{\text{SARSA}} = r + \gamma\, Q(s', a'), \quad a' \sim \varepsilon\text{-greedy}(Q, s')
```

If $s'$ is terminal, the target is just $r$: there is no future to bootstrap from. **Bootstrapping** means the target uses our own current estimate $Q(s', \cdot)$. That makes the target biased (the estimate may be wrong) but low-variance (it depends on one random step, not a whole episode), which is the trade TD makes against Monte Carlo methods that wait for the full return.

The Q-learning target is a sampled version of the Bellman *optimality* equation from the previous lesson, so Q-learning converges to $Q^*$, given that every state-action pair keeps being visited and α shrinks appropriately (formally $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$). SARSA's target is a sample of the Bellman *expectation* equation for the ε-greedy policy itself, so it converges to that policy's values, and to $Q^*$ only if ε decays to 0 over time.

**Expected SARSA** replaces the sampled $Q(s', a')$ by its average under the policy, $\sum_{a'} \pi(a' \mid s') Q(s', a')$: same target in expectation as SARSA, less noise.

### Worked example: one update, two rules

The agent is in cell $s$ with $Q(s, \text{right}) = -2.0$. It moves right, gets $r = -1$, and lands in $s'$ where the table holds:

| $a'$ | up | right | down | left |
|---|---|---|---|---|
| $Q(s', a')$ | −3.0 | −1.5 | −4.0 | −2.0 |

Take $\alpha = 0.5$, $\gamma = 0.9$. Suppose the ε-greedy draw at $s'$ came up "explore" and picked **down**.

**Q-learning:**
1. Best next value: $\max(-3.0, -1.5, -4.0, -2.0) = -1.5$.
2. Target: $y = -1 + 0.9 \times (-1.5) = -1 - 1.35 = -2.35$.
3. TD error: $\delta = -2.35 - (-2.0) = -0.35$.
4. New value: $-2.0 + 0.5 \times (-0.35) = -2.0 - 0.175 = -2.175$.

**SARSA:**
1. Next action's value: $Q(s', \text{down}) = -4.0$.
2. Target: $y = -1 + 0.9 \times (-4.0) = -1 - 3.6 = -4.6$.
3. TD error: $\delta = -4.6 - (-2.0) = -2.6$.
4. New value: $-2.0 + 0.5 \times (-2.6) = -2.0 - 1.3 = -3.3$.

Same transition, very different updates. SARSA was charged for the exploratory "down" it is about to take; Q-learning assumed it would take the best move. Near a cliff, that exploratory move is the one that falls in, which is exactly why SARSA learns to keep its distance.

**Ground truth check.** On this grid the shortest path is 6 moves, so with $\gamma = 1$, $V^*(\text{start}) = -6$. After 500 episodes the script's Q-learning table has $\max_a Q(\text{start}, a) = -6.00$ and matches $V^*$ exactly on every cell of its greedy path. Yet the largest error over *all* 100 entries is 34.87: pairs the agent rarely tries, like walking into the pits from the top rows, still hold rough guesses. A tabular agent is only as good as its coverage.

## 4. Implementation

The full script is [`code/26-reinforcement-learning/q_learning.py`](../../code/26-reinforcement-learning/q_learning.py) (standard library plus matplotlib for the figure). It trains both algorithms on the same grid as the lab, over 10 seeds, and checks against value iteration.

### Build it from scratch

**Step 1: the environment.** A string map keeps the world readable and easy to edit. The step function returns the next cell, the reward, and whether the episode ended; walls and edges leave the agent where it was.

```python
LAYOUT = [
    ".....",
    ".#.#.",
    ".....",
    ".....",
    "SPPPG",
]
def step(state, action):
    r, c = state
    dr, dc = MOVES[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS) or LAYOUT[nr][nc] == "#":
        nr, nc = r, c
    cell = LAYOUT[nr][nc]
    if cell == "P":
        return (nr, nc), PIT_REWARD, True
    return (nr, nc), STEP_REWARD, cell == "G"
```

*Predict:* from the start $(4, 0)$, what does moving right return?

**Step 2: ε-greedy with random tie-breaking.** At the start every Q-value is 0, so every move ties. Always picking the first tied action (say "up") would bias the agent; breaking ties at random avoids it.

```python
def greedy(q, rng):
    best = max(q.values())
    return rng.choice([a for a in ACTIONS if q[a] == best])  # break ties at random

def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return rng.choice(ACTIONS)  # explore
    return greedy(Q[s], rng)  # exploit
```

*Predict:* with ε = 0.1 and four actions, what is the probability the agent takes the greedy action (assuming no ties)? (Not 0.9.)

**Step 3: Q-learning.** Choose, act, update toward the max over next actions. The `if done` branch stops bootstrapping from a terminal state. Note that `max_steps` cuts off long episodes without treating the cutoff as terminal.

```python
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
```

*Predict:* if you deleted the `r if done else` part, what would the target be after falling into a pit, and would it matter here? (Pit and goal cells' Q-values are never updated, so they stay 0.)

**Step 4: SARSA.** The only real change: choose the next action *before* the update and use its value in the target, then actually take that action on the next step.

```python
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
```

*Predict:* if you set ε = 0, are Q-learning and SARSA the same algorithm? (Look at what `a2` becomes.)

### Use the library

Tabular methods are short enough that most people write them by hand, as above, on top of a [Gymnasium](https://gymnasium.farama.org/) environment. Gymnasium ships the classic cliff world as `CliffWalking-v1` (`-v0` in releases before 1.1; 4×12, a cliff row costing −100 that sends you back to the start), with integer states you can index into a NumPy table:

```python
import gymnasium as gym
import numpy as np

env = gym.make("CliffWalking-v1")          # "CliffWalking-v0" on older Gymnasium
Q = np.zeros((env.observation_space.n, env.action_space.n))
s, _ = env.reset(seed=0)
a = env.action_space.sample()
s2, r, terminated, truncated, _ = env.step(a)
Q[s, a] += 0.1 * (r + (0 if terminated else Q[s2].max()) - Q[s, a])   # one Q-learning update
```

The environment adds seeded resets and the `terminated` versus `truncated` split. Once the state is an image or a vector of floats, the table stops working and you need a function approximator; deep RL libraries such as Stable-Baselines3 start there, with [DQN](03-dqn.md).

### Run it

```bash
uv run code/26-reinforcement-learning/q_learning.py
```

```
Q-learning target = -1 + 0.9 * -1.5 = -2.35; new Q = -2.0 + 0.5 * (-2.35 - (-2.0)) = -2.175
SARSA target      = -1 + 0.9 * -4.0 = -4.60; new Q = -2.0 + 0.5 * (-4.60 - (-2.0)) = -3.300
=== 5x5 cliff grid: 500 episodes, alpha=0.1, gamma=1.0, epsilon=0.1, 10 seeds ===
optimal return (6 moves along the pits) = -6
Q-learning  greedy path length per seed = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]  mean return while exploring (last 100 eps) =  -16.12
SARSA       greedy path length per seed = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]  mean return while exploring (last 100 eps) =  -12.79
Q-learning, seed 0: max Q(start) = -6.00 (true -6); max |V - V*| on the greedy path = 0.00; max |Q - Q*| over all pairs = 34.87
saved figures/q-learning-sarsa.png
```

## 5. Engineering

**Where tabular TD is still used.** Small discrete problems: toy control, simple games, inventory with a few stock levels, teaching, and as a sanity check for a deep RL pipeline (if your DQN can't solve a grid that a table solves in a second, the bug is in your code). Beyond about a million state-action pairs, or with continuous states, you need function approximation.

**Choosing which.** Use Q-learning when you can explore freely in training (a simulator) and will act greedily when deployed. Prefer SARSA or Expected SARSA when exploration is costly or will continue in deployment, since it accounts for its own mistakes. Q-learning's off-policy nature has a second benefit: it can learn from data generated by any policy, which is what makes [experience replay in DQN](03-dqn.md) possible.

**Hyperparameters.**
- **α:** constant 0.05 to 0.3 is common for tabular problems. Too high and a single unlucky pit fall swings a value by a lot (try α = 1 in the lab); too low and learning crawls.
- **ε:** start high (often 1.0) and decay to a small floor like 0.05. Theory needs "greedy in the limit with infinite exploration" (GLIE) for convergence to the optimal policy.
- **γ:** set from the horizon you care about, as in the previous lesson.
- **Initial values:** with negative step rewards, initializing at 0 is *optimistic*: every untried move looks better than every tried one, so the agent explores even with ε = 0. Pessimistic initial values can lock in the first path found.

**Maximization bias.** $\max_{a'} Q(s', a')$ over noisy estimates is biased upward: the max picks whichever estimate happens to be too high. With noisy rewards Q-learning overestimates values. **Double Q-learning** keeps two tables, picks the best action with one, and evaluates it with the other; [Double DQN](03-dqn.md) is the deep version.

> [!WARNING]
> **Failure modes.** Never-visited states keep their initial values, so the greedy policy is meaningless there (see the top rows of the trained Q-learning table). Treating a time-limit cutoff as terminal teaches the agent that the world ends at step 100. Rewards on wildly different scales (−100 vs −1) make large α unstable. Using a list or unrounded float vector as a dictionary key silently creates duplicate states.

### Common mistakes

- Bootstrapping from a terminal state (the target after a terminal transition is just $r$).
- In SARSA, choosing $a'$ for the update and then choosing a *different* action to execute. The update and the behavior must use the same $a'$.
- Always breaking ties toward the first action, which biases early exploration.
- Judging Q-learning by its return while exploring and concluding it is worse than SARSA. Evaluate both greedily (ε = 0) if the deployed agent will be greedy.
- Leaving ε fixed at 1.0 (pure random) and reading the learning curve as "not learning".

## 6. Knowledge check

<!-- quiz:q-learning-sarsa -->
**[Take the Q-learning and SARSA quiz](../../quizzes/q-learning-sarsa.md)**
<!-- /quiz -->

**Practice exercise.** $Q(s, \text{up}) = -5$. The agent moves up, gets $r = -1$, and lands in $s'$ with $Q(s', \cdot) = [-4, -3, -6, -5]$ (up, right, down, left). The next action chosen by ε-greedy is "left". With $\alpha = 0.1$ and $\gamma = 1$, compute the new $Q(s, \text{up})$ under Q-learning and under SARSA.

<details>
<summary>Solution</summary>

Q-learning: $y = -1 + 1 \times (-3) = -4$; $\delta = -4 - (-5) = 1$; new $= -5 + 0.1 \times 1 = -4.9$.

SARSA: $y = -1 + 1 \times (-5) = -6$; $\delta = -6 - (-5) = -1$; new $= -5 + 0.1 \times (-1) = -5.1$.

The same transition raises the estimate under Q-learning and lowers it under SARSA, because the exploratory "left" is worse than the greedy "right".

</details>

**Implementation challenge.** Add Expected SARSA to [`q_learning.py`](../../code/26-reinforcement-learning/q_learning.py). Under ε-greedy, the expected next value is $(1 - \varepsilon)\max_{a'} Q(s', a') + \frac{\varepsilon}{4}\sum_{a'} Q(s', a')$ (ignoring ties). Which path length does it learn on the cliff grid with ε = 0.1, and how does its mean return while exploring compare with SARSA's?

<details>
<summary>Solution sketch</summary>

Copy `q_learning` and replace the target with `r + gamma * ((1 - eps) * max(q2.values()) + eps * sum(q2.values()) / 4)` where `q2 = Q[s2]`. It is on-policy like SARSA, so it should also avoid the pit edge, but its targets have less noise (no sampled $a'$), so its learning curve is smoother and usually at least as high as SARSA's. Verify by running 10 seeds as the script does.

</details>

## Summary

- TD learning updates $Q(s, a)$ after every step toward a one-step target, reward plus discounted estimate of the next state: bootstrapped, biased, low-variance, and model-free.
- Q-learning's target uses $\max_{a'} Q(s', a')$ (off-policy, learns $Q^*$); SARSA's uses the next action actually taken (on-policy, learns the values of the exploring policy).
- On a cliff, Q-learning learns the shortest path along the edge and falls in while exploring; SARSA learns a safer, longer path and earns more while exploring.
- ε-greedy and optimistic initial values drive exploration; α, γ, ε need tuning, and unvisited states stay wrong.
- The max in Q-learning overestimates under noise (fix: Double Q-learning), and tables don't scale, which motivates DQN.

**Next:** [Deep Q-networks (DQN)](03-dqn.md)

**Related:** [MDPs, states, actions, and rewards](01-mdps-states-actions-rewards.md) · [Policy gradients and REINFORCE](04-policy-gradients-reinforce.md) · [PPO](05-ppo.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Monte Carlo Methods — Learning from Complete Episodes](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/03-monte-carlo-methods)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Temporal Difference — Q-Learning & SARSA](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/04-q-learning-sarsa).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>What is the difference between Q-learning and SARSA, and which is on-policy?</strong></summary>

Both are one-step temporal-difference methods that update Q(s, a) toward a target of reward plus the discounted value of the next state. The difference is which next value they use. Q-learning uses the max over next actions, so its target describes the greedy policy no matter how the agent actually behaves; that makes it off-policy, and it converges to Q*. SARSA uses the value of the action the agent will actually take next, sampled from its ε-greedy policy, so it learns the value of the policy it is running; that is on-policy. The practical consequence shows up when exploration is risky: on a cliff, Q-learning learns the shortest path along the edge and keeps falling in while exploring, while SARSA learns a safer path because its targets include its own random moves.

</details>

<details>
<summary><strong>When would you choose SARSA over Q-learning, or vice versa?</strong></summary>

I choose based on whether exploration continues in deployment and how costly mistakes are. If I train in a simulator and will deploy a greedy policy, Q-learning gives the better plan, because it learns optimal values regardless of the exploration noise. If the agent keeps exploring in the real system, or mistakes during learning are expensive, SARSA or Expected SARSA is safer, because it optimizes the return of the policy I'm actually running. Off-policy learning also has a data benefit: Q-learning can reuse transitions from any policy, such as a replay buffer or logged data, while SARSA needs fresh on-policy data. A practical middle ground is Expected SARSA, which averages over the next action instead of sampling it, reducing variance while staying on-policy.

</details>

<details>
<summary><strong>Your tabular Q-learning agent's learning curve is flat after thousands of episodes. What do you check?</strong></summary>

First, whether the agent ever reaches a reward signal: if the goal is never found, nothing propagates back, so I'd log how many episodes reach it. Then exploration: ε fixed at 1 means the behavior is random no matter what the table says, and ε at 0 with pessimistic initial values can lock in the first path. I'd check the update for the classic bugs: bootstrapping from terminal states, treating time-limit cutoffs as terminal, or a state key that isn't hashable consistently, so updates go to duplicate entries. Then α: too small learns imperceptibly, too large with big negative rewards oscillates. Finally, I'd evaluate the greedy policy separately from the exploring one, since the curve while exploring can stay flat even after the greedy policy is optimal.

</details>

<details>
<summary><strong>Q(s, a) is −2, reward −1, next-state values are −3, −1.5, −4, −2, and the agent will explore to the −4 action. With α 0.5 and γ 0.9, what are the two updates?</strong></summary>

Q-learning takes the max next value, −1.5. The target is −1 + 0.9 × (−1.5) = −2.35, the TD error is −2.35 − (−2) = −0.35, and the new value is −2 + 0.5 × (−0.35) = −2.175. SARSA uses the value of the action actually taken next, −4. The target is −1 + 0.9 × (−4) = −4.6, the TD error is −2.6, and the new value is −2 + 0.5 × (−2.6) = −3.3. So one exploratory step pulls SARSA's estimate down by 1.3 while Q-learning moves only 0.175. Repeated near a cliff, that difference is what makes SARSA treat the edge as dangerous and route around it, while Q-learning keeps valuing the edge path as if it would never slip.

</details>
