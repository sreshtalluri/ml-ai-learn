---
title: MDPs, states, actions, and rewards
summary: Frame a sequential decision problem as a Markov decision process, compute discounted returns and state values with the Bellman equation, and spot broken formulations before training.
skill: deep-learning
minutes: 40
prerequisites: [learning-paradigms, probability-and-statistics]
related: [q-learning-sarsa, policy-gradients-reinforce, fine-tuning-in-practice]
adapted_from: [phases/09-reinforcement-learning/01-mdps-states-actions-rewards]
---

# MDPs, states, actions, and rewards

> **Mental model.** Reinforcement learning is a game against a world you can't see the rules of. At each turn you look at the board (the state), make a move (the action), and the world hands you a score (the reward) and a new board. A Markov decision process writes those rules down. Every RL algorithm, from tabular Q-learning to the PPO loop that tunes a chat model, is a different way of finding good moves in that same game.

**You will learn to**
- Name the five parts of an MDP (states, actions, transitions, rewards, discount) for a real problem.
- Compute a discounted return and explain the effective horizon $1/(1-\gamma)$.
- Define a policy, a state value $V^\pi$, and an action value $Q^\pi$, and check a value with the Bellman equation by hand.
- Run policy evaluation and value iteration on a small grid.
- Diagnose common formulation bugs: a non-Markov state, sparse or hackable rewards, an unbounded return.

**Why it matters.** Supervised learning gives you the right answer for every input. Reinforcement learning gives you only a score, often late, for a whole sequence of choices. Before any algorithm can learn from that, the problem needs a precise shape. Most "RL doesn't work" bugs are formulation bugs: the state is missing information, the reward rewards the wrong thing, or the return is infinite. The same shape underlies [RLHF and GRPO-style fine-tuning](../15-llms/04-fine-tuning-in-practice.md), where the state is the text so far and each action is the next token.

## 1. Intuition

Picture a robot on a 4×4 floor. It starts in the top-left corner and wants to reach the charger in the bottom-right corner. Each move drains a little battery, so every step costs $-1$. Bumping into a wall wastes the move. The episode ends at the charger.

That is a complete RL problem. Five things define it:

| Piece | In the grid | In an LLM fine-tuning loop |
|---|---|---|
| **State** $s$ | which cell the robot is in | the prompt plus the tokens generated so far |
| **Action** $a$ | up, down, left, right | the next token |
| **Transition** $P(s' \mid s, a)$ | the cell you land in | append the token (deterministic) |
| **Reward** $r$ | $-1$ per move | usually 0 until the end, then a reward-model score or a verifier's pass/fail |
| **Discount** $\gamma$ | how much later rewards count | often 1, since responses are finite |

The agent's behavior is a **policy** $\pi(a \mid s)$: for each state, a probability for each action. The goal is a policy that collects as much total reward as possible, not just the best next reward. That difference is the heart of RL: a move can look bad now (a detour) and be right overall.

**The Markov property.** The next state depends only on the current state and action, not on how you got there. In the grid that holds: the cell is all you need. When it fails, the fix is in the state, not the algorithm. A single frame of a video game hides which way the ball is moving, so DQN stacked the last four frames into one state. A chat model's state is the full context window for the same reason.

**Discounting.** A reward $k$ steps in the future is multiplied by $\gamma^k$. With $\gamma = 0.9$ a reward 10 steps away keeps about a third of its weight; 50 steps away it is nearly gone. A rough rule: $\gamma$ buys a horizon of about $1/(1-\gamma)$ steps.

> [!IMPORTANT]
> **Reward vs return vs value.** A *reward* is the single number from one step. The *return* is the discounted sum of rewards from now to the end of one episode. A *value* is the *expected* return, averaged over everything the policy and world might do. Algorithms learn values; you observe rewards.

## 2. Visualization

![Three panels. Left: discount weight gamma to the k against steps into the future, for gamma 0.5, 0.9, 0.99; the weight falls to near zero after about 5, 30, and 300 steps. Middle: values of the uniform random policy on the 4x4 grid with gamma 0.9, from -9.4 at the start corner to -5.8 next to the goal. Right: optimal values with gamma 1, equal to minus the number of moves to the goal, from -6 at the start to -1 beside the goal.](../../figures/mdps-states-actions-rewards.png)

*Synthetic 4×4 grid, every move costs −1. Left: how fast each discount forgets the future. Middle: a random walker expects to pay a lot before stumbling onto the goal. Right: the best policy pays exactly the number of moves it needs.*

**Try it** (predict first, then run [`code/26-reinforcement-learning/mdp.py`](../../code/26-reinforcement-learning/mdp.py)):
1. In the middle panel, why are the two cells next to the goal equal (−5.8), and why is the start corner worst? Predict what happens to the start corner if you change $\gamma$ to 0.99.
2. The random policy's mean undiscounted return over 5,000 rollouts is about −60 even though the optimum is −6. Predict whether the standard deviation is small or large, then check the script output.
3. In the right panel, values go up by exactly 1 per step toward the goal. What would change if one interior cell were a wall?
4. Change the step reward to −2. Predict the new $V^*$ at the start, and whether the best policy changes.

## 3. The math

### Symbols

| Symbol | Meaning | Shape / type |
|---|---|---|
| $\mathcal{S}, \mathcal{A}$ | set of states, set of actions | finite sets here (16 cells, 4 moves) |
| $s_t, a_t, r_t$ | state, action, reward at step $t$ | one cell, one move, scalar |
| $P(s' \mid s, a)$ | transition probability | $\lvert\mathcal{S}\rvert \times \lvert\mathcal{A}\rvert \times \lvert\mathcal{S}\rvert$ table |
| $\gamma$ | discount factor | scalar in $[0, 1]$ |
| $\pi(a \mid s)$ | policy: probability of action $a$ in state $s$ | $\lvert\mathcal{S}\rvert \times \lvert\mathcal{A}\rvert$, rows sum to 1 |
| $G_t$ | return from step $t$ | scalar |
| $V^\pi(s)$ | state value: expected return from $s$ under $\pi$ | vector, length $\lvert\mathcal{S}\rvert$ |
| $Q^\pi(s, a)$ | action value: expected return taking $a$ in $s$, then following $\pi$ | $\lvert\mathcal{S}\rvert \times \lvert\mathcal{A}\rvert$ |
| $V^*, Q^*$ | optimal values (best policy) | same shapes |

### Return and values

```math
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma G_{t+1}
```

```math
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s], \qquad Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]
```

If every reward is bounded by $R_{\max}$ and $\gamma < 1$, then $\lvert G_t\rvert \le R_{\max}/(1-\gamma)$, which is where the "horizon" $1/(1-\gamma)$ comes from.

### The Bellman equations

Split the return into "this step" plus "everything after", then take expectations:

```math
V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\,\big[r(s, a, s') + \gamma V^\pi(s')\big]
```

```math
V^*(s) = \max_{a} \sum_{s'} P(s' \mid s, a)\,\big[r(s, a, s') + \gamma V^*(s')\big]
```

The first (expectation) equation averages over the policy's actions; it defines the value of a fixed policy. The second (optimality) equation takes the best action; its solution is the value of the best policy. **Policy evaluation** repeats the first as an update until nothing changes. **Value iteration** repeats the second. Both converge for $\gamma < 1$ (and here for $\gamma = 1$ too, because every policy that matters reaches the terminal state).

### Worked example 1: a discounted return

Six moves to the goal, each reward $-1$, $\gamma = 0.9$:

1. $G = -1 - 0.9 - 0.9^2 - 0.9^3 - 0.9^4 - 0.9^5$
2. Powers: $0.9^2 = 0.81$, $0.9^3 = 0.729$, $0.9^4 = 0.6561$, $0.9^5 = 0.59049$.
3. Sum: $1 + 0.9 + 0.81 + 0.729 + 0.6561 + 0.59049 = 4.68559$.
4. $G = -4.6856$. Check with the geometric-series formula: $-\frac{1 - 0.9^6}{1 - 0.9} = -\frac{1 - 0.531441}{0.1} = -4.68559$. ✓

With $\gamma = 1$ the same path returns $-6$; with $\gamma = 0.5$ it returns $-1.9688$, so a short horizon barely tells a 6-step path from a 60-step one.

### Worked example 2: checking a value with the Bellman equation

Uniform random policy ($\pi = 0.25$ for each action), $\gamma = 0.9$. Take the cell $(3, 2)$, just left of the goal. The script's converged values for the cells each action reaches:

| action | next cell | $V^\pi(s')$ | $r + \gamma V^\pi(s')$ |
|---|---|---|---|
| up | (2, 2) | −7.416 | $-1 + 0.9 \times (-7.416) = -7.674$ |
| down | (3, 2), wall bump | −5.757 | $-1 + 0.9 \times (-5.757) = -6.181$ |
| left | (3, 1) | −7.970 | $-1 + 0.9 \times (-7.970) = -8.173$ |
| right | (3, 3), goal | 0 | $-1 + 0.9 \times 0 = -1.000$ |

1. Sum of the last column: $-7.674 - 6.181 - 8.173 - 1.000 = -23.028$.
2. Multiply by $\pi = 0.25$: $-23.028 \times 0.25 = -5.757$.
3. That equals $V^\pi(3, 2) = -5.757$ from the table. ✓ The value is a fixed point: plug it in on the right and it comes back out on the left.

For the optimal values with $\gamma = 1$: $V^*(3,2) = \max(-1 + V^*(2,2), \dots, -1 + V^*(\text{goal})) = \max(-1 - 2, \dots, -1 + 0) = -1$, and by the same rule each step away from the goal subtracts 1, giving $V^*(0,0) = -6$.

## 4. Implementation

The full script is [`code/26-reinforcement-learning/mdp.py`](../../code/26-reinforcement-learning/mdp.py) (NumPy only). It builds the grid, rolls out a random policy, and solves for values two ways.

### Build it from scratch

**Step 1: the world.** The environment is one function: given a state and action, return the next state, the reward, and whether the episode ended. Writing transitions and rewards in code, not a table, is how most simulators work. Clamping the coordinates is the "bump into the wall" rule.

```python
N = 4
STATES = [(r, c) for r in range(N) for c in range(N)]
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
START, GOAL = (0, 0), (3, 3)

def step(s, a):
    if s == GOAL:
        return s, 0.0, True                     # absorbing terminal state
    dr, dc = ACTIONS[a]
    s2 = (min(max(s[0] + dr, 0), N - 1), min(max(s[1] + dc, 0), N - 1))
    return s2, -1.0, s2 == GOAL
```

*Predict:* what does `step((0, 0), "up")` return?

**Step 2: the return.** Working backward turns the infinite sum into one multiply-add per step, using $G_t = r_t + \gamma G_{t+1}$. Every RL codebase has this loop somewhere (it reappears as "returns-to-go" in policy gradients).

```python
def discounted_return(rewards, gamma):
    g = 0.0
    for r in reversed(rewards):   # G_t = r_t + gamma * G_{t+1}
        g = r + gamma * g
    return g
```

*Predict:* `discounted_return([0, 0, 0, 1], 0.5)`?

**Step 3: a policy and a rollout.** A policy maps a state to a distribution over actions. A rollout samples from it until the episode ends. Averaging many rollouts estimates the value by brute force (this is the Monte Carlo method).

```python
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
```

*Predict:* why does `rollout` need `max_steps` even though this grid always reaches the goal eventually?

**Step 4: policy evaluation.** Apply the Bellman expectation equation to every state, over and over, until the largest change is tiny. Updating `V` in place (using new values as soon as they exist) converges a bit faster than keeping two copies.

```python
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
```

*Predict:* with `gamma=1.0` and the uniform policy, this still converges. Would it converge on a grid with no reachable goal?

**Step 5: value iteration.** Replace the average over the policy with a max over actions. The result is $V^*$, and acting greedily on it is optimal.

```python
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
```

*Predict:* how many sweeps until $V^*(0,0)$ reaches −6, if each sweep goes in row-major order from (0,0)?

### Use the library

[Gymnasium](https://gymnasium.farama.org/) is the standard environment interface. Its `FrozenLake-v1` is a 4×4 grid MDP like ours, but slippery by default (stochastic transitions), and it exposes the transition table as `env.unwrapped.P[s][a]`, a list of `(prob, next_state, reward, done)` tuples.

```python
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
P = env.unwrapped.P          # P[s][a] -> [(prob, s2, r, done), ...]; plug into the Bellman update
```

The library adds what our script left implicit: separate `terminated` (the MDP ended) and `truncated` (a time limit cut it off) flags, seeded resets, and a standard `step` signature every RL library understands. The difference between the two flags matters: on truncation you should still bootstrap from the next state's value; on termination you should not.

### Run it

```bash
uv run code/26-reinforcement-learning/mdp.py
```

```
gamma = 0.9: G = -4.6856
mean return = -59.87, std = 51.37  (optimal = -6)
Bellman check at (3, 2), gamma=0.9: ... 0.25 * sum(-1 + 0.9 V) = -5.757 = V(3, 2) = -5.757
=== V* by value iteration (gamma = 1): minus the number of moves to the goal ===
[[-6. -5. -4. -3.]
 ...
saved figures/mdps-states-actions-rewards.png
```

## 5. Engineering

**Write the MDP down before writing code.** For each problem, fill in five lines: state, action, transition (known or only sampled?), reward (when, and of what), and $\gamma$ or horizon.

| Problem | State | Action | Reward | Discount |
|---|---|---|---|---|
| Robot locomotion | joint angles and velocities | continuous torques | forward progress minus energy | 0.99 |
| Board game | board (plus history if rules need it) | a legal move | +1 win, −1 loss, at the end | 1 (finite game) |
| Inventory | stock level, recent demand | order quantity | revenue minus holding and stockout cost | 0.95 to 0.99 |
| LLM RLHF | prompt plus tokens so far | next token | reward-model score at the end, minus a KL penalty | 1 |
| LLM RL with verifiable rewards | prompt plus tokens so far | next token | 1 if the answer checks out, else 0 | 1 |

**Known model vs samples only.** If you can write $P$ and $r$ as tables (small planning problems, games with known rules), dynamic programming like value iteration solves the MDP exactly. Most real problems only let you sample transitions, which is why the next lessons learn from experience.

**Reward design is product design.** The agent optimizes exactly the number you give it. A boat-racing agent from OpenAI (2016) learned to circle a lagoon collecting point pickups forever instead of finishing the race, because the score rewarded pickups. LLMs trained against a reward model learn to please the reward model (longer answers, confident tone) unless something checks them. Define reward from the outcome you want, and keep a held-out check the agent can't see.

**Sparse rewards.** If reward only comes at the very end (win or lose), a random policy may never see a nonzero signal. Options: shape the reward with intermediate signals (carefully, since shaping can be hacked), start from a supervised or imitation-learned policy (this is why LLM RL starts from an SFT model), or use a curriculum of easier tasks.

> [!WARNING]
> **Failure modes.** A state that hides needed information (the agent can't tell two situations apart and dithers); a reward that measures a proxy; $\gamma = 1$ on a task that never ends (values become infinite); rewards on wildly different scales (+1000 for one event, ±1 for others), which make value and gradient estimates noisy.

### Common mistakes

- Confusing reward with return or value, and comparing a single episode's return to a value estimate as if they should match.
- Using $\gamma = 1$ with no time limit or terminal state.
- Treating a time-limit cutoff as a real terminal state, which teaches the agent that the world ends at step 200.
- Leaving information out of the state (velocity, the last few observations, remaining budget) and blaming the algorithm.
- Changing the reward and comparing returns across runs as if the numbers meant the same thing.

## 6. Knowledge check

<!-- quiz:mdps-states-actions-rewards -->
**[Take the MDPs quiz](../../quizzes/mdps-states-actions-rewards.md)**
<!-- /quiz -->

**Practice exercise.** An episode gives rewards $[0, 0, -1, 10]$ and $\gamma = 0.5$. Compute $G_0$, then $G_2$.

<details>
<summary>Solution</summary>

Work backward. $G_3 = 10$. $G_2 = -1 + 0.5 \times 10 = 4$. $G_1 = 0 + 0.5 \times 4 = 2$. $G_0 = 0 + 0.5 \times 2 = 1$. Directly: $0 + 0.5 \times 0 + 0.25 \times (-1) + 0.125 \times 10 = -0.25 + 1.25 = 1$. ✓

</details>

**Implementation challenge.** Make the grid slippery: an action moves in the intended direction with probability 0.8 and in each of the two perpendicular directions with probability 0.1. Change `step` to return a list of `(prob, next_state, reward, done)` and update `value_iteration` to sum over it. Does $V^*(0,0)$ get better or worse than −6, and does the best path change?

<details>
<summary>Solution sketch</summary>

In the Bellman update replace `r + gamma * V[s2]` with `sum(p * (r + gamma * V[s2]) for p, s2, r, _ in outcomes(s, a))`. $V^*(0,0)$ drops below −6 (about −7.4 with $\gamma = 1$), because slips waste moves. On this open grid the best actions stay "down or right", since a sideways slip along a wall just bumps in place; with pits nearby the best path would move away from them.

</details>

## Summary

- An MDP is states, actions, transitions, rewards, and a discount. The Markov property says the state carries everything needed to predict what happens next.
- The return is the discounted sum of future rewards; $\gamma$ sets an effective horizon of about $1/(1-\gamma)$ steps.
- Values are expected returns: $V^\pi(s)$ for a state, $Q^\pi(s,a)$ for a state and first action.
- The Bellman equation splits a value into one step of reward plus the discounted value of where you land. Iterating it gives policy evaluation (average) or value iteration (max).
- Most RL failures start in the formulation: missing state, a hackable or sparse reward, or an unbounded return.

**Next:** [Q-learning and SARSA](02-q-learning-sarsa.md)

**Related:** [Learning paradigms](../01-ml-vocabulary/02-learning-paradigms.md) · [Policy gradients and REINFORCE](04-policy-gradients-reinforce.md) · [Fine-tuning in practice](../15-llms/04-fine-tuning-in-practice.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Dynamic Programming — Policy Iteration & Value Iteration](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/02-dynamic-programming)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [MDPs, States, Actions & Rewards](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/01-mdps-states-actions-rewards).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Formalize a problem as an MDP. What are the pieces, and what does the Markov property buy you?</strong></summary>

An MDP is a tuple of states, actions, a transition distribution P(s′ | s, a), a reward function, and a discount γ. The agent picks actions with a policy π(a | s), and the goal is the policy that maximizes expected discounted return. The Markov property says the next state and reward depend only on the current state and action, not the history. That is what makes the Bellman recursion valid: the value of a state can be written as one step of reward plus the discounted value of the next state, with no need to remember the path. If the property fails, say a single game frame hides velocity, I fix the state, for example by stacking recent frames or using a recurrent network, rather than changing the algorithm. For an LLM the state is the whole context, which is Markov by construction.

</details>

<details>
<summary><strong>How do you choose the discount factor, and what goes wrong at the extremes?</strong></summary>

γ trades off near and far rewards, and its practical meaning is a horizon of about 1/(1 − γ) steps: 0.9 looks about 10 steps ahead, 0.99 about 100. Too small and the agent is myopic: it can't tell a 6-step path from a 60-step one, since both look nearly the same after heavy discounting. Too large and credit assignment gets noisy, because many early actions share responsibility for a distant reward, so value estimates have high variance and learn slowly. γ = 1 is only safe when episodes are guaranteed to end, otherwise returns can be infinite. In practice control tasks use 0.99, short finite tasks like LLM responses often use 1, and I tune γ alongside the episode length rather than treating it as fixed.

</details>

<details>
<summary><strong>Your RL agent finds a way to score high reward while doing something useless. How do you diagnose and fix it?</strong></summary>

That is reward hacking: the agent optimizes the reward I wrote, not the outcome I wanted. I first watch episodes, not just the curve, and log reward components separately to see which one it is exploiting. Then I check the reward against the true goal on held-out evaluations that the agent never trains on, ideally human review or a different metric. Fixes, in order: redefine the reward from the outcome rather than a proxy, remove or cap the exploited shaping term, add constraints or penalties for the bad behavior, and for LLMs keep a KL penalty to the reference model so the policy can't drift far to exploit reward-model blind spots. I also retrain or ensemble the reward model on the new failure cases, since a fixed learned reward will be exploited again.

</details>

<details>
<summary><strong>Every move costs −1 and the goal is 6 moves away. What are the optimal value at the start with γ = 1 and with γ = 0.9?</strong></summary>

With γ = 1 the return is just the sum of six −1 rewards, so V* at the start is −6. With γ = 0.9 the return is −(1 + 0.9 + 0.81 + 0.729 + 0.6561 + 0.59049), which sums to −4.6856; the closed form is −(1 − 0.9⁶)/(1 − 0.9) = −(1 − 0.531441)/0.1. The best policy doesn't change with γ here, since a shorter path is better under any discount when every step costs the same. A follow-up I'd mention: if instead the only reward were +1 at the goal and 0 elsewhere, γ = 1 would make every path that eventually reaches the goal equally good, so you need γ < 1 or a step cost to prefer the short path.

</details>
