---
title: Deep Q-networks (DQN)
summary: Replace the Q-table with a neural network, and use experience replay, a target network, epsilon decay, and the Huber loss to keep bootstrapped training stable.
skill: deep-learning
minutes: 45
prerequisites: [q-learning-sarsa, neural-network-forward-pass, backpropagation]
related: [policy-gradients-reinforce, ppo, training-and-regularization]
adapted_from: [phases/09-reinforcement-learning/05-dqn]
---

# Deep Q-networks (DQN)

> **Mental model.** Tabular Q-learning keeps one number per (state, action) in a lookup table. DQN swaps the table for a neural network that reads a state and outputs one Q-value per action. That is the easy part. The hard part is that the network is now chasing a target it computes itself, from data that arrives in a highly correlated stream. DQN adds two stabilizers: a shuffled memory of past transitions (experience replay) and a slowly updated copy of the network that computes the targets (the target network).

**You will learn to**
- Write the DQN loss and compute a TD target, TD error, and Huber loss by hand for one transition.
- Explain what experience replay and the target network each fix, and what goes wrong without them.
- Implement a DQN from scratch in NumPy: Q-network, replay buffer, target sync, epsilon decay, gradient step.
- Compute a Double DQN target and explain why it reduces overestimation.
- Choose between DQN and policy-gradient methods such as PPO for a given action space.

**Why it matters.** DQN (Mnih et al., 2013 and 2015) was the first time one network with one set of hyperparameters learned to play dozens of Atari games from raw pixels. Its two tricks, replay and target networks, show up in almost every off-policy deep RL method since (Double DQN, Rainbow, SAC, TD3). And its failure mode, an unstable bootstrapped target, is the reason PPO and GRPO, which you meet next, look the way they do.

## 1. Intuition

In [Q-learning](02-q-learning-sarsa.md) the agent keeps a table $Q(s, a)$ and nudges one entry at a time toward the target "reward plus the discounted best value of the next state". That works on a 4×4 grid with 16 states. It fails when the states are Atari screens, chess boards, or robot joint angles: there are far too many to list, and the table cannot generalize from one state to a similar one it has never seen.

So we learn a function instead. A network with weights $\theta$ takes a state and outputs a vector of Q-values, one per action. We train it like a regression model: the label for $Q(s, a)$ is the same TD target as before. Similar states share weights, so experience in one state improves the estimate in its neighbours.

Two things make this harder than ordinary regression:

1. **The data is correlated.** Consecutive transitions come from the same episode and look almost alike. Training on them in order is like fitting a classifier on a dataset sorted by label. **Experience replay** stores recent transitions in a buffer and trains on random minibatches drawn from it, which breaks the correlation and reuses each transition many times.
2. **The target moves.** The label $r + \gamma \max_{a'} Q(s', a')$ uses the same network you are updating. Each gradient step changes the label for the next step, so the network can chase its own tail and blow up. A **target network** is a frozen copy of the weights, $\theta^-$, used only to compute labels. Every $C$ steps you copy the online weights into it. Between copies the regression target holds still.

The combination of function approximation, bootstrapping (targets built from your own estimates), and off-policy data (learning about the greedy policy while acting epsilon-greedily) is called the **deadly triad**: any two are fine, all three together carry no convergence guarantee. DQN does not remove the triad; replay and the target network make it behave in practice.

> [!NOTE]
> DQN still needs exploration. It acts epsilon-greedily: with probability $\varepsilon$ a random action, otherwise the action with the highest predicted Q. $\varepsilon$ starts near 1 (pure exploration) and decays to a small floor such as 0.05.

## 2. Visualization

![Two panels for a synthetic 4x4 GridWorld, three seeds each. Left: 20-episode mean return rises from about -35 to the optimal -6 by episode 150, with and without a target network. Right: the network's max Q at the start state. With a target network (blue) it dips to about -8 to -10 and settles at the true value -5.85. Without one (orange) it swings down to between -24 and -43 in the first 20 episodes before recovering.](../../figures/dqn.png)

*Synthetic environment: 4×4 GridWorld, start (0,0), goal (3,3), reward −1 per step, $\gamma = 0.99$. Both variants use replay; orange copies the weights into the target every step, which is the same as having no target network. Produced by [`code/26-reinforcement-learning/dqn.py`](../../code/26-reinforcement-learning/dqn.py).*

On this tiny problem both variants learn the optimal 6-step path. The difference is in the right panel. The true value of the start state is $-5.85$. With a target network the estimate drifts smoothly toward it. Without one, the estimate overshoots to $-43$ in one seed, seven times too pessimistic, because every update shifts the label it is being fit to. One-hot states make this net almost a table, so it recovers. With pixels and shared features, the same feedback loop is what made naive deep Q-learning diverge.

**Try it** (predict first, then run the script and check):
1. Change `sync_every=100` to `sync_every=1000`. Predict whether learning gets faster or slower, and why a target that is too stale also hurts.
2. Set `warmup=32` and `capacity=32` so the buffer only holds the latest transitions. Predict what happens to the Q(start) curve when replay can no longer decorrelate data.
3. Set `eps_decay_eps=10` so exploration collapses after 10 episodes. Predict whether the agent still finds the goal on every seed.
4. Replace the Huber gradient with the raw TD error (`g = delta`). Predict which panel changes most early in training, when TD errors are large.

## 3. The math

### Symbols

| Symbol | Meaning | Shape / type |
|---|---|---|
| $s, a, r, s'$ | state, action, reward, next state of one transition | state vector $[16]$ here, action index, scalar, $[16]$ |
| $d$ | 1 if $s'$ is terminal, else 0 | scalar |
| $Q(s, \cdot\,; \theta)$ | online network's Q-values for all actions | $[4]$ (one per action) |
| $\theta$ | online network weights (trained every step) | 676 numbers here |
| $\theta^-$ | target network weights (copied from $\theta$ every $C$ steps) | same as $\theta$ |
| $\gamma$ | discount factor | scalar in $[0, 1)$ |
| $\mathcal{D}$ | replay buffer of recent transitions | up to 2,000 tuples here |
| $y$ | TD target (the regression label) | scalar |
| $\delta$ | TD error $Q(s, a; \theta) - y$ | scalar |
| $\varepsilon$ | exploration probability | scalar in $[0, 1]$ |

The network here is $16 \to 32 \to 4$: $16 \times 32 + 32 + 32 \times 4 + 4 = 512 + 32 + 128 + 4 = 676$ parameters.

### The DQN loss

The target uses the frozen network and does not bootstrap past a terminal state:

```math
y = r + \gamma\,(1 - d)\,\max_{a'} Q(s', a'; \theta^-)
```

The loss averages a penalty on the TD error over a minibatch sampled uniformly from the buffer:

```math
L(\theta) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}}\Big[\,\ell\big(Q(s, a; \theta) - y\big)\Big]
```

Gradients flow only through $Q(s, a; \theta)$, never through $y$. In a framework you enforce this with `no_grad` or `detach` on the target computation.

$\ell$ is usually the **Huber loss** with threshold $\kappa = 1$:

```math
\ell(\delta) = \begin{cases} \tfrac12 \delta^2 & |\delta| \le 1 \\ |\delta| - \tfrac12 & |\delta| > 1 \end{cases}
\qquad
\frac{\partial \ell}{\partial \delta} = \operatorname{clip}(\delta, -1, 1)
```

It is squared error near zero and absolute error far away, so a single surprising transition cannot produce a huge gradient. The 2015 DQN paper got the same effect by clipping the error term to $[-1, 1]$.

**Double DQN** (van Hasselt et al., 2016) changes only the target. The online network picks the next action and the target network scores it:

```math
y^{\text{double}} = r + \gamma\,(1 - d)\, Q\big(s', \arg\max_{a'} Q(s', a'; \theta);\ \theta^-\big)
```

Taking a max over noisy estimates is biased upward: if four actions are all truly worth $-5$ and each estimate has noise, the largest of the four estimates is usually above $-5$. Using two networks to select and evaluate means one network's lucky noise is not also the one that scores it.

### Worked example: one transition

Transition $(0,0) \xrightarrow{\text{right}} (0,1)$, $r = -1$, $d = 0$, $\gamma = 0.99$. Mid-training estimates (hand-picked, close to the true values):

- Target network at $s' = (0,1)$, actions (up, down, left, right): $[-5.6, -4.9, -5.8, -5.0]$
- Online network at $s'$: $[-5.5, -5.1, -5.7, -4.95]$
- Online $Q((0,0), \text{right}) = -5.2$

1. Best target-network value at $s'$: $\max = -4.9$ (down).
2. TD target: $y = -1 + 0.99 \times (-4.9) = -1 - 4.851 = -5.851$.
3. TD error: $\delta = Q(s, a) - y = -5.2 - (-5.851) = 0.651$.
4. $|\delta| = 0.651 \le 1$, so Huber is quadratic: $\ell = 0.5 \times 0.651^2 = 0.5 \times 0.423801 = 0.2119$.
5. Gradient with respect to $Q(s, a)$: $\partial \ell / \partial \delta = 0.651$. Gradient descent lowers $Q((0,0), \text{right})$ from $-5.2$ toward $-5.851$. The other three outputs get zero gradient.
6. Compare a large error, $\delta = 3$: Huber $= 3 - 0.5 = 2.5$ with gradient 1; squared error $\tfrac12 \times 3^2 = 4.5$ with gradient 3. Huber caps the step at a third of the size.
7. Double DQN target: the online network's best action at $s'$ is right ($-4.95$ is the largest of its four values). The target network scores right as $-5.0$. So $y^{\text{double}} = -1 + 0.99 \times (-5.0) = -1 - 4.95 = -5.950$.

The Double DQN target is $0.099$ lower, less optimistic, because it did not take the target network's own maximum. For reference, the true optimal value of $(0,0)$ is $-(1 + 0.99 + 0.99^2 + 0.99^3 + 0.99^4 + 0.99^5) = -5.852$.

## 4. Implementation

The full script is [`code/26-reinforcement-learning/dqn.py`](../../code/26-reinforcement-learning/dqn.py) (NumPy only). It builds DQN in four pieces, prints the worked example above, then trains with and without a target network.

### Build it from scratch

**Step 1: the environment and the state encoding.** A 4×4 grid, start top-left, goal bottom-right, $-1$ per step so shorter paths score higher. Moving into a wall leaves you in place. The network cannot read a tuple, so each state becomes a one-hot vector of length 16.

```python
def step(s, a):
    r, c = s
    dr, dc = DELTAS[a]
    nxt = (min(max(r + dr, 0), GRID - 1), min(max(c + dc, 0), GRID - 1))  # walls: stay put
    return nxt, -1.0, nxt == GOAL

def features(s):  # one-hot state, shape [16]
    x = np.zeros(GRID * GRID)
    x[s[0] * GRID + s[1]] = 1.0
    return x
```

*Predict:* with one-hot inputs, does an update for state (0,0) change the Q-values of state (0,1)? (Look at which weights are shared in step 2.)

**Step 2: the Q-network.** Linear, ReLU, linear. One forward pass returns all four action values at once, which is why DQN suits small discrete action sets: the greedy action is one `argmax`.

```python
def init_net(rng, n_in=16, n_hidden=32, n_out=4):
    return {"W1": rng.normal(0, 0.3, (n_in, n_hidden)), "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, 0.3, (n_hidden, n_out)), "b2": np.zeros(n_out)}

def forward(net, X):  # X: [B, 16] -> Q: [B, 4], hidden: [B, 32]
    H = np.maximum(0.0, X @ net["W1"] + net["b1"])
    return H @ net["W2"] + net["b2"], H
```

*Predict:* how would you change this network for an action space of 10,000 discrete actions? For a continuous steering angle?

**Step 3: the Huber TD loss and one gradient step.** The target comes from `target`, the online network is the only one updated, and the `(1.0 - D)` factor stops bootstrapping at the goal. Backprop is written out by hand: only the column for the action actually taken gets a gradient.

```python
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
```

*Predict:* what changes if you delete `(1.0 - D)`? Which state's value is wrong first, and in which direction? To make this Double DQN, which single line changes?

**Step 4: the outer loop: replay, epsilon decay, target sync.** Act epsilon-greedily, store every transition, and once the buffer has `warmup` transitions train on a uniform random minibatch each step. Every `sync_every` environment steps, copy the online weights into the target.

```python
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
```

The buffer is a `deque(maxlen=capacity)`, so the oldest transitions fall off automatically. *Predict:* the agent stores transitions collected under an old, more random policy and still learns from them. Why is that allowed for Q-learning but not for the REINFORCE algorithm in the next lesson?

### Use the library

In practice you use PyTorch through a library. [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (needs `torch`, `gymnasium`, and `stable-baselines3`; not run here) maps one keyword to each piece you just built:

```python
import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")
model = DQN(
    "MlpPolicy", env,
    learning_rate=1e-3,
    buffer_size=50_000,          # replay capacity
    learning_starts=1_000,       # warmup before the first gradient step
    batch_size=32,
    gamma=0.99,
    target_update_interval=500,  # C: hard copy of theta into theta^- every 500 steps
    exploration_fraction=0.1,    # decay epsilon over the first 10% of training ...
    exploration_final_eps=0.05,  # ... down to 0.05
    train_freq=4,                # one gradient step every 4 environment steps
    seed=0,
)
model.learn(total_timesteps=50_000)
```

What the library adds: autograd instead of hand-written backprop, Adam instead of plain SGD, gradient-norm clipping, vectorized environments, and logging. SB3's `DQN` is the vanilla algorithm with the Huber (smooth L1) loss; Double DQN is a one-line change to the target, and single-file implementations such as CleanRL's are good to read next to this lesson. Frame stacking and reward clipping for Atari come from environment wrappers, not from the algorithm.

### Run it

```bash
uv run code/26-reinforcement-learning/dqn.py
```

Key output (from `guide/`):

```
DQN target   y = -1 + 0.99 * max[-5.6, -4.9, -5.8, -5.0] = -5.851
TD error     delta = Q(s,a) - y = -5.2 - (-5.851) = 0.651
Huber loss   0.5 * delta^2 = 0.2119, gradient = 0.651
if delta = 3: Huber = 2.5 (grad 1), MSE 0.5*delta^2 = 4.5 (grad 3)
Double DQN   online picks right, target scores it -5.0 -> y = -5.950

=== DQN, target net (sync every 100 steps), 300 episodes, seeds [0, 1, 2] ===
mean return, last 50 episodes: -6.32   (optimal -6)
max Q(0,0) at the end per seed: [-5.57, -5.8, -5.84]   (true V*(0,0) = -5.852)
lowest max Q(0,0) during training per seed: [-10.1, -9.0, -8.1]

=== DQN, no target net (sync every step), 300 episodes, seeds [0, 1, 2] ===
mean return, last 50 episodes: -6.37   (optimal -6)
lowest max Q(0,0) during training per seed: [-42.7, -33.3, -24.5]
```

The first five lines reproduce the worked example. The last-50 mean is slightly below $-6$ because $\varepsilon$ stays at 0.05, so about one action in twenty is random.

## 5. Engineering

**When DQN fits.** Discrete, small action sets (game controls, a handful of routing or recommendation choices) with a simulator or plenty of logged transitions. Because it is off-policy and replays old data, it is sample-efficient compared with on-policy methods. For continuous actions use an actor-critic such as SAC or TD3, which keep the replay buffer and target networks but add a policy network. For LLM fine-tuning, the action space is the whole vocabulary at every token, and practice uses policy-gradient methods ([PPO](05-ppo.md), GRPO) rather than Q-learning.

**The improvements that stuck.** Rainbow (Hessel et al., 2018) combined six extensions and reported that they add up:
- *Double DQN:* separate action selection from evaluation in the target (section 3). Cheap, and a sensible default.
- *Dueling architecture:* the network outputs a state value $V(s)$ and per-action advantages $A(s, a)$, combined as $Q = V + A - \text{mean}_a A$. Helps when many actions have similar value.
- *Prioritized experience replay:* sample transitions with large TD error more often, and correct the resulting bias with importance-sampling weights.
- *n-step returns, distributional Q (C51, QR-DQN), and noisy networks* for learned exploration.

**Hyperparameters that matter.** The 2015 Atari DQN used a 1M-transition buffer, minibatch 32, target sync every 10,000 steps, $\varepsilon$ decayed from 1.0 to 0.1 over the first million frames, rewards clipped to $\{-1, 0, +1\}$, and 4 stacked frames as the state so velocity is visible. Treat these as a starting point for Atari-like problems, not universal values.

**Compute and memory.** The buffer often dominates memory. One 84×84 grayscale frame is 7,056 bytes. Storing a 4-frame state and next state per transition would be $2 \times 4 \times 7{,}056 = 56{,}448$ bytes, about 56 GB for a million transitions; storing each frame once and indexing into it brings that to about 7 GB.

> [!WARNING]
> **Failure modes.** Divergence or oscillating Q-values (check the target sync and learning rate); overestimation, with Q-values far above any achievable return (use Double DQN); policy collapse after epsilon hits its floor too early; a non-Markov state (single frames with no velocity); and reward scales so large that gradients explode.

### Common mistakes

- Letting gradients flow through the target $y$. It must be computed with the target network and treated as a constant.
- Bootstrapping past a terminal state (forgetting $(1 - d)$), or treating a time-limit cutoff as terminal. A truncated episode is not a real end, so you should still bootstrap there.
- Training on the first few transitions before the buffer has a reasonable warmup.
- Syncing the target every step (no stabilization) or so rarely that targets are stale.
- Decaying $\varepsilon$ to zero, or decaying it before the agent has ever reached the reward.
- Judging progress by the training loss. A falling TD loss can mean the network agrees with its own wrong targets; watch episode return on evaluation runs with $\varepsilon \approx 0$.

## 6. Knowledge check

<!-- quiz:dqn -->
**[Take the DQN quiz](../../quizzes/dqn.md)**
<!-- /quiz -->

**Practice exercise.** A transition goes from $s$ to a terminal goal state with $r = +10$, $\gamma = 0.9$. The target network's values at $s'$ are $[3.0, 4.0]$. The online network's $Q(s, a) = 7.5$. Compute the TD target, the TD error, the Huber loss, and its gradient. Then do the same for a non-terminal version of the transition.

<details>
<summary>Solution</summary>

Terminal: $d = 1$, so $y = 10 + 0.9 \times 0 \times 4.0 = 10$. $\delta = 7.5 - 10 = -2.5$. $|\delta| > 1$, so Huber $= 2.5 - 0.5 = 2.0$ and the gradient is $\operatorname{clip}(-2.5, -1, 1) = -1$: the update raises $Q(s,a)$.

Non-terminal: $y = 10 + 0.9 \times 4.0 = 10 + 3.6 = 13.6$. $\delta = 7.5 - 13.6 = -6.1$. Huber $= 6.1 - 0.5 = 5.6$, gradient $-1$. Squared error would give $\tfrac12 \times 6.1^2 = 18.605$ with gradient $-6.1$, more than six times larger.

</details>

**Implementation challenge.** Add Double DQN to [`dqn.py`](../../code/26-reinforcement-learning/dqn.py): in `train_step`, choose the next action with the online network and evaluate it with the target network. Then add Gaussian noise ($\sigma = 2$) to every reward and compare the final $\max_a Q((0,0), a)$ of DQN and Double DQN against the true value $-5.852$ over 5 seeds.

<details>
<summary>Solution sketch</summary>

Replace the target line with:

```python
q_next_online, _ = forward(online, S2)
a_star = q_next_online.argmax(axis=1)
y = R + gamma * (1.0 - D) * q_next[np.arange(len(A)), a_star]
```

With noisy rewards the vanilla `max` picks whichever action got lucky noise, so its Q(start) tends to sit above $-5.852$ (optimistic). Double DQN's estimate should be closer, often slightly below. Report the mean and spread over seeds rather than a single run, since five seeds of a toy problem is a small sample.

</details>

## Summary

- DQN is Q-learning with a neural network: $Q(s, \cdot\,; \theta)$ outputs one value per action, trained by regression toward the TD target.
- Experience replay breaks the correlation between consecutive transitions and reuses data; it works because Q-learning is off-policy.
- The target network $\theta^-$ freezes the regression label between syncs; without it the label moves with every step and estimates can swing wildly or diverge.
- Huber loss caps the gradient at 1, so rare large TD errors cannot dominate an update. Epsilon decays from 1 to a small floor.
- Double DQN uses the online net to select and the target net to evaluate, reducing max-operator overestimation; dueling heads and prioritized replay are other standard upgrades.
- DQN suits small discrete action sets; continuous control uses SAC or TD3, and LLM fine-tuning uses policy gradients.

**Next:** [Policy gradients and REINFORCE](04-policy-gradients-reinforce.md)

**Related:** [Q-learning and SARSA](02-q-learning-sarsa.md) · [PPO](05-ppo.md) · [Backpropagation](../11-gradient-descent-backprop/02-backpropagation.md) · [Training and regularization](../12-training-regularization/01-training-and-regularization.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Deep Q-Networks (DQN)](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/05-dqn).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>What problems do experience replay and the target network each solve in DQN?</strong></summary>

They fix two different instabilities. Experience replay fixes the data: transitions arrive as a stream where consecutive samples are nearly identical and the distribution shifts as the policy changes. SGD assumes roughly independent samples, so DQN stores recent transitions in a buffer and trains on uniform random minibatches. That decorrelates the batch and lets each transition be reused many times, which improves sample efficiency. It is only valid because Q-learning is off-policy: the target uses a max, not the behaviour policy's action. The target network fixes the label. The TD target uses Q of the next state, so with one network every gradient step moves the target too, and the regression can chase itself into oscillation or divergence. A frozen copy, synced every C steps or softly averaged, holds the target still long enough to fit it.

</details>

<details>
<summary><strong>When would you pick DQN over PPO, and when the other way round?</strong></summary>

DQN suits small discrete action spaces where samples are expensive. It is off-policy, so it can learn from a replay buffer of old transitions and even from logged data, and it often needs fewer environment steps. Its weaknesses: the greedy action needs a max over actions, which breaks down for continuous or huge action spaces; bootstrapped targets plus function approximation can be unstable; and it tends to need more tuning. PPO is on-policy, so it throws data away after a few epochs and needs more samples, but it is stable, handles continuous actions and very large discrete spaces (it only samples from a policy distribution), and scales well with many parallel simulators. Cheap simulator or continuous control: PPO or SAC. Atari-like discrete control with a costly simulator: DQN variants such as Rainbow. LLM fine-tuning: policy gradients, since the action space is the vocabulary.

</details>

<details>
<summary><strong>Your DQN's predicted Q-values keep climbing far above any return the agent can actually achieve. What is happening and what do you check?</strong></summary>

That is overestimation, and the usual source is the max in the target. With noisy estimates, the max over actions picks whichever action got positive noise, so the target is biased upward; bootstrapping then feeds that inflated value into earlier states and the error compounds. First I check the setup: is the target network actually used, and is its sync interval reasonable? Are terminal states masked, so the agent is not bootstrapping past the end? Is the learning rate too high, are rewards scaled sensibly, and is the loss Huber? Then I switch to Double DQN, where the online network selects the next action and the target network evaluates it, which removes most of the bias. I monitor mean predicted Q against actual discounted returns from evaluation episodes; they should track each other.

</details>

<details>
<summary><strong>Size the replay buffer for an Atari DQN with 1 million transitions of 4 stacked 84×84 grayscale frames.</strong></summary>

One frame is 84 × 84 = 7,056 bytes as uint8. A state is 4 frames, 28,224 bytes, and a transition needs both the state and the next state, so a naive buffer stores 56,448 bytes per transition. Times one million is about 56 GB, which is too much for most single machines. But consecutive states share 3 of their 4 frames, and the next state shares 3 frames with the current one. So real implementations store each frame once in a circular array, plus the action, reward, and done flag, and rebuild stacks by index at sampling time. That is about 7,056 bytes per transition, roughly 7 GB for a million. Further savings come from compression, a smaller buffer, or storing observations on disk. Always use uint8, never float32, which would multiply everything by four.

</details>
