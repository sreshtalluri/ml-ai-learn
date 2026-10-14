<!-- GENERATED from dqn.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Deep Q-networks (DQN)

Covers the lesson [Deep Q-networks (DQN)](../lessons/26-reinforcement-learning/03-dqn.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/dqn/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

A non-terminal transition has $r = -1$ and $\gamma = 0.99$. The target network's Q-values at $s'$ are $[-5.6, -4.9, -5.8, -5.0]$. What is the DQN target $y$?

<details>
<summary>Answer</summary>

**-5.851** (within ±0.001)

$y = r + \gamma \max_{a'} Q(s', a'; \theta^-) = -1 + 0.99 \times (-4.9) = -1 - 4.851 = -5.851$.

</details>

## 2. Calculation (medium)

The TD error for one transition is $\delta = -2.5$. What is the Huber loss with threshold 1?

<details>
<summary>Answer</summary>

**2** (within ±0.001)

$|\delta| = 2.5 > 1$, so the loss is linear: $|\delta| - 0.5 = 2.5 - 0.5 = 2.0$. Its gradient is clipped to $-1$, while squared error would give $\tfrac12 \times 6.25 = 3.125$ with gradient $-2.5$.

</details>

## 3. Calculation (hard)

Same transition: $r = -1$, $\gamma = 0.99$. The online network's Q-values at $s'$ are $[-5.5, -5.1, -5.7, -4.95]$; the target network's are $[-5.6, -4.9, -5.8, -5.0]$. What is the Double DQN target?

<details>
<summary>Answer</summary>

**-5.95** (within ±0.001)

The online network selects the action: the largest of its values is $-4.95$ (index 3). The target network evaluates that action: $-5.0$. So $y = -1 + 0.99 \times (-5.0) = -5.95$, lower than the vanilla target $-5.851$.

</details>

## 4. Multiple choice (medium)

Why can DQN train on transitions collected many episodes ago, under a more random policy?

- **A.** Because old transitions are only used during the warmup phase.
- **B.** Because Q-learning is off-policy; its target uses a max over next actions, not the action the old policy took.
- **C.** Because the replay buffer reweights old transitions by how likely the current policy is to take them.
- **D.** Because the target network is a copy of the old policy that collected the data.

<details>
<summary>Answer</summary>

**B.** Because Q-learning is off-policy; its target uses a max over next actions, not the action the old policy took.

The Q-learning target $r + \gamma \max_{a'} Q(s', a')$ evaluates the greedy policy no matter which policy generated $(s, a, r, s')$, so any past transition is valid training data.

- **A:** After warmup, minibatches keep sampling the whole buffer, old and new transitions alike.
- **C:** Uniform replay does no reweighting; prioritized replay uses importance weights for its own sampling bias, not to correct for the behaviour policy.
- **D:** The target network is a recent copy of the Q-network, synced every C steps, and has nothing to do with which policy collected the data.

</details>

## 5. Multiple choice (hard)

A teammate's DQN uses the same network to compute both $Q(s, a)$ and the target $r + \gamma \max_{a'} Q(s', a')$. Q-values oscillate and sometimes blow up. What is the most direct fix?

- **A.** Switch the loss from Huber to mean squared error.
- **B.** Remove the replay buffer and train on each transition as it arrives.
- **C.** Add a target network, a copy of the weights synced every C steps, and compute targets from it.
- **D.** Increase epsilon so the agent explores more.

<details>
<summary>Answer</summary>

**C.** Add a target network, a copy of the weights synced every C steps, and compute targets from it.

With one network, every gradient step moves the regression label too, so the fit chases itself. Freezing the target weights between syncs holds the label still.

- **A:** Squared error gives larger gradients for large TD errors, which makes instability worse, not better.
- **B:** Training on the stream in order adds correlated updates on top of the moving target, the other cause of instability DQN was built to remove.
- **D:** More exploration changes which data is collected but leaves the moving-target feedback loop in place.

</details>

## 6. Multiple choice (medium)

In a task with reward -1 per step, a DQN's Q-values for states next to the goal keep drifting more negative than the true value of -1, and the error spreads to earlier states. Which bug is most likely?

- **A.** The replay buffer is too large.
- **B.** Epsilon decays too slowly.
- **C.** The target network is synced too often.
- **D.** The target bootstraps past terminal states (the $(1 - d)$ mask is missing).

<details>
<summary>Answer</summary>

**D.** The target bootstraps past terminal states (the $(1 - d)$ mask is missing).

At the goal the target should be just $r = -1$. Without the mask it becomes $-1 + \gamma \max Q(s_{\text{goal}}, \cdot)$, adding a meaningless value for the terminal state, and that error propagates backward through bootstrapping.

- **A:** A large buffer means older data is replayed, which slows adaptation but does not bias values near the goal in a consistent direction.
- **B:** Slow epsilon decay means more random actions, which lowers episode returns but does not change what the target is for a goal transition.
- **C:** Frequent syncing can cause oscillation, but it does not create a consistent bias in states that end the episode.

</details>

## 7. Multiple choice (medium)

You need to control a robot arm whose action is 7 continuous joint torques. Which choice fits best?

- **A.** Tabular Q-learning over the joint angles.
- **B.** An actor-critic such as SAC or TD3, which keeps replay and target networks but learns a policy network that outputs continuous actions.
- **C.** DQN with one output per action, after discretizing each torque into 100 levels.
- **D.** DQN with a single output, the torque value.

<details>
<summary>Answer</summary>

**B.** An actor-critic such as SAC or TD3, which keeps replay and target networks but learns a policy network that outputs continuous actions.

DQN needs a max over actions, which is only cheap for a small discrete set. Actor-critic methods replace the max with a learned policy and handle continuous actions directly.

- **A:** Continuous joint angles give an infinite state space, so there is no table to fill.
- **C:** 100 levels for each of 7 joints is $100^7 = 10^{14}$ joint actions, far too many outputs to enumerate.
- **D:** A Q-network scores actions; it does not output one. Picking the action still requires a max over all possible torques.

</details>

## 8. Multiple choice (easy)

What does the Huber loss do that squared error does not?

- **A.** It makes the Q-network's outputs bounded between -1 and 1.
- **B.** It corrects the upward bias of the max in the target.
- **C.** It caps the gradient magnitude at the threshold, so a rare large TD error cannot produce a huge update.
- **D.** It removes the need for a target network.

<details>
<summary>Answer</summary>

**C.** It caps the gradient magnitude at the threshold, so a rare large TD error cannot produce a huge update.

Huber is quadratic for $|\delta| \le 1$ and linear beyond, so $\partial \ell / \partial \delta = \operatorname{clip}(\delta, -1, 1)$.

- **A:** It clips the gradient of the loss, not the outputs. Q-values can be any size.
- **B:** Overestimation comes from the max operator; Double DQN addresses it.
- **D:** Huber limits step size but the label still moves with the network; the target network is a separate fix.

</details>

## 9. Calculation (easy)

A Q-network takes a one-hot state of length 16, has one hidden layer of 32 units, and outputs 4 action values. Counting weights and biases, how many parameters does it have?

<details>
<summary>Answer</summary>

**676**

$16 \times 32 + 32 = 544$ for the hidden layer and $32 \times 4 + 4 = 132$ for the output: $544 + 132 = 676$.

</details>

## 10. Multiple choice (hard)

Your DQN's TD loss falls steadily, but evaluation returns are flat. What is the best interpretation?

- **A.** Training has converged to the optimal policy, and the evaluation episodes are too short to show it.
- **B.** The learning rate is too low; raise it until the returns improve.
- **C.** The replay buffer is working correctly, so the problem must be in the environment.
- **D.** The loss only measures agreement with the network's own bootstrapped targets, so it can fall while values stay wrong; trust evaluation return.

<details>
<summary>Answer</summary>

**D.** The loss only measures agreement with the network's own bootstrapped targets, so it can fall while values stay wrong; trust evaluation return.

TD loss is not a metric of task success. The network can become self-consistent with wrong targets (for example, an agent that never reached the reward). Watch greedy evaluation returns and compare predicted Q with actual returns.

- **A:** A falling loss says nothing about optimality, and flat returns are evidence against it.
- **B:** The loss is falling, so learning is happening; a higher learning rate tends to destabilize bootstrapped training rather than fix wrong targets.
- **C:** A working buffer does not rule out bad exploration, wrong targets, or a missing terminal mask.

</details>
