<!-- GENERATED from mdps-states-actions-rewards.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: MDPs, states, actions, and rewards

Covers the lesson [MDPs, states, actions, and rewards](../lessons/26-reinforcement-learning/01-mdps-states-actions-rewards.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/mdps-states-actions-rewards/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

An episode gives rewards $[0, 0, -1, 10]$ with $\gamma = 0.5$. What is the return $G_0$?

<details>
<summary>Answer</summary>

**1** (within ±0.001)

Work backward: $G_3 = 10$, $G_2 = -1 + 0.5 \times 10 = 4$, $G_1 = 0 + 0.5 \times 4 = 2$, $G_0 = 0 + 0.5 \times 2 = 1$.

</details>

## 2. Calculation (easy)

Using the rule of thumb $1/(1-\gamma)$, roughly how many steps ahead does an agent with $\gamma = 0.98$ look?

<details>
<summary>Answer</summary>

**50** (within ±0.5)

$1 / (1 - 0.98) = 1 / 0.02 = 50$ steps.

</details>

## 3. Calculation (medium)

Uniform random policy over 4 actions, $\gamma = 0.9$, every move costs $-1$. From cell $s$ the four actions lead to cells with values $-8$, $-6$, $-8$, and $0$ (the terminal goal). What is $V^\pi(s)$ according to the Bellman expectation equation?

<details>
<summary>Answer</summary>

**-5.95** (within ±0.001)

Each action contributes $0.25 \times (-1 + 0.9\,V(s'))$: $-1 + 0.9 \times (-8) = -8.2$, $-1 + 0.9 \times (-6) = -6.4$, $-8.2$ again, and $-1 + 0 = -1$. Sum $= -23.8$; times $0.25$ gives $-5.95$.

</details>

## 4. Multiple choice (medium)

An agent plays Pong from a single screen frame and can't learn to hit the ball. Which change addresses the root cause?

- **A.** Increase the discount factor to 0.999.
- **B.** Stack the last few frames (or use a recurrent network) so the state includes the ball's velocity.
- **C.** Give a larger reward for hitting the ball.
- **D.** Switch from Q-learning to a policy-gradient method.

<details>
<summary>Answer</summary>

**B.** Stack the last few frames (or use a recurrent network) so the state includes the ball's velocity.

One frame shows the ball's position but not its direction, so the state is not Markov. The fix belongs in the state representation.

- **A:** A longer horizon doesn't add the missing velocity information.
- **C:** Reward shaping can't help if the agent can't tell which way the ball is moving.
- **D:** Any algorithm struggles when two situations that need different actions look identical.

</details>

## 5. Multiple choice (easy)

Which statement correctly separates reward, return, and value?

- **A.** Reward and return are the same; value is the reward at the goal.
- **B.** Reward is one step's number; return is the discounted sum over one episode; value is the expected return from a state.
- **C.** Value is the reward the environment gives; return is what the agent predicts.
- **D.** Return is the expected reward of the next step only.

<details>
<summary>Answer</summary>

**B.** Reward is one step's number; return is the discounted sum over one episode; value is the expected return from a state.

Rewards are observed per step, returns are sums over one trajectory, and values average returns over everything the policy and environment might do.

- **A:** Return sums many rewards, and value is an expectation, not a single reward.
- **C:** The environment gives rewards; values are what the agent estimates.
- **D:** Return includes all future rewards, discounted, not just the next one.

</details>

## 6. Multiple choice (medium)

A continuing task (no terminal state, no time limit) uses $\gamma = 1$ and a reward of $+1$ per step. What goes wrong?

- **A.** Nothing; $\gamma = 1$ is always the safest choice.
- **B.** The agent becomes myopic.
- **C.** Returns and values are unbounded, so value estimates diverge.
- **D.** The Markov property is violated.

<details>
<summary>Answer</summary>

**C.** Returns and values are unbounded, so value estimates diverge.

With no end and no discount the sum of rewards grows without limit. Use $\gamma < 1$ or a finite horizon.

- **A:** $\gamma = 1$ is only safe when every episode is guaranteed to end.
- **B:** Myopia comes from a small $\gamma$, not $\gamma = 1$.
- **D:** The Markov property concerns the state, not the discount.

</details>

## 7. Multiple choice (hard)

A summarization model trained with RL against a learned reward model starts writing longer and longer summaries, and human raters like them less. What is the most likely diagnosis?

- **A.** The discount factor is too low.
- **B.** Reward hacking. The policy exploits a bias in the reward model (it favors length), so the proxy reward rises while true quality falls.
- **C.** The state is not Markov.
- **D.** The learning rate is too high, so training diverged.

<details>
<summary>Answer</summary>

**B.** Reward hacking. The policy exploits a bias in the reward model (it favors length), so the proxy reward rises while true quality falls.

The policy optimizes the reward you wrote. Fixes include a KL penalty to the reference model, length-controlled rewards, retraining the reward model on the new failures, and held-out human evaluation.

- **A:** Responses are short episodes, often with $\gamma = 1$; the discount isn't the issue.
- **C:** The context window is the state and is Markov by construction.
- **D:** Divergence would look like collapse, not a steady drift toward one exploitable feature.

</details>

## 8. Multiple choice (medium)

You frame RL fine-tuning of an LLM on math problems with a checker that verifies the final answer. Which MDP formulation fits?

- **A.** State is the last token; action is a whole response; reward is per-token perplexity.
- **B.** State is the prompt plus tokens so far; action is the next token; reward is 1 at the end if the checker passes, else 0; $\gamma = 1$.
- **C.** State is the checker's output; action is the prompt; reward is response length.
- **D.** State is the model weights; action is a gradient step; reward is training loss.

<details>
<summary>Answer</summary>

**B.** State is the prompt plus tokens so far; action is the next token; reward is 1 at the end if the checker passes, else 0; $\gamma = 1$.

This is the standard RL-with-verifiable-rewards setup. The reward is sparse (only at the end), so methods like GRPO compare several sampled answers per prompt.

- **A:** The last token alone isn't Markov, and perplexity isn't the goal.
- **C:** Rewarding length invites reward hacking and doesn't match the goal.
- **D:** That describes the optimizer, not the decision problem the policy faces.

</details>

## 9. Multiple choice (medium)

How does value iteration differ from policy evaluation?

- **A.** Value iteration takes a max over actions in the Bellman update; policy evaluation averages over the policy's action probabilities.
- **B.** Value iteration needs no transition model; policy evaluation does.
- **C.** Policy evaluation finds the optimal policy; value iteration only scores a fixed one.
- **D.** They are the same algorithm with a different stopping rule.

<details>
<summary>Answer</summary>

**A.** Value iteration takes a max over actions in the Bellman update; policy evaluation averages over the policy's action probabilities.

Value iteration iterates the Bellman optimality equation and yields $V^*$; policy evaluation iterates the expectation equation for a fixed $\pi$.

- **B:** Both are dynamic programming and both need the model $P$ and $r$.
- **C:** It's the other way round.
- **D:** The max vs average is a real difference in what they compute.

</details>
