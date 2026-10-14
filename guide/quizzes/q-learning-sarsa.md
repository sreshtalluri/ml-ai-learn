<!-- GENERATED from q-learning-sarsa.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Q-learning and SARSA

Covers the lesson [Q-learning and SARSA](../lessons/26-reinforcement-learning/02-q-learning-sarsa.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/q-learning-sarsa/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

$Q(s,a) = -5$, reward $r = -1$, next-state values $[-4, -3, -6, -5]$, $\alpha = 0.1$, $\gamma = 1$. What is the new $Q(s,a)$ after one **Q-learning** update?

<details>
<summary>Answer</summary>

**-4.9** (within ±0.001)

Target $= -1 + \max(-4,-3,-6,-5) = -1 - 3 = -4$. New $= -5 + 0.1 \times (-4 - (-5)) = -5 + 0.1 = -4.9$.

</details>

## 2. Calculation (medium)

Same transition as before ($Q(s,a) = -5$, $r = -1$, next values $[-4, -3, -6, -5]$, $\alpha = 0.1$, $\gamma = 1$), but the agent's next action is the one with value $-5$. What is the new $Q(s,a)$ under **SARSA**?

<details>
<summary>Answer</summary>

**-5.1** (within ±0.001)

Target $= -1 + (-5) = -6$. New $= -5 + 0.1 \times (-6 - (-5)) = -5 - 0.1 = -5.1$.

</details>

## 3. Calculation (medium)

With ε-greedy, ε = 0.2, and 4 actions with a unique best action, what is the probability the agent takes the best action?

<details>
<summary>Answer</summary>

**0.85** (within ±0.001)

$(1 - 0.2) + 0.2 / 4 = 0.8 + 0.05 = 0.85$. A random move can also pick the best action.

</details>

## 4. Multiple choice (easy)

Why is SARSA called on-policy?

- **A.** It only works with a deterministic policy.
- **B.** Its target uses the action the current (exploring) policy actually takes next, so it learns that policy's values.
- **C.** It updates the policy directly with gradients.
- **D.** It needs a model of the environment.

<details>
<summary>Answer</summary>

**B.** Its target uses the action the current (exploring) policy actually takes next, so it learns that policy's values.

Q-learning's max makes it learn the greedy policy's values while behaving differently (off-policy).

- **A:** SARSA typically runs a stochastic ε-greedy policy.
- **C:** That describes policy-gradient methods.
- **D:** SARSA is model-free.

</details>

## 5. Multiple choice (medium)

On a cliff grid with ε = 0.1 held fixed, Q-learning's greedy path runs along the cliff edge and SARSA's runs one row away. Which statement is true during training?

- **A.** Q-learning earns a higher average return because its path is shorter.
- **B.** SARSA earns a higher average return, because Q-learning sometimes explores off the edge.
- **C.** Both earn exactly the optimal return.
- **D.** SARSA never reaches the goal.

<details>
<summary>Answer</summary>

**B.** SARSA earns a higher average return, because Q-learning sometimes explores off the edge.

While exploring, Q-learning pays −100 now and then; SARSA's longer path costs a couple of extra steps but avoids most falls. Evaluated greedily (ε = 0), Q-learning's path is better.

- **A:** A shorter path doesn't help when random steps next to the cliff cost 100.
- **C:** Exploration costs both of them; neither achieves −6 on every episode.
- **D:** SARSA reaches the goal reliably via its safer path.

</details>

## 6. Multiple choice (hard)

A tabular Q-learning agent on a large maze shows a flat learning curve at the minimum return after 5,000 episodes. Logs show it never once reached the goal. What is the best first fix?

- **A.** Lower the learning rate.
- **B.** Make sure exploration can reach the goal. Use a higher ε or optimistic initial values, a curriculum, or a shaped reward, since no reward signal has ever propagated.
- **C.** Switch to SARSA.
- **D.** Increase γ to 1.

<details>
<summary>Answer</summary>

**B.** Make sure exploration can reach the goal. Use a higher ε or optimistic initial values, a curriculum, or a shaped reward, since no reward signal has ever propagated.

TD learning can only spread value backward from rewards the agent has seen. If the goal was never found, every Q-value reflects only step costs.

- **A:** A smaller α slows learning further and doesn't create a reward signal.
- **C:** SARSA has the same problem; neither has seen the goal.
- **D:** Discounting doesn't matter if the goal is never reached.

</details>

## 7. Multiple choice (medium)

Rewards are noisy, and Q-learning's estimates come out consistently higher than the true values. What causes this, and what fixes it?

- **A.** The discount is too low; raise γ.
- **B.** Maximization bias. The max over noisy estimates picks values that happen to be too high. Double Q-learning picks the action with one table and evaluates it with another.
- **C.** ε is too small; increase exploration.
- **D.** The table was initialized at zero; initialize it at −100.

<details>
<summary>Answer</summary>

**B.** Maximization bias. The max over noisy estimates picks values that happen to be too high. Double Q-learning picks the action with one table and evaluates it with another.

$\mathbb{E}[\max_a \hat Q] \ge \max_a \mathbb{E}[\hat Q]$. Decoupling selection from evaluation removes the upward bias. Double DQN is the deep version.

- **A:** The bias comes from the max, not the discount.
- **B:** Exploration doesn't remove the max's bias.
- **D:** Pessimistic initialization hurts exploration and doesn't address noise in the max.

</details>

## 8. Multiple choice (medium)

Episodes are cut off at 100 steps by a time limit. How should the update treat the transition at step 100?

- **A.** As terminal. The target is just $r$.
- **B.** As non-terminal. Still bootstrap from $s'$, because the world didn't end, only the episode was truncated.
- **C.** Skip the update entirely.
- **D.** Set the target to $-100$ to discourage long episodes.

<details>
<summary>Answer</summary>

**B.** As non-terminal. Still bootstrap from $s'$, because the world didn't end, only the episode was truncated.

Treating truncation as termination teaches the agent that states near step 100 have no future. Gymnasium separates `terminated` from `truncated` for this reason.

- **A:** That's correct only for real terminal states such as the goal or a pit.
- **C:** Dropping the transition throws away valid experience.
- **D:** This invents a penalty that is not part of the task.

</details>

## 9. Multiple choice (hard)

You train a warehouse robot in the real building, not a simulator. It must keep exploring a little after deployment, and collisions are expensive. Which tabular method fits best?

- **A.** Q-learning, because it converges to the optimal policy.
- **B.** SARSA or Expected SARSA, because it optimizes the return of the exploring policy actually being run, so it keeps a margin from hazards.
- **C.** Pure random exploration with ε = 1.
- **D.** Value iteration, because it is exact.

<details>
<summary>Answer</summary>

**B.** SARSA or Expected SARSA, because it optimizes the return of the exploring policy actually being run, so it keeps a margin from hazards.

When exploration continues and mistakes are costly, on-policy methods account for their own randomness.

- **A:** Q-learning's plan is optimal only for a greedy agent; with continued exploration it walks close to hazards.
- **C:** Random behavior never exploits what it has learned.
- **D:** Value iteration needs a known transition model, which you don't have for a real building.

</details>
