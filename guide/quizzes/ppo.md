<!-- GENERATED from ppo.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Proximal policy optimization (PPO)

Covers the lesson [Proximal policy optimization (PPO)](../lessons/26-reinforcement-learning/05-ppo.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/ppo/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

With $\varepsilon = 0.2$, a sample has ratio $r = 1.5$ and advantage $\hat A = 2$. What is $\min(r \hat A, \operatorname{clip}(r, 0.8, 1.2)\, \hat A)$?

<details>
<summary>Answer</summary>

**2.4** (within ±0.001)

Unclipped $1.5 \times 2 = 3.0$; clipped $1.2 \times 2 = 2.4$; the min is $2.4$, the clipped term, so this sample has zero gradient.

</details>

## 2. Calculation (medium)

Same $\varepsilon = 0.2$. A sample has $r = 1.5$ and $\hat A = -1$. What is the clipped objective for this sample?

<details>
<summary>Answer</summary>

**-1.5** (within ±0.001)

Unclipped $-1.5$; clipped $1.2 \times (-1) = -1.2$; the min is $-1.5$, the unclipped term. A bad action was made more likely, so the gradient flows fully and pushes it back down.

</details>

## 3. Calculation (hard)

The last two steps of an episode have rewards $-1, -1$, critic values $V = -1.8, -0.9$, and the episode then terminates. With $\gamma = 0.99$, $\lambda = 0.95$, what is the GAE advantage of the second-to-last step?

<details>
<summary>Answer</summary>

**-0.18505** (within ±0.0005)

$\delta_{\text{last}} = -1 + 0 + 0.9 = -0.1$. $\delta_{\text{prev}} = -1 + 0.99 \times (-0.9) + 1.8 = -0.091$. $\hat A = -0.091 + 0.9405 \times (-0.1) = -0.18505$.

</details>

## 4. Multiple choice (easy)

What does the clip in PPO's objective accomplish?

- **A.** It bounds the advantage estimates to $[-\varepsilon, \varepsilon]$.
- **B.** It removes the incentive to move a sampled action's probability more than about $\varepsilon$ (relative) in the direction its advantage favors, so one batch can be reused for several epochs without the policy running away.
- **C.** It guarantees the KL divergence between old and new policy stays below $\varepsilon$.
- **D.** It clips gradient norms to $\varepsilon$.

<details>
<summary>Answer</summary>

**B.** It removes the incentive to move a sampled action's probability more than about $\varepsilon$ (relative) in the direction its advantage favors, so one batch can be reused for several epochs without the policy running away.

Outside the band, the min picks a constant term, so the gradient is zero for that sample. It is an approximate trust region.

- **A:** The advantage isn't clipped; the ratio is.
- **C:** There's no hard KL guarantee; many epochs or shared weights can still move the policy far. That's why KL is monitored.
- **D:** Gradient-norm clipping is a separate, optional safeguard.

</details>

## 5. Multiple choice (hard)

In your PPO code the clip fraction is exactly 0.00 on every update, even with 10 epochs and a large learning rate, and the policy is unstable. What is the most likely bug?

- **A.** The clip range is too large.
- **B.** Advantages are normalized.
- **C.** `logp_old` is recomputed from the current policy inside the epoch loop, so the ratio is always 1 and the clip never acts.
- **D.** The value loss coefficient is too small.

<details>
<summary>Answer</summary>

**C.** `logp_old` is recomputed from the current policy inside the epoch loop, so the ratio is always 1 and the clip never acts.

The old log-probs must be frozen from the policy that collected the data, before any update. If they track the current policy, $r = 1$ always and PPO degenerates into repeated vanilla policy-gradient steps on stale data.

- **A:** A large $\varepsilon$ lowers the clip fraction but wouldn't make it exactly zero across all updates at a large learning rate.
- **B:** Normalization is standard and doesn't prevent clipping.
- **D:** A weak critic gives noisier advantages but doesn't affect the ratio.

</details>

## 6. Multiple choice (medium)

Approx KL per update jumps from about 0.02 to 0.5 and returns become erratic. Which change is the best first step?

- **A.** Lower the learning rate and/or the number of epochs, and stop an update's epochs early once KL passes a target such as 0.02.
- **B.** Increase the clip range to 0.5 so fewer samples are clipped.
- **C.** Remove advantage normalization.
- **D.** Increase the discount factor to 0.999.

<details>
<summary>Answer</summary>

**A.** Lower the learning rate and/or the number of epochs, and stop an update's epochs early once KL passes a target such as 0.02.

High KL means each update moves the policy too far for the ratio correction to be reliable. Step size and epoch count are the direct levers; early stopping on KL is a cheap guard.

- **B:** A larger clip range lets the policy move even further per update.
- **C:** Without normalization, step sizes depend on reward scale, which usually makes things worse.
- **D:** The discount changes the horizon, not how far each update moves the policy.

</details>

## 7. Multiple choice (medium)

In classic PPO-based RLHF for an LLM, which models are in play during the RL stage?

- **A.** Only the policy; the reward is computed by a program.
- **B.** The policy and a frozen reference model only.
- **C.** The policy (trained), a frozen reference model for the KL penalty, a reward model scoring responses, and a value model (critic, trained).
- **D.** The policy and a replay buffer of human-written answers.

<details>
<summary>Answer</summary>

**C.** The policy (trained), a frozen reference model for the KL penalty, a reward model scoring responses, and a value model (critic, trained).

Four models: that memory and tuning cost is a main reason DPO and GRPO became popular.

- **A:** That describes RL with a verifiable reward, and even then you usually keep a reference model.
- **B:** That is DPO's setup, which has no RL loop.
- **D:** PPO is on-policy; it samples fresh responses rather than replaying human answers.

</details>

## 8. Multiple choice (medium)

You're training a math-reasoning model where each final answer can be checked exactly. GPU memory is tight. Which approach fits best?

- **A.** PPO with a learned reward model and a value model.
- **B.** GRPO, which samples a group of answers per prompt, uses group-normalized rewards as advantages (no critic), and keeps PPO's clip plus a KL term to a reference.
- **C.** DQN over the vocabulary.
- **D.** Supervised fine-tuning only, since RL can't use exact-match rewards.

<details>
<summary>Answer</summary>

**B.** GRPO, which samples a group of answers per prompt, uses group-normalized rewards as advantages (no critic), and keeps PPO's clip plus a KL term to a reference.

A verifier replaces the reward model, and the group baseline replaces the critic, saving two large models' worth of memory.

- **A:** A learned reward model is unnecessary with an exact checker, and the critic costs memory you don't have.
- **C:** A max over a 100k-token action space at every step is impractical; LLM RL uses policy gradients.
- **D:** Exact-match rewards are the setting where RL with verifiable rewards works best.

</details>

## 9. Multiple choice (hard)

Your episodes are cut off at 50 steps by a time limit. When computing GAE for a cut-off episode, what should the value of the state after the last step be?

- **A.** The critic's estimate $V(s_{50})$, because the task didn't end; only your data collection stopped.
- **B.** Zero, because the episode is over.
- **C.** The mean return of the batch.
- **D.** The last reward, repeated.

<details>
<summary>Answer</summary>

**A.** The critic's estimate $V(s_{50})$, because the task didn't end; only your data collection stopped.

Truncation is not termination. Treating it as terminal teaches the critic that the world ends at step 50, biasing values downward or upward depending on the reward sign.

- **B:** Zero is right only for a true terminal state.
- **C:** The batch mean is unrelated to the value of that specific state.
- **D:** That has no basis in the Bellman equation.

</details>
