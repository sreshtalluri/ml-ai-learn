<!-- GENERATED from policy-gradients-reinforce.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Policy gradients and REINFORCE

Covers the lesson [Policy gradients and REINFORCE](../lessons/26-reinforcement-learning/04-policy-gradients-reinforce.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/policy-gradients-reinforce/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

An episode has rewards $[-1, -1, -1]$ and $\gamma = 0.9$. What is the return-to-go $G_0$?

<details>
<summary>Answer</summary>

**-2.71** (within ±0.001)

Backward: $G_2 = -1$, $G_1 = -1 + 0.9 \times (-1) = -1.9$, $G_0 = -1 + 0.9 \times (-1.9) = -2.71$.

</details>

## 2. Calculation (medium)

A softmax policy gives probabilities $[0.2, 0.5, 0.3]$ for three actions. The agent took action 2 (the second). What is $\partial \log \pi(a{=}2) / \partial z_2$, the gradient with respect to that action's own logit?

<details>
<summary>Answer</summary>

**0.5** (within ±0.001)

For a softmax, $\partial \log \pi(a) / \partial z_j = \mathbb{1}[j = a] - \pi(j)$. For $j = a$: $1 - 0.5 = 0.5$.

</details>

## 3. Calculation (hard)

Logits at a state are $[1, 0]$ (two actions). The agent took action 2, advantage $A = 2$, learning rate $0.5$. After one REINFORCE step on the logits, what is the probability of action 2? (Round to 3 decimals.)

<details>
<summary>Answer</summary>

**0.614** (within ±0.002)

$\pi = [0.7311, 0.2689]$. Gradient of $\log \pi(2)$: $[-0.7311, 0.7311]$. Step $= 0.5 \times 2 \times$ gradient $= [-0.7311, 0.7311]$. New logits $[0.2689, 0.7311]$. New $\pi(2) = e^{0.7311}/(e^{0.2689} + e^{0.7311}) = 2.0774 / 3.3859 = 0.6135$.

</details>

## 4. Multiple choice (medium)

Why does subtracting a baseline $b(s)$ from the return leave the policy gradient unbiased?

- **A.** Because the baseline is learned, so its errors average out over training.
- **B.** Because $\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a \mid s)] = \nabla \sum_a \pi(a \mid s) = \nabla 1 = 0$, so any term that doesn't depend on the action contributes zero on average.
- **C.** Because the baseline is subtracted from every step, so it cancels between the first and last step.
- **D.** Because rewards are normalized, so a constant shift has no effect.

<details>
<summary>Answer</summary>

**B.** Because $\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a \mid s)] = \nabla \sum_a \pi(a \mid s) = \nabla 1 = 0$, so any term that doesn't depend on the action contributes zero on average.

The expected score function is zero, so $b(s)$ times it is zero in expectation for any state-only $b$. Variance changes; the mean does not.

- **A:** A bad baseline is still unbiased, and a learned one isn't unbiased because it is learned. The zero-mean score function is the reason.
- **C:** Nothing telescopes here; each step's term is zero in expectation on its own.
- **D:** Unbiasedness holds without any normalization.

</details>

## 5. Multiple choice (hard)

A teammate proposes the baseline $b(s, a) = Q(s, a)$, "since it's a better estimate than $V(s)$". What is wrong?

- **A.** Nothing; $Q$ is strictly more informative, so variance drops further.
- **B.** The advantage $G_t - Q(s_t, a_t)$ would always be exactly zero.
- **C.** A baseline that depends on the action is no longer zero-mean against $\nabla \log \pi(a \mid s)$, so the gradient becomes biased.
- **D.** $Q$ cannot be learned with a neural network.

<details>
<summary>Answer</summary>

**C.** A baseline that depends on the action is no longer zero-mean against $\nabla \log \pi(a \mid s)$, so the gradient becomes biased.

The proof that baselines are free needs $b$ to factor out of the sum over actions. With $b(s, a)$ it doesn't, so the estimator is biased. The usual advantage is $Q(s,a) - V(s)$: $Q$ (or its sample $G_t$) is the signal, $V$ is the baseline.

- **A:** Lower variance is worthless if the gradient points in the wrong direction on average.
- **B:** $G_t$ is a sample and $Q$ an expectation, so they differ; the problem is bias, not a zero signal.
- **D:** DQN learns $Q$ with a neural network; learnability isn't the issue.

</details>

## 6. Multiple choice (hard)

One of eight REINFORCE seeds sits at the worst possible return from episode 30 onward while the others reach the optimum. Its policy puts probability 1.0 on walking into a wall at the start. What best explains this?

- **A.** The baseline is biased, so this seed's gradient points the wrong way.
- **B.** The discount factor is too high.
- **C.** The environment is stochastic, and this seed was unlucky with transitions.
- **D.** Entropy collapse. Early noisy updates made a bad action near-certain, every episode then looks the same, and the gradient through near-1 probabilities is near zero, so it never escapes.

<details>
<summary>Answer</summary>

**D.** Entropy collapse. Early noisy updates made a bad action near-certain, every episode then looks the same, and the gradient through near-1 probabilities is near zero, so it never escapes.

With $p \approx 1$, $\nabla \log \pi = \text{onehot} - p \approx 0$, and identical episodes give no contrast. Fixes are a lower learning rate, an entropy bonus, advantage normalization, and larger batches.

- **A:** A state-only baseline cannot bias the gradient, and the other seeds use the same code.
- **B:** Discount affects all seeds equally and doesn't make one policy deterministic.
- **C:** The grid is deterministic; the randomness is in the policy's own early samples.

</details>

## 7. Multiple choice (easy)

Why weight $\nabla \log \pi(a_t \mid s_t)$ by the return-to-go $G_t$ rather than the whole episode's return?

- **A.** Rewards before step $t$ can't be affected by $a_t$; including them adds variance without changing the expected gradient.
- **B.** It makes the estimator biased but faster.
- **C.** Because the discount factor would otherwise be applied twice.
- **D.** It's required for the softmax gradient to be correct.

<details>
<summary>Answer</summary>

**A.** Rewards before step $t$ can't be affected by $a_t$; including them adds variance without changing the expected gradient.

Past rewards are independent of the current action given the past, so their term has zero expectation. Dropping them is a free variance reduction (causality).

- **B:** Returns-to-go keep the estimator unbiased.
- **C:** Discounting is unrelated; the issue is which rewards an action can influence.
- **D:** The softmax gradient is the same either way; only its weight changes.

</details>

## 8. Multiple choice (medium)

You fine-tune an LLM with a unit-test reward (1 if tests pass, else 0) given once at the end of each response. Which advantage scheme is the closest to what GRPO does?

- **A.** Train a separate value network per token and use GAE with $\lambda = 0.95$.
- **B.** Use the raw reward (0 or 1) as the advantage for every token.
- **C.** Sample several responses per prompt and give every token of response $i$ the advantage $(R_i - \text{mean}(R)) / \text{std}(R)$.
- **D.** Use a replay buffer of past responses and a target network.

<details>
<summary>Answer</summary>

**C.** Sample several responses per prompt and give every token of response $i$ the advantage $(R_i - \text{mean}(R)) / \text{std}(R)$.

GRPO uses the group as a baseline: a REINFORCE estimator with no learned critic, plus PPO-style clipping and a KL term to a reference model.

- **A:** That is PPO-style RLHF with a critic, the component GRPO removes.
- **B:** With rewards in $\{0, 1\}$, failing responses would get zero signal and every passing one the same push; there's no baseline.
- **D:** That's DQN machinery; GRPO is on-policy.

</details>

## 9. Multiple choice (medium)

Which situation most favors a policy-gradient method over DQN?

- **A.** A small discrete action set where environment samples are very expensive.
- **B.** A continuous 6-dimensional action space, or an action space as large as a 100,000-token vocabulary.
- **C.** A problem where you need to reuse old data from a replay buffer many times.
- **D.** A deterministic environment with 16 states.

<details>
<summary>Answer</summary>

**B.** A continuous 6-dimensional action space, or an action space as large as a 100,000-token vocabulary.

Policy gradients only need $\log \pi$ of the sampled action, so continuous or huge action spaces are natural. DQN needs a max over all actions.

- **A:** Small discrete actions with expensive samples favor off-policy, replay-based DQN.
- **C:** Replay is a strength of off-policy value methods; REINFORCE is on-policy.
- **D:** A tiny tabular problem is easily solved by either, or by plain Q-learning.

</details>
