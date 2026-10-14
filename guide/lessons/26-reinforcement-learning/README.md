---
title: Reinforcement learning
summary: "Learn from rewards instead of labels: MDPs and the Bellman equation, Q-learning and SARSA, deep Q-networks, policy gradients, and PPO, the algorithm behind RLHF."
skill: deep-learning
---

# Module 26: Reinforcement learning

Supervised learning has the right answer for every input. Reinforcement learning only gets a score, often late, for a sequence of choices. This module builds the core algorithms from scratch on small grid worlds, so every number can be checked by hand. It is the groundwork for RLHF and the PPO-based and GRPO-style fine-tuning in [Module 15](../15-llms/README.md): there a token is an action, a response is an episode, and a reward model or verifier hands out the score.

| Idea | What it learns | Where it shows up |
|---|---|---|
| MDP and Bellman equation | the shape of the problem; values as expected return | every RL method |
| Temporal-difference learning | action values from one step at a time | Q-learning, SARSA, DQN, critics |
| Function approximation | values for states too many to list | DQN |
| Policy gradient | the policy directly, from sampled returns | REINFORCE, GRPO |
| Trust region and clipping | stable policy updates | PPO, RLHF |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [MDPs, states, actions, and rewards](01-mdps-states-actions-rewards.md) | frame a problem as an MDP, compute returns and values, and run policy evaluation and value iteration |
| 2 | [Q-learning and SARSA](02-q-learning-sarsa.md) | compute TD updates by hand and explain why on-policy and off-policy agents take different paths along a cliff |
| 3 | [Deep Q-networks](03-dqn.md) | train a Q-network with experience replay and a target network, and explain why each is needed |
| 4 | [Policy gradients and REINFORCE](04-policy-gradients-reinforce.md) | derive the policy gradient, use a baseline to cut variance, and connect it to LLM fine-tuning |
| 5 | [Proximal policy optimization](05-ppo.md) | compute the clipped objective and GAE, and explain how PPO and GRPO are used to fine-tune LLMs |

**Interactive lab:** [Q-learning on a grid world](https://sreshtalluri.github.io/ml-ai-learn/labs/q-learning/)
