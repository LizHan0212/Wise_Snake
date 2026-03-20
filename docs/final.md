---
layout: default
title: Final Report
---

# Project Summary

Snake is a game that requires planning, long-term reward optimization, and spatial reasoning. Although simple for humans, it is challenging for algorithms because the agent must balance immediate rewards with long-term survival while avoiding collisions and traps.

Traditional approaches such as tabular methods and search-based algorithms struggle due to the large state space and dynamic environment. Reinforcement learning provides a suitable framework, as it allows the agent to learn directly through interaction with the environment instead of relying on hand-crafted rules.

In this project, we trained an AI agent to play an enhanced Snake environment that includes obstacles and multiple fruit types. We explored several reinforcement learning methods—Tabular Q-learning, Deep Q-Network (DQN), and Proximal Policy Optimization (PPO)—to compare their performance in terms of learning efficiency, generalization, and long-term planning.

---

# Approach

## Problem Setup

- Actions: 4 directions (up, down, left, right)  
- State: representation of the board (snake body, fruits, barriers)  
- Goal: maximize cumulative reward  

### Challenges

- Large state space (15×15 grid)  
- Sparse rewards make learning difficult  
- Neural network models require significant computation  

---

## Environment Setup

The environment consists of:

- 15×15 grid  
- 7 randomly placed barriers  
- 7 fruits at all times:
  - 1 Blue fruit (+5 reward)
  - 2 Yellow fruits (+3 reward)
  - 4 Red fruits (+1 reward)

When a fruit is consumed, a new fruit of the same type is randomly generated.

Observation space:

0 = empty  
1 = snake head  
2 = snake body  
3 = fruit  
4 = barrier  

Action space:

0 = up  
1 = down  
2 = left  
3 = right  

Episode termination:
- Collision with wall  
- Collision with barrier  
- Collision with snake body  

<img width="500" height="350" src="https://github.com/user-attachments/assets/e8b5e199-f9be-4b44-b14a-1f24d9039e57" />

---

## State Representation

Initial state encoding included:

- Danger in adjacent cells  
- Current direction  
- Nearest food direction  

We improved the representation by adding:

- Distance to obstacles (raycast-style features)  
- Flood-fill reachable space  
- Manhattan distance to food  

This richer feature-based representation reduces the effective state space and provides the agent with more meaningful spatial information, improving learning efficiency.

---

## Reward Function 

Final reward design:

- Positive reward for food collection  
- Penalty for collision  
- Turning penalty  
- Idle step penalty  

We removed survival-only rewards to prevent the agent from learning degenerate behaviors such as circling without progress, and to better align rewards with the actual objective of collecting food.

---

## Algorithms

### Monte Carlo Tree Search (MCTS)

- Builds a search tree from current state  
- Uses simulated rollouts to evaluate actions  

Limitations:
- Large branching factor  
- Limited planning depth  
- Does not learn across episodes  

---

### Tabular Q-Learning

- Stores Q-values for (state, action) pairs  
- Uses epsilon-greedy exploration  
- Selects action with highest Q-value  

Limitations:
- Large state space  
- Requires exact state matches  

---

### Deep Q-Network (DQN)

- Neural network approximates Q-values  
- Uses replay buffer and target network for stability  
- Can generalize across similar states  

Limitations:
- Sparse rewards cause noisy and delayed updates  
- Learning is unstable without sufficient data  
- Requires significantly more training steps than tabular methods  

---

### Proximal Policy Optimization (PPO)

- Uses actor-critic architecture  
- Learns both policy and value function  
- Uses clipped objective for stable updates  

Limitations:
- On-policy (cannot reuse old data)  
- Requires many environment interactions  
- Higher computational cost  

---

## Training Setup

- ~50,000 episodes (up to 500 steps per episode)  
- Randomized obstacle placement each episode  
- Episode ends on collision or full grid  
- Training monitored using TensorBoard and console logs 

Key hyperparameters (DQN):

For DQN, we used Stable-Baselines3 with an MLP policy. Key hyperparameters included a learning rate of 1e-4, replay buffer size of 100,000, batch size of 64, discount factor 0.99, and target update interval of 1000. Exploration followed an epsilon-greedy strategy decaying to 0.05.

The final DQN setup also used feature-based observations, reward shaping, and randomized reset seeds across episodes.

---

# Evaluation

## Metrics

- Average reward per episode  
- Survival time (steps)  
- Snake length  

---

## Quantitative Results (50k episodes)

| Model | Avg Return | Avg Steps | Avg Length | Training Time |
|------|-----------|----------|-----------|--------------|
| MCTS | -2.95 | 6.5 | 3.2 | 3.66 mins |
| Tabular Q | 24.72 | 83.5 | 18.9 | 4.52 mins |
| DQN | -0.58 | 12.8 | 3.6 | 12.22 mins |
| PPO | 121.19 | 418.7 | 74.8 | 4.66 hours |

PPO achieved the best performance among all tested methods. Its actor-critic structure and trajectory-based updates allow it to better handle delayed rewards and produce more stable learning, although at a higher computational cost.

---

## Qualitative Results

Observed behaviors during training:

- Zigzagging between actions  
- Wall-following behavior  
- Circling to avoid collisions  
- Food fixation (moving toward food without considering danger)  

To address these behaviors, we adjusted:

- Turning penalties  
- Step penalties  
- Collision penalties  
- State representation  

---

## Failure Cases

- Sparse rewards slow down learning  
- MCTS cannot handle long-term planning due to limited rollout depth  
- Some agents prioritize survival over reward optimization  

### DQN Performance Explanation

The DQN model underperformed compared to Tabular Q-learning and PPO due to the combination of high-dimensional input and sparse reward signals. Since rewards are delayed, it is difficult for the network to correctly attribute value to earlier actions. With limited training steps, the model does not receive sufficient consistent feedback to learn stable Q-value estimates, resulting in slow and unstable learning.

---

## Before vs After Training

- Before training: random movement, frequent collisions  
- After training: improved navigation, longer survival, better reward accumulation  

---

# Resources Used

Libraries:
- Gymnasium  
- NumPy  
- PyTorch  
- Stable-Baselines3  

Tools:
- TensorBoard for training visualization and analysis  

References:
- Sutton & Barto, *Reinforcement Learning: An Introduction*  
- Stable-Baselines3 Documentation  
- OpenAI Gym / Gymnasium Documentation  

AI Usage:
- AI tools (ChatGPT) were used to assist with debugging, understanding reinforcement learning concepts, and refining code structure.
