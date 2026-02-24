---
layout: default
title: Status
---

## Project Summary

WiseSnake trains an agent to play an enhanced version of the Snake game in a 15×15 grid environment with multiple fruit types and randomly generated barriers. The objective is to maximize cumulative reward while avoiding collisions with walls, barriers, or the snake’s own body.

At this stage, we have implemented tabular Q-learning baseline and started DQN implementation using Stable-Baselines3. Both methods have been trained and evaluated under randomized environment configurations.

---

## Approach

### Environment Setup

The environment consists of:

- 15×15 grid
- 7 randomly placed barriers
- 7 fruits at all times:
  - 1 Blue fruit (+5 reward)
  - 2 Yellow fruits (+3 reward)
  - 4 Red fruits (+1 reward)

When a fruit is consumed, a new fruit of the same type is randomly generated.

Observation space:
15×15 integer grid encoding:

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

<img width="500" height="350" alt="photo3" src="https://github.com/user-attachments/assets/e8b5e199-f9be-4b44-b14a-1f24d9039e57" />

### Reward Function 

We refined the reward structure to prevent degenerate behaviors:

- +1 / +3 / +5 for fruit consumption  
- −0.2 for turning  
- Additional penalty for death  
- Small negative penalty for steps without fruit  

The step penalty discourages the agent from learning infinite looping behavior.  
The death penalty accelerates learning of collision avoidance.

---

### Tabular Q-Learning Baseline

We implemented Tabular Q-learning with a reduced feature-based state representation due to the intractability of the full 15×15 state space.

Additional improvements:

- Masked illegal backward moves inside `epsilon_greedy()` so the agent does not explore moves disallowed by the environment.
- This prevents unnecessary branching in Q-table updates.
- Added a real-time rendering module to visually observe learning behavior (used for qualitative analysis).

<img width="300" height="400" alt="photo1" src="https://github.com/user-attachments/assets/59acd81c-3c4c-4e45-875f-17373b4f9275" />            <img width="300" height="400" alt="47c5c4bbdefc9e1d1175abc2255870bd" src="https://github.com/user-attachments/assets/4c008fef-642e-4378-a4c8-1c509cc31bb4" />
### Deep Q-Network (DQN)



## Evaluation

We evaluate using:

- Average survival steps
- Average reward points


Tabular Q-Learning:
- Average 154 survival steps per run
- Average 50 reward points per run
- 3.26 reward points per 10 steps



---

## Remaining Goals and Challenges

### Goals


### Challenges


---

## Resources Used

- NumPy
- PyTorch (via SB3)

AI tools were used for debugging and data analysis.
