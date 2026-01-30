---
layout: default
title: Proposal
---
## Summary of the Project

In this project, we aim to train an agent called the Wise Snake to play an enhanced version of the classic snake game. 

The environment extends from the standard snake setup by including some of these possible addon:
- Randomly generated barriers 
- Multiple types of fruits with different reward values (e.g., +1, +3, +5). 
- Multiple snakes
- Action Penalties (e.g., turning -0.3)
- Snake only sees a local window

The goal is to train an agent that learns an optimal policy for maximizing cumulative reward while avoiding collisions. 


## Project Goals

- Minimum goal:     The snake can at least survive for a resonable time
- Realistic goal:   The snake try to get a decent score without dying
- Moonshot goal:    The snake can get a relatively average high score per certain steps

## AI/ML Algorithms

- Use Tabular Q as a baseline to start with
- Use DQN if the baseline is not sufficient enough
- Explore other on-policy based algorithm if needed

## Evaluation Plan

The agent will be scored using a weighted combination of survival time (number of steps) and accumulated reward.
During early training with low survival time, the evaluation metric will place greater emphasis on survival time. Once the agent consistently surpasses that survival threshold, the weighting will shift to emphasize cumulative reward.  

## AI Tool Usage

None was used so far
