"""
This module just exists to enable watching the snake play
and learn in real time. Training is much slower since it has to render
all the moves.
"""

import os
import numpy as np

from environment.snake_env import SnakeEnv, EnvConfig
from train.q_learning import encode_state

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
Q_PATH = os.path.join(PROJECT_ROOT, "trained_parameter", "q_table.npy")

def main():
    Q = np.load(Q_PATH)

    cfg = EnvConfig(
        grid_size=15,
        max_steps=500,
        seed=0,
        render_fps=120,
    )

    env = SnakeEnv(cfg, render_mode="human")
    obs, _ = env.reset(seed=np.random.randint(0, 2**31 - 1))

    terminated = False
    truncated = False

    while True:
        s = encode_state(obs)
        a = int(np.argmax(Q[s]))
        obs, r, terminated, truncated, info = env.step(a)

        still_open = env.render()
        if not still_open:
            break

        if terminated or truncated:
            obs, _ = env.reset(seed=np.random.randint(0, 2**31 - 1))
            terminated = truncated = False

    env.close()

if __name__ == "__main__":
    main()