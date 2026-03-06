"""
Monte Carlo Tree Search (MCTS) for the Snake environment.

Uses a simple one-step lookahead: for each action, run multiple random rollouts,
pick the action with the highest average return.

Run directly for headless training:
    python train/mcts.py

Run with visible pygame window via watch.py:
    python train/watch.py mcts
"""

import os
import sys
import random
import numpy as np
import gymnasium as gym

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.snake_env import SnakeEnv, EnvConfig
from utils.seed import seed_everything


class RenderAfterStepWrapper(gym.Wrapper):
    """Calls env.render() after every step and reset so the pygame window updates."""

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.env.render()
        return obs, r, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.render()
        return obs, info


def make_env(seed: int, render_mode: str | None = None):
    env_cfg = EnvConfig(
        grid_size=15,
        max_steps=500,
        seed=seed,
        render_fps=120 if render_mode == "human" else 12,
        window_title=os.environ.get("WISE_SNAKE_ALGO_NAME", "Wise Snake"),
    )
    env = SnakeEnv(env_cfg, render_mode=render_mode)
    if render_mode == "human":
        env = RenderAfterStepWrapper(env)
    return env


def _rollout_return(env: SnakeEnv, state, action: int, max_steps: int, gamma: float = 0.99) -> float:
    """Run one rollout: set state, take action, then random actions until done. Return discounted sum of rewards."""
    env.set_state(state)
    obs, r, term, trunc, _ = env.step(action)
    total = float(r)
    discount = gamma
    steps = 1
    while (not term) and (not trunc) and steps < max_steps:
        # random valid action (avoid reverse if we want; env already blocks reverse)
        a = random.randint(0, 3)
        obs, r, term, trunc, _ = env.step(a)
        total += discount * float(r)
        discount *= gamma
        steps += 1
    return total


def _mcts_action(snake_env: SnakeEnv, state, n_rollouts: int, max_rollout_steps: int, gamma: float) -> int:
    """For each of 4 actions, run n_rollouts rollouts; return action with highest mean return."""
    action_returns: list[list[float]] = [[] for _ in range(4)]
    for a in range(4):
        for _ in range(n_rollouts):
            ret = _rollout_return(snake_env, state, a, max_rollout_steps, gamma)
            action_returns[a].append(ret)
    means = [np.mean(returns) if returns else -1e9 for returns in action_returns]
    return int(np.argmax(means))


def train(
    seed: int = 0,
    episodes: int = 500,
    max_steps: int = 500,
    n_rollouts: int = 25,
    max_rollout_steps: int = 80,
    gamma: float = 0.99,
    render_mode: str | None = None,
    log_every: int = 100,
):
    seed_everything(seed)

    play_env = make_env(seed, render_mode=render_mode)
    # Unwrapped SnakeEnv for get_state/set_state
    snake_play = play_env.unwrapped if hasattr(play_env, "unwrapped") else play_env
    # Separate env for rollouts (no render)
    rollout_env = SnakeEnv(
        EnvConfig(grid_size=15, max_steps=max_steps, seed=seed + 1),
        render_mode=None,
    )

    recent_returns = []
    recent_lengths = []
    recent_fruits = []
    recent_snake_lengths = []
    last_avg_ret = None
    last_avg_len = None
    last_avg_snake_len = None
    last_avg_fruit = None

    for ep in range(1, episodes + 1):
        obs, _ = play_env.reset(seed=seed + ep)
        if render_mode == "human":
            still_open = play_env.render()
            if not still_open:
                play_env.close()
                rollout_env.close()
                print("Training stopped.")
                return
        state = snake_play.get_state()
        ep_return = 0.0
        ep_steps = 0
        terminated = False
        truncated = False

        while (not terminated) and (not truncated) and ep_steps < max_steps:
            action = _mcts_action(rollout_env, state, n_rollouts, max_rollout_steps, gamma)
            obs, r, terminated, truncated, info = play_env.step(action)
            state = snake_play.get_state()
            ep_return += float(r)
            ep_steps += 1

            if render_mode == "human":
                still_open = play_env.render()
                if not still_open:
                    play_env.close()
                    rollout_env.close()
                    print("Training stopped.")
                    return

        final_length = len(snake_play._snake)
        fruits_eaten = final_length - 3
        recent_returns.append(ep_return)
        recent_lengths.append(ep_steps)
        recent_fruits.append(fruits_eaten)
        recent_snake_lengths.append(final_length)
        if len(recent_returns) > log_every:
            recent_returns.pop(0)
            recent_lengths.pop(0)
            recent_fruits.pop(0)
            recent_snake_lengths.pop(0)

        if ep % log_every == 0:
            avg_ret = sum(recent_returns) / len(recent_returns)
            avg_len = sum(recent_lengths) / len(recent_lengths)
            avg_snake_len = sum(recent_snake_lengths) / len(recent_snake_lengths)
            avg_fruit = sum(recent_fruits) / len(recent_fruits)
            last_avg_ret = avg_ret
            last_avg_len = avg_len
            last_avg_snake_len = avg_snake_len
            last_avg_fruit = avg_fruit
            algo_label = os.environ.get("WISE_SNAKE_ALGO_NAME")
            prefix = f"[{algo_label}] " if algo_label else ""
            print(
                f"{prefix}Episode {ep:5d} | "
                f"avg_return={avg_ret:.2f} | avg_steps={avg_len:.1f} | "
                f"avg_len={avg_snake_len:.1f} | avg_fruit={avg_fruit:.1f}"
            )

    # If launched via train_all, write final stats for comparison.
    if os.environ.get("WISE_SNAKE_FROM_TRAIN_ALL") == "1":
        if last_avg_ret is None and recent_returns:
            last_avg_ret = sum(recent_returns) / len(recent_returns)
            last_avg_len = sum(recent_lengths) / len(recent_lengths)
            last_avg_snake_len = sum(recent_snake_lengths) / len(recent_snake_lengths)
            last_avg_fruit = sum(recent_fruits) / len(recent_fruits)
        if last_avg_ret is not None:
            stats_path = os.path.join(PROJECT_ROOT, "trained_parameter", "mcts_final_stats.txt")
            try:
                with open(stats_path, "w", encoding="utf-8") as f:
                    f.write(
                        f"avg_return={last_avg_ret:.4f} | "
                        f"avg_steps={last_avg_len:.4f} | "
                        f"avg_len={last_avg_snake_len:.4f} | "
                        f"avg_fruit={last_avg_fruit:.4f}\n"
                    )
            except OSError:
                pass

    play_env.close()
    rollout_env.close()
    print("MCTS training finished.")


def main(render_mode: str | None = None, episodes: int = 3000):
    train(
        seed=0,
        episodes=episodes,
        max_steps=500,
        n_rollouts=25,
        max_rollout_steps=80,
        gamma=0.99,
        render_mode=render_mode,
        log_every=100,
    )


def run_training(render: bool = False, total: int = 3000):
    """Run MCTS training. total=episodes. If render=True, display the game window while running."""
    main(render_mode="human" if render else None, episodes=total)


if __name__ == "__main__":
    import sys
    try:
        total = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    except (ValueError, IndexError):
        total = 3000
    main(episodes=total)
