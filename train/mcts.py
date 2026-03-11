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
from collections import deque

import numpy as np
import gymnasium as gym

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.snake_env import SnakeEnv, EnvConfig
from utils.seed import seed_everything

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]


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
    """Run one rollout: set state, take action, then heuristic-guided actions until done.

    The heuristic prefers moves that keep the snake alive and move its head closer
    to the nearest fruit, which makes rollouts much more informative than pure random
    trajectories in this sparse, mostly-negative reward environment.
    """
    env.set_state(state)
    obs, r, term, trunc, _ = env.step(action)
    total = float(r)
    discount = gamma
    steps = 1

    # Precompute action->(dr,dc) mapping (same as env).
    action_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _find_head(obs_arr: np.ndarray) -> tuple[int, int] | None:
        pos = np.argwhere(obs_arr == 1)
        if pos.size == 0:
            return None
        r_h, c_h = pos[0]
        return int(r_h), int(c_h)

    def _nearest_fruit(obs_arr: np.ndarray) -> tuple[int, int] | None:
        fruits = np.argwhere(obs_arr == 3)
        if fruits.size == 0:
            return None
        # Return fruit with minimum Manhattan distance to head.
        head = _find_head(obs_arr)
        if head is None:
            return None
        hr, hc = head
        best = None
        best_d = 10**9
        for fr, fc in fruits:
            fr_i, fc_i = int(fr), int(fc)
            d = abs(fr_i - hr) + abs(fc_i - hc)
            if d < best_d:
                best_d = d
                best = (fr_i, fc_i)
        return best

    def _heuristic_action(obs_arr: np.ndarray) -> int:
        """
        Choose an action that keeps the snake alive, preserves future space,
        and moves its head toward fruit when possible.

        Heuristic terms (per candidate move):
        - Reject moves that immediately hit walls, body, or barriers.
        - Strong bonus for eating fruit immediately.
        - Bonus for reducing Manhattan distance to nearest fruit.
        - Bonus for having a large reachable free area from the new head position.
        - Mild preference for staying away from walls/corners.
        """
        head = _find_head(obs_arr)
        if head is None:
            return random.randint(0, 3)
        hr, hc = head
        target = _nearest_fruit(obs_arr)
        n = obs_arr.shape[0]

        def _reachable_area(start_r: int, start_c: int) -> int:
            """Approximate how much free space is available from (start_r,start_c)."""
            visited = set()
            q = deque()
            visited.add((start_r, start_c))
            q.append((start_r, start_c))
            count = 0
            while q:
                r0, c0 = q.popleft()
                count += 1
                for dr0, dc0 in action_dirs:
                    rr, cc = r0 + dr0, c0 + dc0
                    if not (0 <= rr < n and 0 <= cc < n):
                        continue
                    if (rr, cc) in visited:
                        continue
                    cell = int(obs_arr[rr, cc])
                    # Treat empty or fruit cells as traversable.
                    if cell in (0, 3):
                        visited.add((rr, cc))
                        q.append((rr, cc))
            return count

        best_a = None
        best_score = -1e9
        for a, (dr, dc) in enumerate(action_dirs):
            nr, nc = hr + dr, hc + dc
            # Check bounds.
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            cell = int(obs_arr[nr, nc])
            # Reject moves that would hit body or barrier.
            if cell == 2 or cell == 4:
                continue

            score = 0.0

            # Strong bonus for eating fruit immediately.
            if cell == 3:
                score += 10.0

            # Bonus for moving closer to nearest fruit.
            if target is not None:
                fr, fc = target
                old_d = abs(fr - hr) + abs(fc - hc)
                new_d = abs(fr - nr) + abs(fc - nc)
                score += 1.0 * (old_d - new_d)  # positive if we get closer

            # Bonus for large reachable free area from new head position.
            area = _reachable_area(nr, nc)
            score += 0.05 * area

            # Slight preference for staying away from walls/corners.
            wall_dist = min(nr, nc, n - 1 - nr, n - 1 - nc)
            score += 0.2 * wall_dist

            # Small per-move penalty to prefer shorter routes.
            score -= 1.0

            # Slight random tie-breaker.
            score += random.uniform(-0.01, 0.01)

            if score > best_score:
                best_score = score
                best_a = a

        if best_a is not None:
            return best_a
        # If everything looked bad, just sample a random action.
        return random.randint(0, 3)

    while (not term) and (not trunc) and steps < max_steps:
        a = _heuristic_action(obs)
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
    n_rollouts: int = 40,
    max_rollout_steps: int = 120,
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

    log_dir = os.path.join(PROJECT_ROOT, "runs_mcts")
    writer = SummaryWriter(log_dir=log_dir) if SummaryWriter else None

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
            if writer is not None:
                # SB3-style tags (so TensorBoard looks familiar across algorithms)
                writer.add_scalar("rollout/ep_rew_mean", avg_ret, ep)
                writer.add_scalar("rollout/ep_len_mean", avg_len, ep)
                writer.add_scalar("train/avg_return", avg_ret, ep)
                writer.add_scalar("train/avg_steps", avg_len, ep)
                writer.add_scalar("train/avg_len", avg_snake_len, ep)
                writer.add_scalar("train/avg_fruit", avg_fruit, ep)
            algo_label = os.environ.get("WISE_SNAKE_ALGO_NAME")
            prefix = f"[{algo_label}] " if algo_label else ""
            print(
                f"{prefix}Episode {ep:5d} | "
                f"avg_return={avg_ret:.2f} | avg_steps={avg_len:.1f} | "
                f"avg_len={avg_snake_len:.1f} | avg_fruit={avg_fruit:.1f}"
            )

    if writer is not None:
        writer.close()
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
        n_rollouts=40,
        max_rollout_steps=120,
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
