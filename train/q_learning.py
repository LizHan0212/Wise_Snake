import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from environment.snake_env import SnakeEnv, EnvConfig
from utils.seed import seed_everything


# Always resolve paths relative to project root (so PyCharm working dir doesn't matter)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_PATH_DEFAULT = os.path.join(PROJECT_ROOT, "trained_parameter", "q_table.npy")


@dataclass
class QTrainConfig:
    seed: int = 0

    episodes: int = 3000
    max_steps_per_episode: int = 500  # should match EnvConfig.max_steps

    alpha: float = 0.1       # learning rate
    gamma: float = 0.95      # discount factor

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.999  # per-episode decay

    save_path: str = SAVE_PATH_DEFAULT
    log_every: int = 100


# ---------- State Encoding  ----------
# 3 danger bits (straight/left/right)
# 4 direction one-hot (up/down/left/right)
# 4 food direction one-hot (up/down/left/right) toward closest fruit
def _find_head(obs: np.ndarray) -> Tuple[int, int]:
    pos = np.argwhere(obs == 1)
    if pos.size == 0:
        raise RuntimeError("No head found in observation.")
    r, c = pos[0]
    return int(r), int(c)


def _infer_direction_from_obs(obs: np.ndarray) -> Tuple[int, int]:
    hr, hc = _find_head(obs)
    n = obs.shape[0]
    neighbors = [(hr - 1, hc), (hr + 1, hc), (hr, hc - 1), (hr, hc + 1)]
    for nr, nc in neighbors:
        if 0 <= nr < n and 0 <= nc < n and obs[nr, nc] == 2:
            dr = hr - nr
            dc = hc - nc
            return dr, dc
    return 0, 1  # fallback RIGHT


def _turn_left(dr: int, dc: int) -> Tuple[int, int]:
    return -dc, dr


def _turn_right(dr: int, dc: int) -> Tuple[int, int]:
    return dc, -dr


def _is_danger(obs: np.ndarray, r: int, c: int) -> int:
    n = obs.shape[0]
    if not (0 <= r < n and 0 <= c < n):
        return 1  # wall
    v = int(obs[r, c])
    # danger: body (2) or barrier (4)
    return 1 if (v == 2 or v == 4) else 0


def _closest_fruit_direction(obs: np.ndarray) -> Tuple[int, int, int, int]:
    hr, hc = _find_head(obs)
    fruits = np.argwhere(obs == 3)
    if fruits.size == 0:
        return 0, 0, 0, 0

    best = None
    best_d = 10**9
    for fr, fc in fruits:
        fr, fc = int(fr), int(fc)
        d = abs(fr - hr) + abs(fc - hc)
        if d < best_d:
            best_d = d
            best = (fr, fc)

    fr, fc = best
    up = 1 if fr < hr else 0
    down = 1 if fr > hr else 0
    left = 1 if fc < hc else 0
    right = 1 if fc > hc else 0
    return up, down, left, right


def encode_state(obs: np.ndarray) -> int:
    hr, hc = _find_head(obs)
    dr, dc = _infer_direction_from_obs(obs)

    ds = _is_danger(obs, hr + dr, hc + dc)
    ldr, ldc = _turn_left(dr, dc)
    dl = _is_danger(obs, hr + ldr, hc + ldc)
    rdr, rdc = _turn_right(dr, dc)
    drg = _is_danger(obs, hr + rdr, hc + rdc)

    dir_up = 1 if (dr, dc) == (-1, 0) else 0
    dir_down = 1 if (dr, dc) == (1, 0) else 0
    dir_left = 1 if (dr, dc) == (0, -1) else 0
    dir_right = 1 if (dr, dc) == (0, 1) else 0

    food_up, food_down, food_left, food_right = _closest_fruit_direction(obs)

    bits = [
        ds, dl, drg,
        dir_up, dir_down, dir_left, dir_right,
        food_up, food_down, food_left, food_right
    ]

    state = 0
    for b in bits:
        state = (state << 1) | int(b)
    return state


def _epsilon_greedy(Q: np.ndarray, s: int, eps: float) -> int:
    if random.random() < eps:
        return random.randint(0, 3)
    return int(np.argmax(Q[s]))


def train(cfg: QTrainConfig) -> np.ndarray:
    seed_everything(cfg.seed)

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    env = SnakeEnv(
        EnvConfig(grid_size=15, max_steps=cfg.max_steps_per_episode, seed=cfg.seed),
        render_mode=None,
    )

    num_states = 2048
    num_actions = 4
    Q = np.zeros((num_states, num_actions), dtype=np.float32)

    eps = cfg.eps_start
    recent_returns = []
    recent_lengths = []

    for ep in range(1, cfg.episodes + 1):
        # vary episode seed but remain reproducible
        obs, _ = env.reset(seed=cfg.seed + ep)
        s = encode_state(obs)

        ep_return = 0.0
        ep_steps = 0
        terminated = False
        truncated = False

        while (not terminated) and (not truncated) and ep_steps < cfg.max_steps_per_episode:
            a = _epsilon_greedy(Q, s, eps)

            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = encode_state(obs2)

            best_next = float(np.max(Q[s2]))
            td_target = float(r) + cfg.gamma * best_next * (0.0 if terminated else 1.0)
            Q[s, a] = Q[s, a] + cfg.alpha * (td_target - Q[s, a])

            s = s2
            ep_return += float(r)
            ep_steps += 1

        eps = max(cfg.eps_end, eps * cfg.eps_decay)

        recent_returns.append(ep_return)
        recent_lengths.append(ep_steps)
        if len(recent_returns) > cfg.log_every:
            recent_returns.pop(0)
            recent_lengths.pop(0)

        if ep % cfg.log_every == 0:
            avg_ret = sum(recent_returns) / len(recent_returns)
            avg_len = sum(recent_lengths) / len(recent_lengths)
            print(f"Episode {ep:5d} | eps={eps:.3f} | avg_return={avg_ret:.2f} | avg_steps={avg_len:.1f}")

    env.close()

    np.save(cfg.save_path, Q)
    print(f"Saved Q-table to: {cfg.save_path}")
    return Q


def main():
    cfg = QTrainConfig()
    train(cfg)


if __name__ == "__main__":
    main()
