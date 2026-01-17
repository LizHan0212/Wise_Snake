from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    grid_size: int = 12
    max_steps: int = 500
    seed: int = 0

    render_fps: int = 12

    cell_px: int = 32
    margin_px: int = 2
    hud_px: int = 40

    window_scale: float = 1.5

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 12}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode

        n = self.cfg.grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 3, (n, n), dtype=np.int8)

        self._rng = np.random.default_rng(self.cfg.seed)

        # game state
        self._snake: List[Tuple[int, int]] = []
        self._dir: Tuple[int, int] = (0, 1)
        self._fruit: Tuple[int, int] = (0, 0)
        self._steps: int = 0
        self._terminated: bool = False
        self._total_reward: float = 0.0

        # pygame
        self._pg_inited = False
        self._pg = None
        self._screen = None
        self._clock = None
        self._font = None
        self._quit_requested = False

        # control
        self._control_mode = "auto"     # "auto" | "human"
        self._pending_action: Optional[int] = None

    # ---------------- Gym API ----------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        n = self.cfg.grid_size
        self._steps = 0
        self._terminated = False
        self._quit_requested = False
        self._total_reward = 0.0

        r = n // 2
        c = n // 2
        self._snake = [(r, c), (r, c - 1), (r, c - 2)]
        self._dir = (0, 1)
        self._fruit = self._spawn_fruit()

        return self._make_obs(), {}

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("Episode ended. Call reset().")

        self._steps += 1

        new_dir = self._action_to_dir(action)
        if new_dir[0] == -self._dir[0] and new_dir[1] == -self._dir[1]:
            new_dir = self._dir
        self._dir = new_dir

        head_r, head_c = self._snake[0]
        dr, dc = self._dir
        new_head = (head_r + dr, head_c + dc)

        reward = 0.0

        if not self._in_bounds(new_head) or new_head in self._snake:
            self._terminated = True
            return self._make_obs(), reward, True, False, {}

        self._snake.insert(0, new_head)

        if new_head == self._fruit:
            reward = 1.0
            self._total_reward += reward
            self._fruit = self._spawn_fruit()
        else:
            self._snake.pop()

        truncated = self._steps >= self.cfg.max_steps
        info = {"steps": self._steps, "length": len(self._snake), "total_reward": self._total_reward}
        return self._make_obs(), reward, False, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()
            return not self._quit_requested
        return self._render_ansi()

    def close(self):
        if self._pg_inited and self._pg:
            self._pg.quit()
        self._pg_inited = False

    # ---------------- control helpers ----------------
    def get_control_mode(self):
        return self._control_mode

    def pop_human_action(self):
        a = self._pending_action
        self._pending_action = None
        return a

    # ---------------- helpers ----------------
    def _in_bounds(self, cell):
        r, c = cell
        n = self.cfg.grid_size
        return 0 <= r < n and 0 <= c < n

    def _action_to_dir(self, action):
        return [(-1, 0), (1, 0), (0, -1), (0, 1)][action]

    def _spawn_fruit(self):
        n = self.cfg.grid_size
        occ = set(self._snake)
        while True:
            r = int(self._rng.integers(0, n))
            c = int(self._rng.integers(0, n))
            if (r, c) not in occ:
                return (r, c)

    def _make_obs(self):
        n = self.cfg.grid_size
        g = np.zeros((n, n), dtype=np.int8)
        fr, fc = self._fruit
        g[fr, fc] = 3
        hr, hc = self._snake[0]
        g[hr, hc] = 1
        for r, c in self._snake[1:]:
            g[r, c] = 2
        return g

    def _render_ansi(self):
        g = self._make_obs()
        out = []
        for r in range(self.cfg.grid_size):
            row = []
            for c in range(self.cfg.grid_size):
                v = int(g[r, c])
                row.append("." if v == 0 else "H" if v == 1 else "o" if v == 2 else "F")
            out.append(" ".join(row))
        return "\n".join(out)

    # ---------------- pygame ----------------
    def _init_pygame(self):
        import pygame
        self._pg = pygame
        pygame.init()

        s = self.cfg.window_scale
        cell = int(self.cfg.cell_px * s)
        margin = int(self.cfg.margin_px * s)
        hud = int(self.cfg.hud_px * s)

        n = self.cfg.grid_size
        w = n * (cell + margin) + margin
        h = hud + w

        self._cell = cell
        self._margin = margin
        self._hud = hud

        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Wise Snake")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, int(24 * s))
        self._pg_inited = True

    def _handle_input(self):
        pygame = self._pg
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit_requested = True
                elif event.key == pygame.K_h:
                    self._control_mode = "human" if self._control_mode == "auto" else "auto"
                elif event.key == pygame.K_UP:
                    self._pending_action = 0
                elif event.key == pygame.K_DOWN:
                    self._pending_action = 1
                elif event.key == pygame.K_LEFT:
                    self._pending_action = 2
                elif event.key == pygame.K_RIGHT:
                    self._pending_action = 3
                elif event.key == pygame.K_w:
                    self._pending_action = 0
                elif event.key == pygame.K_s:
                    self._pending_action = 1
                elif event.key == pygame.K_a:
                    self._pending_action = 2
                elif event.key == pygame.K_d:
                    self._pending_action = 3

    def _render_pygame(self):
        if not self._pg_inited:
            self._init_pygame()

        pygame = self._pg
        self._handle_input()
        if self._quit_requested:
            return

        g = self._make_obs()
        n = self.cfg.grid_size

        self._screen.fill((20, 20, 20))

        # HUD
        pygame.draw.rect(self._screen, (15, 15, 15), (0, 0, self._screen.get_width(), self._hud))
        text = (
            f"Mode: {self._control_mode.upper()}   "
            f"Reward: {self._total_reward:.0f}   "
            f"Length: {len(self._snake)}   "
            f"Steps: {self._steps}   "
            f"[H] toggle"
        )
        surf = self._font.render(text, True, (230, 230, 230))
        self._screen.blit(surf, (10, 8))

        # Grid
        y0 = self._hud
        colors = {
            0: (35, 35, 35),
            1: (0, 200, 0),
            2: (0, 120, 0),
            3: (220, 50, 50),
        }
        for r in range(n):
            for c in range(n):
                x = self._margin + c * (self._cell + self._margin)
                y = y0 + self._margin + r * (self._cell + self._margin)
                pygame.draw.rect(
                    self._screen,
                    colors[int(g[r, c])],
                    pygame.Rect(x, y, self._cell, self._cell),
                    border_radius=6,
                )

        pygame.display.flip()
        self._clock.tick(self.cfg.render_fps)
