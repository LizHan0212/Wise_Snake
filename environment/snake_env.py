from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces

@dataclass
class EnvConfig:
    grid_size: int = 15
    max_steps: int = 500

    # If None => random session seed chosen once per program run (new window => new layout)
    # Restart button will reuse the same session seed (same layout within the same run)
    seed: Optional[int] = None

    barrier_count: int = 7
    turn_penalty: float = -0.05

    # fruit counts by reward: 1x(+5), 2x(+3), 4x(+1) => 7 total
    fruit_spec: Tuple[Tuple[int, int], ...] = ((5, 1), (3, 2), (1, 4))

    # rendering
    render_fps: int = 12
    cell_px: int = 32
    margin_px: int = 2
    hud_px: int = 40
    window_scale: float = 1.5


class SnakeEnv(gym.Env):
    """
    Observation (N x N int8):
        0 empty
        1 snake head
        2 snake body
        3 fruit
        4 barrier
    Actions:
        0 up, 1 down, 2 left, 3 right
    Reward:
        +1/+3/+5 for fruit, -0.2 for turning, 0 otherwise
    Termination:
        wall / body / barrier collision
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 12}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode

        n = self.cfg.grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=4, shape=(n, n), dtype=np.int8)

        # New env every time restarts
        if self.cfg.seed is None:
            self._session_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        else:
            self._session_seed = int(self.cfg.seed)

        self._rng = np.random.default_rng(self._session_seed)

        # state
        self._snake: List[Tuple[int, int]] = []
        self._dir: Tuple[int, int] = (0, 1)
        self._fruit: Dict[Tuple[int, int], int] = {}  # pos -> reward (1/3/5)
        self._barriers: set[Tuple[int, int]] = set()

        self._steps = 0
        self._terminated = False
        self._total_reward = 0.0

        # interactive control
        self._control_mode = "auto"
        self._pending_action: Optional[int] = None

        # pygame (lazy)
        self._pg_inited = False
        self._pg = None
        self._screen = None
        self._clock = None
        self._font = None
        self._quit_requested = False

        # cached scaled sizes
        self._cell = None
        self._margin = None
        self._hud = None

        # game-over overlay
        self._game_over = False
        self._restart_button = None
        self._restart_hover = False

    # ---------------- public helpers for runners ----------------
    def get_control_mode(self) -> str:
        return self._control_mode

    def pop_human_action(self) -> Optional[int]:
        a = self._pending_action
        self._pending_action = None
        return a

    def is_game_over(self) -> bool:
        return self._game_over

    def get_session_seed(self) -> int:
        return self._session_seed

    # ---------------- Gym API ----------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        if seed is None:
            seed = self._session_seed

        self._rng = np.random.default_rng(int(seed))

        n = self.cfg.grid_size
        self._steps = 0
        self._terminated = False
        self._quit_requested = False
        self._total_reward = 0.0

        # clear game-over overlay
        self._game_over = False
        self._restart_button = None
        self._restart_hover = False

        # RANDOMIZE start each reset (but deterministic given the seed)
        dir_options = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self._dir = dir_options[int(self._rng.integers(0, 4))]
        dr, dc = self._dir

        while True:
            hr = int(self._rng.integers(0, n))
            hc = int(self._rng.integers(0, n))
            body1 = (hr - dr, hc - dc)
            body2 = (hr - 2 * dr, hc - 2 * dc)
            if self._in_bounds((hr, hc)) and self._in_bounds(body1) and self._in_bounds(body2):
                self._snake = [(hr, hc), body1, body2]
                break

        self._barriers = set()
        self._fruit = {}

        self._spawn_barriers(self.cfg.barrier_count)
        self._spawn_all_fruits()

        return self._make_obs(), {}

    def step(self, action: int):
        # freeze dynamics while game-over overlay is active
        if self._game_over:
            return self._make_obs(), 0.0, True, False, {}

        if self._terminated:
            raise RuntimeError("Episode ended. Call reset() before step().")

        self._steps += 1

        new_dir = self._action_to_dir(action)

        # prevent direct reversal
        if new_dir[0] == -self._dir[0] and new_dir[1] == -self._dir[1]:
            new_dir = self._dir

        reward = 0.0
        if new_dir != self._dir:
            reward += self.cfg.turn_penalty

        # penalize steps where the snake does nothing to deter infinite circles
        reward += -0.05

        self._dir = new_dir

        head_r, head_c = self._snake[0]
        dr, dc = self._dir
        new_head = (head_r + dr, head_c + dc)

        # termination checks
        if (not self._in_bounds(new_head)) or (new_head in self._snake) or (new_head in self._barriers):
            self._terminated = True
            self._game_over = True

            # added penalty for game over so the snake learns to avoid death
            reward -= 3

            self._total_reward += reward
            return self._make_obs(), reward, True, False, {}

        # move
        self._snake.insert(0, new_head)

        # fruit
        if new_head in self._fruit:
            fruit_reward = float(self._fruit.pop(new_head))
            reward += fruit_reward
            self._spawn_one_fruit(int(fruit_reward))  # respawn same type
        else:
            self._snake.pop()

        self._total_reward += reward

        truncated = (self._steps >= self.cfg.max_steps)
        info = {
            "steps": self._steps,
            "length": len(self._snake),
            "total_reward": self._total_reward,
            "fruit_count": len(self._fruit),
            "barrier_count": len(self._barriers),
        }
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
        self._pg = None
        self._screen = None
        self._clock = None
        self._font = None

    # ---------------- mechanics helpers ----------------
    def _action_to_dir(self, action: int) -> Tuple[int, int]:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)][action]

    def _in_bounds(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        n = self.cfg.grid_size
        return 0 <= r < n and 0 <= c < n

    def _occupied_cells(self) -> set[Tuple[int, int]]:
        return set(self._snake) | set(self._barriers) | set(self._fruit.keys())

    def _random_empty_cell(self) -> Tuple[int, int]:
        n = self.cfg.grid_size
        occ = self._occupied_cells()
        while True:
            r = int(self._rng.integers(0, n))
            c = int(self._rng.integers(0, n))
            if (r, c) not in occ:
                return (r, c)

    def _spawn_barriers(self, k: int):
        for _ in range(k):
            self._barriers.add(self._random_empty_cell())

    def _spawn_all_fruits(self):
        for reward, count in self.cfg.fruit_spec:
            for _ in range(count):
                self._spawn_one_fruit(int(reward))

    def _spawn_one_fruit(self, reward: int):
        self._fruit[self._random_empty_cell()] = reward

    def _make_obs(self) -> np.ndarray:
        n = self.cfg.grid_size
        grid = np.zeros((n, n), dtype=np.int8)

        for r, c in self._barriers:
            grid[r, c] = 4

        for (r, c) in self._fruit.keys():
            grid[r, c] = 3

        hr, hc = self._snake[0]
        grid[hr, hc] = 1
        for r, c in self._snake[1:]:
            grid[r, c] = 2

        return grid

    def _render_ansi(self) -> str:
        grid = self._make_obs()
        n = self.cfg.grid_size
        out = []
        for r in range(n):
            row = []
            for c in range(n):
                v = int(grid[r, c])
                row.append("." if v == 0 else "H" if v == 1 else "o" if v == 2 else "F" if v == 3 else "#")
            out.append(" ".join(row))
        return "\n".join(out)

    # ---------------- pygame ----------------
    def _init_pygame(self):
        import pygame
        self._pg = pygame
        pygame.init()

        s = self.cfg.window_scale
        self._cell = int(self.cfg.cell_px * s)
        self._margin = int(self.cfg.margin_px * s)
        self._hud = int(self.cfg.hud_px * s)

        n = self.cfg.grid_size
        w = n * (self._cell + self._margin) + self._margin
        h = self._hud + w

        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Wise Snake")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, int(24 * s))

        self._pg_inited = True

    def _button_rect(self):
        w = self._screen.get_width()
        h = self._screen.get_height()
        bw = int(180 * self.cfg.window_scale)
        bh = int(52 * self.cfg.window_scale)
        x = w // 2 - bw // 2
        y = h // 2 - bh // 2
        return self._pg.Rect(x, y, bw, bh)

    def _handle_input(self):
        pygame = self._pg

        if self._game_over and self._screen is not None:
            mx, my = pygame.mouse.get_pos()
            rect = self._restart_button if self._restart_button is not None else self._button_rect()
            self._restart_hover = rect.collidepoint(mx, my)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit_requested = True
                    return

                if event.key == pygame.K_h:
                    self._control_mode = "human" if self._control_mode == "auto" else "auto"
                    self._pending_action = None
                    return

                if event.key in (pygame.K_UP, pygame.K_w):
                    self._pending_action = 0
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self._pending_action = 1
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    self._pending_action = 2
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self._pending_action = 3

                if self._game_over and event.key == pygame.K_r:
                    # restart using session seed -> same layout within the same run
                    self.reset(seed=None)
                    return

            if self._game_over and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                rect = self._restart_button if self._restart_button is not None else self._button_rect()
                if rect.collidepoint(event.pos):
                    # restart using session seed -> same layout within the same run
                    self.reset(seed=None)
                    return

    def _draw_game_over_overlay(self):
        pygame = self._pg
        w, h = self._screen.get_size()

        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self._screen.blit(overlay, (0, 0))

        title = self._font.render("GAME OVER", True, (255, 80, 80))
        title_rect = title.get_rect(center=(w // 2, h // 2 - int(70 * self.cfg.window_scale)))
        self._screen.blit(title, title_rect)

        rect = self._button_rect()
        self._restart_button = rect

        btn_color = (80, 190, 80) if self._restart_hover else (60, 160, 60)
        pygame.draw.rect(self._screen, btn_color, rect, border_radius=12)

        label = self._font.render("Restart", True, (255, 255, 255))
        label_rect = label.get_rect(center=rect.center)
        self._screen.blit(label, label_rect)

        hint = self._font.render("Click Restart or press R", True, (220, 220, 220))
        hint_rect = hint.get_rect(center=(w // 2, h // 2 + int(70 * self.cfg.window_scale)))
        self._screen.blit(hint, hint_rect)

    def _render_pygame(self):
        if not self._pg_inited:
            self._init_pygame()

        pygame = self._pg
        self._handle_input()
        if self._quit_requested:
            return

        grid = self._make_obs()
        n = self.cfg.grid_size

        self._screen.fill((20, 20, 20))

        pygame.draw.rect(self._screen, (15, 15, 15), (0, 0, self._screen.get_width(), self._hud))
        hud_text = (
            f"Mode: {self._control_mode.upper()}   "
            f"Reward: {self._total_reward:.2f}   "
            f"Length: {len(self._snake)}   "
            f"Steps: {self._steps}   "
            f"[H] toggle"
        )
        surf = self._font.render(hud_text, True, (230, 230, 230))
        self._screen.blit(surf, (10, 8))

        empty = (35, 35, 35)
        head = (0, 200, 0)
        body = (0, 120, 0)
        barrier = (140, 140, 140)
        fruit_color = {1: (220, 50, 50), 3: (240, 200, 50), 5: (60, 140, 255)}

        y0 = self._hud
        for r in range(n):
            for c in range(n):
                v = int(grid[r, c])
                if v == 0:
                    color = empty
                elif v == 1:
                    color = head
                elif v == 2:
                    color = body
                elif v == 4:
                    color = barrier
                else:
                    rew = self._fruit.get((r, c), 1)
                    color = fruit_color.get(rew, (220, 50, 50))

                x = self._margin + c * (self._cell + self._margin)
                y = y0 + self._margin + r * (self._cell + self._margin)

                pygame.draw.rect(
                    self._screen,
                    color,
                    pygame.Rect(x, y, self._cell, self._cell),
                    border_radius=6,
                )

        if self._game_over:
            self._draw_game_over_overlay()

        pygame.display.flip()
        self._clock.tick(self.cfg.render_fps)
