import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.snake_env import SnakeEnv, EnvConfig
from utils.seed import seed_everything

LOG_EVERY = 100
MAX_STEPS_PER_EPISODE = 500


class TrainingLogCallback(BaseCallback):
    """Print episode stats in the same format as tab_q/mcts."""

    def __init__(self, log_every: int = LOG_EVERY, max_episodes: int | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.recent_returns: list[float] = []
        self.recent_lengths: list[float] = []
        self.recent_snake_lengths: list[float] = []
        self.recent_fruits: list[float] = []
        self.last_length: float = 3.0
        # When launched via train_all, this will be set to the algorithm program name.
        self.algo_label = os.environ.get("WISE_SNAKE_ALGO_NAME")
        # Final stats snapshot (filled in _print_stats).
        self.last_avg_ret: float | None = None
        self.last_avg_len: float | None = None
        self.last_avg_snake_len: float | None = None
        self.last_avg_fruits: float | None = None
        # Optional path for writing final stats when run via train_all.
        self.stats_path: str | None = None

    def _on_step(self) -> bool:
        # SB3 passes per-env info in "infos" (vectorized) or "info" (non-vectorized).
        infos = self.locals.get("infos")
        if infos is None:
            infos = self.locals.get("info", {})
        info = infos
        if isinstance(infos, (list, tuple)):
            info = infos[0] if infos else {}

        # Track latest snake length every step so we have it when episode ends.
        if isinstance(info, dict) and "length" in info:
            try:
                self.last_length = float(info["length"])
            except (TypeError, ValueError):
                self.last_length = 3.0
        if "episode" not in info:
            return True
        ep_info = info["episode"]
        r, steps = float(ep_info["r"]), float(ep_info["l"])
        snake_len = self.last_length
        fruits = snake_len - 3
        self.episode_count += 1
        self.recent_returns.append(r)
        self.recent_lengths.append(steps)
        self.recent_snake_lengths.append(snake_len)
        self.recent_fruits.append(fruits)
        if len(self.recent_returns) > self.log_every:
            self.recent_returns.pop(0)
            self.recent_lengths.pop(0)
            self.recent_snake_lengths.pop(0)
            self.recent_fruits.pop(0)
        if self.episode_count % self.log_every == 0:
            self._print_stats()

        # If a max episode budget is set (used when rendering via watch/train_all),
        # stop training once we hit it so we don't exceed the requested episodes.
        if self.max_episodes is not None and self.episode_count >= self.max_episodes:
            return False

        return True

    def _print_stats(self) -> None:
        if not self.recent_returns:
            return
        avg_ret = sum(self.recent_returns) / len(self.recent_returns)
        avg_len = sum(self.recent_lengths) / len(self.recent_lengths)
        avg_snake_len = sum(self.recent_snake_lengths) / len(self.recent_snake_lengths)
        avg_fruits = sum(self.recent_fruits) / len(self.recent_fruits)
        self.last_avg_ret = avg_ret
        self.last_avg_len = avg_len
        self.last_avg_snake_len = avg_snake_len
        self.last_avg_fruits = avg_fruits
        prefix = f"[{self.algo_label}] " if self.algo_label else ""
        print(
            f"{prefix}Episode {self.episode_count:5d} | "
            f"avg_return={avg_ret:.2f} | avg_steps={avg_len:.1f} | "
            f"avg_len={avg_snake_len:.1f} | avg_fruit={avg_fruits:.1f}"
        )

    def _on_training_end(self) -> None:
        # If we didn't hit an exact multiple of log_every, still print the last window.
        if self.episode_count > 0 and self.episode_count % self.log_every != 0:
            self._print_stats()
        # If running under train_all and a stats_path is set, write final stats for comparison.
        if os.environ.get("WISE_SNAKE_FROM_TRAIN_ALL") == "1" and self.stats_path and self.last_avg_ret is not None:
            try:
                with open(self.stats_path, "w", encoding="utf-8") as f:
                    f.write(
                        f"avg_return={self.last_avg_ret:.4f} | "
                        f"avg_steps={self.last_avg_len:.4f} | "
                        f"avg_len={self.last_avg_snake_len:.4f} | "
                        f"avg_fruit={self.last_avg_fruits:.4f}\n"
                    )
            except OSError:
                pass


class FlatFloatObsWrapper(gym.ObservationWrapper):
    """
    Converts (N,N) int8 grid into flat float32 vector in [0,1] for an MLP DQN.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space
        assert isinstance(old_space, spaces.Box)
        n = int(np.prod(old_space.shape))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        obs = obs.astype(np.float32) / 4.0
        return obs.reshape(-1)


class RenderAfterStepWrapper(gym.Wrapper):
    """Calls env.render() after every step and reset so the pygame window updates during SB3 training."""

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        still_open = self.env.render()  # only used when make_env(..., render_mode="human")
        if not still_open:
            # Signal quit to the training loop and end the episode.
            info = dict(info) if isinstance(info, dict) else {}
            info["quit"] = True
            return obs, r, True, True, info
        return obs, r, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        still_open = self.env.render()
        if not still_open:
            info = dict(info) if isinstance(info, dict) else {}
            info["quit"] = True
        return obs, info


def make_env(seed: int, render_mode: str | None = None):
    env_cfg = EnvConfig(
        grid_size=15,
        max_steps=MAX_STEPS_PER_EPISODE,
        seed=seed,
        render_fps=120 if render_mode == "human" else 12,
        window_title=os.environ.get("WISE_SNAKE_ALGO_NAME", "Wise Snake"),
    )
    env = SnakeEnv(env_cfg, render_mode=render_mode)
    env = FlatFloatObsWrapper(env)
    env = Monitor(env)  # removes warning + ensures correct episode stats
    if render_mode == "human":
        env = RenderAfterStepWrapper(env)
    return env


def eval_across_seeds(model: DQN, base_seed: int = 1000, n_episodes: int = 30, max_steps: int = 500):
    """
    Runs evaluation episodes with different reset seeds so we don't keep testing the same layout.
    Reports reward + steps.
    """
    rewards = []
    steps_list = []
    env = make_env(seed=0)

    for i in range(n_episodes):
        reset_seed = base_seed + i
        obs, _ = env.reset(seed=reset_seed)

        done = False
        trunc = False
        ep_r = 0.0
        steps = 0

        while (not done) and (not trunc) and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(int(action))
            ep_r += float(r)
            steps += 1

        rewards.append(ep_r)
        steps_list.append(steps)

    env.close()

    rewards = np.array(rewards, dtype=np.float32)
    steps_list = np.array(steps_list, dtype=np.int32)

    print("Eval across seeds")
    print(f"  Episodes: {n_episodes}")
    print(f"  Mean reward: {rewards.mean():.3f} | Std reward: {rewards.std():.3f}")
    print(f"  Mean steps : {steps_list.mean():.2f} | Max steps : {steps_list.max()}")


def main(render_mode: str | None = None, episodes: int = 3000):
    seed = 0
    seed_everything(seed)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(project_root, "trained_parameter")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "dqn_model.zip")

    env = make_env(seed, render_mode=render_mode)
    if render_mode == "human":
        # Ensure the pygame window is created before training starts.
        env.reset()
        env.render()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=0,
        # tensorboard_log=os.path.join(project_root, "runs_dqn"),
        seed=seed,
    )

    # Treat episodes as the primary budget; convert to timesteps for SB3,
    # and also enforce the same episode cap in the callback.
    callback = TrainingLogCallback(log_every=LOG_EVERY, max_episodes=episodes)
    callback.stats_path = os.path.join(save_dir, "dqn_final_stats.txt")
    total_timesteps = episodes * MAX_STEPS_PER_EPISODE
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(model_path)
    print("Saved DQN model to:", model_path)

    # Skip eval across seeds when launched via train_all to avoid clutter.
    if os.environ.get("WISE_SNAKE_FROM_TRAIN_ALL") != "1":
        eval_across_seeds(model, base_seed=1000, n_episodes=30, max_steps=500)

    env.close()


def run_training(render: bool = False, total: int = 3000):
    """
    Run DQN training.
    `total` is interpreted as the number of episodes,
    both for headless runs and when called via watch/train_all.
    """
    main(render_mode="human" if render else None, episodes=total)


if __name__ == "__main__":
    try:
        total = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    except (ValueError, IndexError):
        total = 3000
    # When run directly, interpret CLI argument as episodes for consistency
    # with tab_q and mcts.
    main(episodes=total)

