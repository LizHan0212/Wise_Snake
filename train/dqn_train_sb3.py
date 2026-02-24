import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from environment.snake_env import SnakeEnv, EnvConfig
from utils.seed import seed_everything


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


def make_env(seed: int):
    env = SnakeEnv(
        EnvConfig(grid_size=15, max_steps=500, seed=seed),
        render_mode=None,
    )
    env = FlatFloatObsWrapper(env)
    env = Monitor(env)  # removes warning + ensures correct episode stats
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


def main():
    seed = 0
    seed_everything(seed)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(project_root, "trained_parameter")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "dqn_model.zip")

    env = make_env(seed)

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
        verbose=1,
        tensorboard_log=os.path.join(project_root, "runs_dqn"),
        seed=seed,
    )

    total_timesteps = 300000  #can be changed as needed
    model.learn(total_timesteps=total_timesteps)

    model.save(model_path)
    print("Saved DQN model to:", model_path)

    eval_across_seeds(model, base_seed=1000, n_episodes=30, max_steps=500)

    env.close()


if __name__ == "__main__":
    main()
