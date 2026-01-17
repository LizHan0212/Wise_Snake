import time
from snake_env import SnakeEnv, EnvConfig


def main():
    env = SnakeEnv(EnvConfig(grid_size=15, max_steps=500, seed=0), render_mode="human")
    env.reset(seed=0)

    try:
        while True:
            alive = env.render()
            if not alive:
                break

            # Human mode: do NOT move unless a keypress action exists
            if env.get_control_mode() == "human":
                action = env.pop_human_action()
                if action is None:
                    time.sleep(0.01)
                    continue
            else:
                action = env.action_space.sample()

            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                env.reset(seed=0)

    finally:
        env.close()


if __name__ == "__main__":
    main()
