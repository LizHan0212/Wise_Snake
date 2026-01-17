import time
from snake_env import SnakeEnv, EnvConfig

def main():
    env = SnakeEnv(EnvConfig(grid_size=15), render_mode="human")
    env.reset()

    try:
        while True:
            alive = env.render()
            if not alive:
                break

            mode = env.get_control_mode()

            if mode == "human":
                action = env.pop_human_action()
                if action is None:
                    time.sleep(0.01)
                    continue
            else:
                action = env.action_space.sample()

            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                env.reset()

    finally:
        env.close()

if __name__ == "__main__":
    main()
