"""
Primary execution point for watching training in real time.

Runs a training algorithm with the game window visible. The agent still learns
and retains what it knows between episodes.

Usage:
    python train/watch.py <algorithm> [episodes]
    e.g. python train/watch.py tab_q
         python train/watch.py tab_q 1000000

Default episodes: 3000.
Training is slower when rendering because every step is displayed.
"""

import importlib.util
import sys
import os


def _normalize_module_name(arg: str) -> str:
    """Turn 'tab_q.py' or 'tab_q' into 'tab_q'."""
    base = arg.strip()
    if base.endswith(".py"):
        base = base[:-3]
    return base


def main():
    if len(sys.argv) < 2:
        print("Usage: python train/watch.py <algorithm> [episodes]")
        print("  algorithm: e.g. tab_q, dqn, ppo, mcts")
        print("  episodes: optional, default 3000")
        sys.exit(1)

    try:
        total = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    except ValueError:
        print("Error: episodes must be an integer.")
        sys.exit(1)

    name = _normalize_module_name(sys.argv[1])
    train_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(train_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    module_name = f"train.{name}"
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(f"No module named '{module_name}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        print(f"Error: Could not load algorithm module '{module_name}'.")
        print(f"  {e}")
        print("  Use e.g. tab_q or dqn (with or without .py)")
        sys.exit(1)

    if not hasattr(module, "run_training"):
        print(f"Error: '{module_name}' has no function run_training(render=..., total=...).")
        sys.exit(1)

    print(f"Starting training with rendering: {module_name} (episodes={total})")
    module.run_training(render=True, total=total)


if __name__ == "__main__":
    main()

