"""
Run all four training algorithms (tab_q, mcts, ppo, dqn)
simultaneously for side-by-side comparison.

Usage (headless, no windows):
    python train/train_all.py [episodes]

Usage (with rendering, via watch.py):
    python train/train_all.py watch [episodes]

The headless mode runs the training scripts directly (like running each
algorithm on its own). The watch mode internally invokes:
    python train/watch.py tab_q [episodes]
    python train/watch.py mcts [episodes]
    python train/watch.py ppo [episodes]
    python train/watch.py dqn [episodes]
"""

import os
import sys
import subprocess


def _parse_episodes_from(argv, index: int, default: int = 3000) -> int:
    """Parse an optional integer from argv[index], with a fallback default."""
    if len(argv) > index:
        try:
            return int(argv[index])
        except ValueError:
            print("Error: episodes must be an integer.")
            sys.exit(1)
    return default


def _run_all_with_watch(episodes: int) -> int:
    """Spawn one watch.py process per algorithm and wait for all."""
    train_dir = os.path.dirname(os.path.abspath(__file__))
    watch_path = os.path.join(train_dir, "watch.py")

    algos = ["tab_q.py", "mcts.py", "ppo.py", "dqn.py"]

    procs: list[subprocess.Popen] = []
    base_env = os.environ.copy()
    for algo in algos:
        cmd = [sys.executable, watch_path, algo, str(episodes)]
        print(f"Launching: {' '.join(cmd)}")
        env = base_env.copy()
        # Used by training scripts to customize window titles and logging.
        env["WISE_SNAKE_ALGO_NAME"] = algo
        # Signal to training scripts that they are being run via train_all
        # so they can skip eval_across_seeds and write final stats summaries.
        env["WISE_SNAKE_FROM_TRAIN_ALL"] = "1"
        procs.append(subprocess.Popen(cmd, cwd=os.path.dirname(train_dir), env=env))

    exit_code = 0
    for p in procs:
        try:
            code = p.wait()
            if code != 0 and exit_code == 0:
                exit_code = code
        except KeyboardInterrupt:
            # If user hits Ctrl+C in this launcher, terminate children.
            for child in procs:
                child.terminate()
            exit_code = 1
            break

    # After all algorithms finish, read their final stats (written by each script)
    # and print a compact comparison table.
    project_root = os.path.dirname(train_dir)
    stats_dir = os.path.join(project_root, "trained_parameter")
    print("\n=== Final stats (last logging window) ===")
    for algo in algos:
        algo_name = os.path.splitext(algo)[0]
        stats_file = os.path.join(stats_dir, f"{algo_name}_final_stats.txt")
        line = None
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                line = f.readline().strip()
        except OSError:
            line = None
        if line:
            print(f"{algo_name:8s}: {line}")
        else:
            print(f"{algo_name:8s}: (no stats written)")

    return exit_code


def _run_all_headless(episodes: int) -> int:
    """Run all algorithms directly (no rendering) and wait for all."""
    train_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(train_dir)

    algos = ["tab_q.py", "mcts.py", "ppo.py", "dqn.py"]

    procs: list[subprocess.Popen] = []
    base_env = os.environ.copy()
    for algo in algos:
        script_path = os.path.join(train_dir, algo)
        cmd = [sys.executable, script_path, str(episodes)]
        print(f"Launching headless: {' '.join(cmd)}")
        env = base_env.copy()
        env["WISE_SNAKE_ALGO_NAME"] = algo
        env["WISE_SNAKE_FROM_TRAIN_ALL"] = "1"
        procs.append(subprocess.Popen(cmd, cwd=project_root, env=env))

    exit_code = 0
    for p in procs:
        try:
            code = p.wait()
            if code != 0 and exit_code == 0:
                exit_code = code
        except KeyboardInterrupt:
            for child in procs:
                child.terminate()
            exit_code = 1
            break

    # After all algorithms finish, read their final stats (written by each script)
    # and print a compact comparison table.
    stats_dir = os.path.join(project_root, "trained_parameter")
    print("\n=== Final stats (last logging window) ===")
    for algo in algos:
        algo_name = os.path.splitext(algo)[0]
        stats_file = os.path.join(stats_dir, f"{algo_name}_final_stats.txt")
        line = None
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                line = f.readline().strip()
        except OSError:
            line = None
        if line:
            print(f"{algo_name:8s}: {line}")
        else:
            print(f"{algo_name:8s}: (no stats written)")

    return exit_code


def main():
    # No arguments or a single integer => headless mode.
    if len(sys.argv) == 1:
        episodes = 3000
        code = _run_all_headless(episodes)
        sys.exit(code)

    # First argument might be an integer (episodes) or a mode string.
    first = sys.argv[1]
    try:
        episodes = int(first)
        code = _run_all_headless(episodes)
        sys.exit(code)
    except ValueError:
        mode = first.strip().lower()

    if mode in {"watch.py", "watch"}:
        episodes = _parse_episodes_from(sys.argv, 2)
        code = _run_all_with_watch(episodes)
        sys.exit(code)

    print("Error: unsupported mode.")
    print("Usage (headless): python train/train_all.py [episodes]")
    print("Usage (watch)   : python train/train_all.py watch [episodes]")
    sys.exit(1)


if __name__ == "__main__":
    main()

