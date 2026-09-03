import argparse
from pathlib import Path

from interface.app import run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a Skyjo series against the learned policy."
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("policies/skyjo_ppo_policy.pkl"),
        help="Path to a pickled shared PPO policy (legacy Q-tables are also supported).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.policy)
