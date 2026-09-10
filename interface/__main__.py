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
        default=Path("policies/skyjo_mappo_actor.pt"),
        help="Path to a versioned recurrent MAPPO actor checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducible seed for cards and stochastic policy actions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.policy, seed=arguments.seed)
