"""Command-line entry point for the meta-learning experiment."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .experiment import ExperimentConfig, compare, evolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("generations", 30), ("population", 24), ("train-worlds", 12),
                          ("test-worlds", 40), ("episodes", 18), ("seed", 7)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig(
        generations=args.generations, population=args.population,
        train_worlds=args.train_worlds, test_worlds=args.test_worlds,
        episodes=args.episodes, seed=args.seed,
    )
    genome, history = evolve(config)
    for record in history:
        print(json.dumps(record, sort_keys=True))
    report = {"config": asdict(config), "best_genome": asdict(genome),
              "held_out": compare(genome, config)}
    print(json.dumps(report, sort_keys=True))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
