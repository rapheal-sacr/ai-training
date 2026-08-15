"""Does the outer loop beat its own starting population?

Two controls on the same held-out worlds:
  1. the best genome of generation 0 vs the genome after all generations
  2. pure random search given the same total genome-evaluation budget

Requires analysis/out/sweep.json (run sweep.py first):

    python analysis/gen0.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# _mutate is imported deliberately: the control must sample the exact distribution
# evolve() draws its initial population from, not a re-derived approximation of it.
from meta_discovery.experiment import (
    ExperimentConfig, Genome, _mutate, make_worlds, run_lifetime,
)

OUT = Path(__file__).resolve().parent / "out"
CONTROL_SEEDS = (1, 2, 3, 4, 5)
INITIAL_SCALE = 0.8  # the scale evolve() uses to seed generation 0


def config_for(seed: int) -> ExperimentConfig:
    return ExperimentConfig(generations=30, population=24, train_worlds=12,
                            test_worlds=40, episodes=18, seed=seed)


def main() -> None:
    runs = json.loads((OUT / "sweep.json").read_text())
    rows = []
    for r in runs:
        config = config_for(r["seed"])
        worlds = make_worlds(config.test_worlds, config.width,
                             random.Random(config.seed + 1_000_000))
        eval_seed = config.seed + 2_000_000
        rows.append({
            "seed": r["seed"],
            "gen0": run_lifetime(Genome(**r["genome_track"][0]), worlds, config, eval_seed),
            "final": run_lifetime(Genome(**r["best_genome"]), worlds, config, eval_seed),
            "train_gen0": r["train_fitness"][0],
            "train_final": r["train_fitness"][-1],
        })

    (OUT / "gen0.json").write_text(json.dumps(rows, indent=1))

    g0 = [row["gen0"]["success_rate"] for row in rows]
    fn = [row["final"]["success_rate"] for row in rows]
    d = [b - a for a, b in zip(g0, fn)]
    print(f"held-out success  gen0-best {statistics.mean(g0):.4f}   final {statistics.mean(fn):.4f}   "
          f"delta {statistics.mean(d):+.4f}")
    print(f"final beats gen0-best on {sum(1 for x in d if x > 0)}/{len(d)} seeds "
          f"(ties {sum(1 for x in d if x == 0)})")
    print(f"delta range {min(d):+.4f} .. {max(d):+.4f}   median {statistics.median(d):+.4f}")

    tg = [row["train_final"] - row["train_gen0"] for row in rows]
    print(f"\ntrain fitness gain first->last generation: mean {statistics.mean(tg):+.3f}  "
          f"improved on {sum(1 for x in tg if x > 0)}/{len(tg)} seeds")

    print("\n=== control: pure random search, same evaluation budget ===")
    for seed in CONTROL_SEEDS:
        config = config_for(seed)
        rng = random.Random(config.seed)
        train = make_worlds(config.train_worlds, config.width, rng)
        budget = config.generations * config.population
        best = max((_mutate(Genome(), rng, INITIAL_SCALE) for _ in range(budget)),
                   key=lambda g: run_lifetime(g, train, config, config.seed + 10_000)["fitness"])
        worlds = make_worlds(config.test_worlds, config.width,
                             random.Random(config.seed + 1_000_000))
        rs = run_lifetime(best, worlds, config, config.seed + 2_000_000)["success_rate"]
        row = next(x for x in rows if x["seed"] == seed)
        print(f"  seed {seed}: random-search {rs:.4f}   evolved {row['final']['success_rate']:.4f}   "
              f"gen0-best {row['gen0']['success_rate']:.4f}")


if __name__ == "__main__":
    main()
