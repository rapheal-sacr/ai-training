"""Run the experiment across many seeds so single-run noise cannot carry a conclusion.

Writes analysis/out/sweep.json. Run from anywhere:

    python analysis/sweep.py
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_discovery.experiment import ExperimentConfig, compare, evolve

OUT = Path(__file__).resolve().parent / "out"
SEEDS = range(1, 41)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    runs = []
    for seed in SEEDS:
        config = ExperimentConfig(generations=30, population=24, train_worlds=12,
                                  test_worlds=40, episodes=18, seed=seed)
        genome, history = evolve(config)
        held_out = compare(genome, config)
        runs.append({
            "seed": seed,
            "best_genome": asdict(genome),
            "held_out": held_out,
            "train_fitness": [rec["train"]["fitness"] for rec in history],
            "train_success": [rec["train"]["success_rate"] for rec in history],
            "train_deaths": [rec["train"]["deaths"] for rec in history],
            # Best genome of every generation, so the outer loop can be audited later.
            "genome_track": [rec["best_genome"] for rec in history],
        })
        ev, bl = held_out["evolved"], held_out["baseline"]
        print(f"seed {seed:2d}  evolved {ev['success_rate']:.3f} vs baseline {bl['success_rate']:.3f}  "
              f"delta {ev['success_rate'] - bl['success_rate']:+.3f}  "
              f"deaths {ev['deaths']:.0f}/{bl['deaths']:.0f}  "
              f"steps {ev['mean_steps']:.2f}/{bl['mean_steps']:.2f}", flush=True)

    (OUT / "sweep.json").write_text(json.dumps(runs, indent=1))

    deltas = sorted(r["held_out"]["evolved"]["success_rate"]
                    - r["held_out"]["baseline"]["success_rate"] for r in runs)
    wins = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if d == 0)
    print(f"\nevolved wins {wins}/{len(runs)}, ties {ties}, losses {len(runs) - wins - ties}")
    print(f"success-rate delta: mean {statistics.mean(deltas):+.4f}  "
          f"median {statistics.median(deltas):+.4f}  "
          f"min {deltas[0]:+.4f}  max {deltas[-1]:+.4f}")
    print(f"wrote {OUT / 'sweep.json'}")


if __name__ == "__main__":
    main()
