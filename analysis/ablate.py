"""Separate the two things a genome changes: inner-reward shaping vs learning hyperparameters.

Each evolved genome is re-tested twice — once with only its reward weights transplanted onto
default settings, once with only its settings — on the exact worlds and seed compare() uses.

Requires analysis/out/sweep.json (run sweep.py first):

    python analysis/ablate.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_discovery.experiment import ExperimentConfig, Genome, make_worlds, run_lifetime

OUT = Path(__file__).resolve().parent / "out"
REWARD_KEYS = ("outcome", "novelty", "progress", "energy")
HYPER_KEYS = ("learning_rate", "discount", "exploration")
VARIANTS = ("baseline", "hyper_only", "reward_only", "evolved")


def main() -> None:
    runs = json.loads((OUT / "sweep.json").read_text())
    rows = []
    for r in runs:
        config = ExperimentConfig(generations=30, population=24, train_worlds=12,
                                  test_worlds=40, episodes=18, seed=r["seed"])
        evolved, base = Genome(**r["best_genome"]), Genome()
        worlds = make_worlds(config.test_worlds, config.width,
                             random.Random(config.seed + 1_000_000))
        eval_seed = config.seed + 2_000_000
        variants = {
            "baseline": base,
            "hyper_only": replace(base, **{k: getattr(evolved, k) for k in HYPER_KEYS}),
            "reward_only": replace(base, **{k: getattr(evolved, k) for k in REWARD_KEYS}),
            "evolved": evolved,
        }
        scores = {n: run_lifetime(g, worlds, config, eval_seed) for n, g in variants.items()}
        rows.append({"seed": r["seed"], "scores": scores})
        print(f"seed {r['seed']:2d}  " +
              "  ".join(f"{n}={scores[n]['success_rate']:.3f}" for n in VARIANTS), flush=True)

    (OUT / "ablation.json").write_text(json.dumps(rows, indent=1))

    print("\n=== mean held-out result across seeds ===")
    for name in VARIANTS:
        print(f"{name:12s} "
              f"success {statistics.mean(row['scores'][name]['success_rate'] for row in rows):.4f}  "
              f"deaths {statistics.mean(row['scores'][name]['deaths'] for row in rows):6.1f}  "
              f"steps {statistics.mean(row['scores'][name]['mean_steps'] for row in rows):.2f}")

    print("\n=== evolved gene values across seeds (default in parens) ===")
    defaults = Genome()
    for key in REWARD_KEYS + HYPER_KEYS:
        vals = [r["best_genome"][key] for r in runs]
        print(f"{key:14s} mean {statistics.mean(vals):+.3f}  median {statistics.median(vals):+.3f}  "
              f"min {min(vals):+.3f}  max {max(vals):+.3f}   (default {getattr(defaults, key):+.2f})")


if __name__ == "__main__":
    main()
