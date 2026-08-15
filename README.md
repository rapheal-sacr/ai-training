# Meta-learning discovery lab

This repository contains a small, dependency-free experiment for testing one of
the ideas in the accompanying AI-training discussion: **can an outer evolutionary
loop discover the internal learning signals that help an agent adapt during its
lifetime?**

The experiment deliberately separates two kinds of signal:

* **Inner reward** trains a tabular Q-learner during its lifetime. A genome chooses
  the weights for outcome, novelty, progress, and energy-preservation signals.
* **Outer fitness** never sees that shaped reward. It measures only consequences:
  targets reached, remaining energy, and deaths across previously unseen worlds.

This separation makes reward hacking visible. Evolution may propose any inner
reward mixture, but only real task outcomes determine which goal-writing
heuristic reproduces. This is a toy model—not evidence of intent, consciousness,
or genuine stakes—but it provides a concrete way to test meta-learned objectives.

## Run it

Python 3.10+ is the only requirement.

```bash
python -m meta_discovery --generations 30 --population 24 --seed 7
```

The command prints one JSON record per generation and a final held-out comparison
between the evolved genome and a fixed, outcome-only baseline. Save a reproducible
report with:

```bash
python -m meta_discovery --generations 30 --population 24 --seed 7 \
  --report experiment.json
```

Useful options:

```text
--generations N   outer-loop evolutionary generations
--population N    genomes evaluated per generation
--train-worlds N  worlds used to select genomes
--test-worlds N   held-out worlds used only in the final comparison
--episodes N      lifetime learning episodes per world
--seed N          controls every source of randomness
```

## What to look for

1. `best_genome` shows the inner signals and learning hyperparameters selected by
   consequence-only fitness.
2. `train.fitness` should improve over generations (small runs remain noisy).
3. The final `held_out` result tests whether evolution found a reusable learning
   strategy rather than memorizing its training worlds.
4. Compare `evolved.success_rate` and `baseline.success_rate`. The experiment is
   allowed to falsify the hypothesis: the evolved learner is not guaranteed to
   win for every seed or budget.

## Test

```bash
python -m unittest discover -s tests -v
```

## Boundaries of the experiment

The grid worlds reset and fitness is designer-selected. Evolution therefore does
not solve irreversibility or remove the outer objective. It tests the narrower,
measurable claim that selection on outcomes can discover useful *inner* rewards
and adaptation rules without directly optimizing those rewards.
