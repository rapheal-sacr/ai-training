# Findings: 40-seed sweep, ablation, and outer-loop controls

Results from running `meta_discovery` across 40 seeds with two controls added. Nothing in
`meta_discovery/` was modified; the scripts in `analysis/` import it unchanged.

Fixed configuration for every run below: `generations 30, population 24, train_worlds 12,
test_worlds 40, episodes 18, width 9, energy_budget 16, elite_fraction 0.25, mutation_scale 0.22`,
seeds 1–40. That is 29,840 genome evaluations and 6,485,760 learning episodes, or about 21 seconds
on one core.

## 1. The evolved learner transfers — on every seed

The README notes the evolved learner is not guaranteed to win for every seed, so one run settles
nothing. Across 40 seeds it won all 40 times.

| held-out success rate | value |
|---|---|
| evolved beats baseline | 40 / 40 seeds |
| mean delta | +0.1227 |
| median delta | +0.1271 |
| range | +0.0597 … +0.2083 |

Deaths fall from 315.5 to 227.2 per run and the mean episode shortens from 7.59 steps to 3.84.

## 2. The gain is the inner reward, not the hyperparameters

A genome changes two things at once. Transplanting each half separately onto the default genome,
evaluated on the identical worlds and seed `compare()` uses:

| variant | success | deaths | steps |
|---|---|---|---|
| baseline genome | 0.5618 | 315.5 | 7.59 |
| + evolved hyperparameters only | 0.5705 | 309.2 | 7.86 |
| + evolved reward weights only | 0.6754 | 233.7 | 4.05 |
| full evolved genome | 0.6845 | 227.2 | 3.84 |

Reward weights alone recover 0.114 of the 0.123 total gain — about 93%. Learning hyperparameters
alone are worth 0.009. **This is the repository's central claim holding up:** a fitness function
that never observes the shaped reward still selects shaped rewards that work.

## 3. What it discovers: dense progress, almost no exploration

Two genes move the same direction in all 40 runs.

| gene | default | mean | median | seeds moved |
|---|---|---|---|---|
| `outcome` | +1.00 | +1.503 | +1.474 | 31/40 above |
| `novelty` | 0.00 | −0.385 | −0.453 | 34/40 below |
| `progress` | 0.00 | **+1.261** | +1.344 | **40/40 above** (min +0.359) |
| `energy` | 0.00 | +0.068 | +0.168 | 25/40 above |
| `learning_rate` | 0.35 | +0.392 | +0.345 | 20/40 above |
| `discount` | 0.90 | +0.604 | +0.664 | 27/40 below |
| `exploration` | 0.25 | **+0.031** | +0.010 | **40/40 below** |

The pairing is coherent. The default learner gets one bit of feedback per episode, at the end, so it
must explore to find anything. Once `progress` supplies gradient at every step, exploration is
mostly a way to walk into traps — so selection drives it to 0.01, the floor `clipped()` allows.

`energy` never settles: its mean of +0.068 spans the full legal range across seeds, which is what an
ignored gene looks like under drift.

## 4. The outer loop is not the mechanism

This one complicates the story. `evolve()` seeds its initial population with
`_mutate(Genome(), rng, 0.8)` — a wide scatter — and the best of those 24 draws is already almost
as good as anything the following 30 generations produce.

| held-out success rate | value |
|---|---|
| best genome of generation 0 | 0.6826 |
| genome after 30 generations | 0.6845 |
| delta | **+0.0019** |
| final beats generation-0 best | 20/40 seeds (3 ties) |
| delta range | −0.0139 … +0.0208 |

Training fitness *does* climb — a mean +2.24 by the last generation, improving on 36/40 seeds. That
gain does not transfer. Selection is fitting the 12 training worlds.

A second control makes the point from the other side. Pure random search, drawing from the same
distribution `evolve()` seeds generation 0 from and given the same 720-genome budget, **matches or
beats the evolved genome on 4 of 5 seeds tried**:

| seed | random search | evolved | generation-0 best |
|---|---|---|---|
| 1 | 0.6125 | 0.6111 | 0.6097 |
| 2 | 0.6625 | 0.6681 | 0.6694 |
| 3 | 0.6694 | 0.6639 | 0.6583 |
| 4 | 0.7181 | 0.7069 | 0.7181 |
| 5 | 0.7875 | 0.7833 | 0.7750 |

None of this falsifies the headline claim: consequence-only fitness really does find inner rewards
that beat the outcome-only default, and finds the same one every time. What it falsifies is the
narrower reading that the *evolutionary* loop is the mechanism. At this problem size, selection is
doing the work of a single well-scattered random draw.

The likely cause is headroom — 12 training worlds, 216 attempts, and a 7-gene space that a broad
first sample already covers. Ways to give the loop something left to find:

- raise `--train-worlds` so training fitness stops being easy to fit
- lower the initial mutation scale (currently hard-coded 0.8 in `evolve()`) so generation 0 starts
  closer to the default genome
- enlarge the genome, so one random sweep no longer covers the space

## Reproducing

```bash
python -m unittest discover -s tests -v     # repo tests
python analysis/sweep.py                    # 40 seeds  -> analysis/out/sweep.json
python analysis/ablate.py                   # reward vs hyperparameter transplants
python analysis/gen0.py                     # generation-0 and random-search controls
```

`sweep.py` must run first; the other two read its output. All three are dependency-free and write
their JSON to `analysis/out/`, which is gitignored.
