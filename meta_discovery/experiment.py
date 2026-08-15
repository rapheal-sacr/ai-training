"""Evolution of reward-shaping and learning rules in a family of grid worlds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import random
from typing import Iterable


@dataclass(frozen=True)
class Genome:
    """Heritable parameters controlling learning, rather than the final policy."""

    outcome: float = 1.0
    novelty: float = 0.0
    progress: float = 0.0
    energy: float = 0.0
    learning_rate: float = 0.35
    discount: float = 0.9
    exploration: float = 0.25

    def clipped(self) -> "Genome":
        return Genome(
            outcome=max(-1.5, min(2.5, self.outcome)),
            novelty=max(-1.5, min(2.5, self.novelty)),
            progress=max(-1.5, min(2.5, self.progress)),
            energy=max(-1.5, min(2.5, self.energy)),
            learning_rate=max(0.02, min(0.95, self.learning_rate)),
            discount=max(0.0, min(0.995, self.discount)),
            exploration=max(0.01, min(0.9, self.exploration)),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    generations: int = 30
    population: int = 24
    train_worlds: int = 12
    test_worlds: int = 40
    episodes: int = 18
    width: int = 9
    energy_budget: int = 16
    elite_fraction: float = 0.25
    mutation_scale: float = 0.22
    seed: int = 7

    def validate(self) -> None:
        if min(self.generations, self.population, self.train_worlds, self.test_worlds,
               self.episodes) < 1:
            raise ValueError("counts must be positive")
        if self.width < 5 or self.energy_budget < 2:
            raise ValueError("width must be >= 5 and energy_budget >= 2")
        if not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1]")


@dataclass(frozen=True)
class World:
    start: int
    target: int
    traps: frozenset[int]


def make_worlds(count: int, width: int, rng: random.Random) -> list[World]:
    worlds = []
    for _ in range(count):
        start, target = rng.sample(range(width), 2)
        candidates = [p for p in range(width) if p not in (start, target)]
        traps = frozenset(rng.sample(candidates, k=rng.randrange(0, 3)))
        worlds.append(World(start, target, traps))
    return worlds


def _state(position: int, target: int, energy: int) -> tuple[int, int, int]:
    direction = (target > position) - (target < position)
    return position, direction, energy // 4


def run_lifetime(genome: Genome, worlds: Iterable[World], config: ExperimentConfig,
                 seed: int) -> dict[str, float]:
    """Learn from inner rewards; score exclusively from observable consequences."""
    rng = random.Random(seed)
    successes = deaths = energy_left = total_steps = 0
    world_list = list(worlds)
    for world in world_list:
        q: dict[tuple[tuple[int, int, int], int], float] = {}
        visits: dict[int, int] = {}
        for episode in range(config.episodes):
            position, energy = world.start, config.energy_budget
            for step in range(config.energy_budget):
                state = _state(position, world.target, energy)
                epsilon = genome.exploration / (1.0 + episode / 5.0)
                if rng.random() < epsilon:
                    action = rng.choice((-1, 1))
                else:
                    values = [q.get((state, action), 0.0) for action in (-1, 1)]
                    action = (-1, 1)[rng.randrange(2)] if values[0] == values[1] else (-1, 1)[values[1] > values[0]]
                old_distance = abs(world.target - position)
                position = max(0, min(config.width - 1, position + action))
                energy -= 1
                visits[position] = visits.get(position, 0) + 1
                reached = position == world.target
                died = position in world.traps or energy == 0
                progress = old_distance - abs(world.target - position)
                inner_reward = (
                    genome.outcome * (1.0 if reached else -1.0 if died else 0.0)
                    + genome.novelty / visits[position]
                    + genome.progress * progress
                    + genome.energy * (energy / config.energy_budget if reached else 0.0)
                )
                next_state = _state(position, world.target, energy)
                future = 0.0 if reached or died else max(q.get((next_state, a), 0.0) for a in (-1, 1))
                old_q = q.get((state, action), 0.0)
                q[(state, action)] = old_q + genome.learning_rate * (
                    inner_reward + genome.discount * future - old_q
                )
                total_steps += 1
                if reached:
                    successes += 1
                    energy_left += energy
                    break
                if died:
                    deaths += 1
                    break
    attempts = len(world_list) * config.episodes
    success_rate = successes / attempts
    # Fitness is deliberately independent of every shaped inner-reward weight.
    fitness = successes + 0.03 * energy_left - 0.35 * deaths
    return {
        "fitness": round(fitness, 6),
        "success_rate": round(success_rate, 6),
        "deaths": float(deaths),
        "mean_steps": round(total_steps / attempts, 6),
    }


def _mutate(parent: Genome, rng: random.Random, scale: float) -> Genome:
    values = asdict(parent)
    for key in values:
        values[key] += rng.gauss(0.0, scale if key in {"outcome", "novelty", "progress", "energy"} else scale / 2)
    return Genome(**values).clipped()


def evolve(config: ExperimentConfig) -> tuple[Genome, list[dict[str, object]]]:
    """Evolve meta-parameters using fixed common-random-number training worlds."""
    config.validate()
    rng = random.Random(config.seed)
    worlds = make_worlds(config.train_worlds, config.width, rng)
    population = [_mutate(Genome(), rng, 0.8) for _ in range(config.population)]
    history = []
    for generation in range(config.generations):
        scored = [(run_lifetime(g, worlds, config, config.seed + 10_000 + generation), g)
                  for g in population]
        scored.sort(key=lambda item: item[0]["fitness"], reverse=True)
        best_metrics, best = scored[0]
        history.append({"generation": generation, "train": best_metrics, "best_genome": asdict(best)})
        elite_count = max(1, round(config.population * config.elite_fraction))
        elites = [genome for _, genome in scored[:elite_count]]
        population = elites + [
            _mutate(rng.choice(elites), rng, config.mutation_scale)
            for _ in range(config.population - elite_count)
        ]
    final = max(population, key=lambda g: run_lifetime(g, worlds, config, config.seed + 99_999)["fitness"])
    return final, history


def compare(genome: Genome, config: ExperimentConfig) -> dict[str, object]:
    """Evaluate evolved and outcome-only learners on untouched deterministic worlds."""
    worlds = make_worlds(config.test_worlds, config.width, random.Random(config.seed + 1_000_000))
    evaluation_config = replace(config, episodes=config.episodes)
    seed = config.seed + 2_000_000
    return {
        "evolved": run_lifetime(genome, worlds, evaluation_config, seed),
        "baseline": run_lifetime(Genome(), worlds, evaluation_config, seed),
        "test_worlds": config.test_worlds,
    }
