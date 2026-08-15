import random
import unittest

from meta_discovery.experiment import (
    ExperimentConfig, Genome, compare, evolve, make_worlds, run_lifetime,
)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(generations=2, population=5, train_worlds=3,
                                       test_worlds=4, episodes=4, seed=13)

    def test_worlds_and_lifetime_are_reproducible(self):
        worlds = make_worlds(4, 9, random.Random(2))
        first = run_lifetime(Genome(), worlds, self.config, 99)
        self.assertEqual(first, run_lifetime(Genome(), worlds, self.config, 99))
        self.assertGreaterEqual(first["success_rate"], 0)
        self.assertLessEqual(first["success_rate"], 1)

    def test_evolution_and_held_out_comparison(self):
        genome, history = evolve(self.config)
        self.assertEqual(len(history), self.config.generations)
        self.assertIsInstance(genome, Genome)
        result = compare(genome, self.config)
        self.assertEqual(result["test_worlds"], 4)
        self.assertEqual(set(result), {"evolved", "baseline", "test_worlds"})

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            evolve(ExperimentConfig(population=0))


if __name__ == "__main__":
    unittest.main()
