import unittest

from dgm_agent.evolution.metrics import (
    goldilocks_status,
    normalized_compression_distance,
    runtime_score,
    self_complexity,
    task_solution_epiplexity,
)


class MetricsTests(unittest.TestCase):
    def test_ncd_is_small_for_identical_repetitive_text(self):
        text = "pandas dataframe validation " * 200
        # NCD is close, but not equal, to zero because zlib has finite-size overhead.
        self.assertLess(normalized_compression_distance(text, text), 0.5)

    def test_task_solution_epiplexity_is_scaled_ncd(self):
        x = "load csv and calculate mean" * 20
        y = "import pandas as pd; df = pd.read_csv('x.csv')" * 20
        self.assertAlmostEqual(task_solution_epiplexity(x, y), 2 * normalized_compression_distance(x, y))

    def test_goldilocks_boundaries_are_inclusive(self):
        self.assertEqual(goldilocks_status(0.5), "PASS")
        self.assertEqual(goldilocks_status(2.2), "PASS")
        self.assertEqual(goldilocks_status(0.49), "LOW")
        self.assertEqual(goldilocks_status(2.21), "HIGH")

    def test_runtime_score_matches_paper_equation(self):
        self.assertEqual(runtime_score(0.0), 0.5)
        self.assertEqual(runtime_score(0.4), 0.7)
        self.assertEqual(runtime_score(1.0), 1.0)
        with self.assertRaises(ValueError):
            runtime_score(1.1)

    def test_self_complexity_is_finite(self):
        value = self_complexity("def solve():\n    return 1\n" * 100)
        self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
