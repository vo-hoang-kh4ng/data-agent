import sys
import unittest

from dgm_agent.evolution.strategy_validator import runtime_validate, static_validate


GOOD = """
class EvolutionStrategy:
    def select_parent(self, archive, rng):
        return archive[0]["id"] if archive else None
"""


class StrategyValidatorTests(unittest.TestCase):
    def test_accepts_valid_interface(self):
        self.assertTrue(static_validate(GOOD).valid)
        self.assertTrue(runtime_validate(GOOD, sys.executable, 3.0).valid)

    def test_rejects_forbidden_import(self):
        source = "import os\n" + GOOD
        result = static_validate(source)
        self.assertFalse(result.valid)
        self.assertIn("forbidden import", result.reason)

    def test_rejects_runtime_timeout(self):
        source = """
class EvolutionStrategy:
    def select_parent(self, archive, rng):
        while True:
            pass
"""
        result = runtime_validate(source, sys.executable, 0.2)
        self.assertFalse(result.valid)
        self.assertIn("exceeded", result.reason)


if __name__ == "__main__":
    unittest.main()
