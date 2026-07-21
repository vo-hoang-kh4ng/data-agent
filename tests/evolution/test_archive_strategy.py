import random
import tempfile
import unittest
from pathlib import Path

from dgm_agent.evolution.archive import ArchiveNode, EvolutionArchive
from dgm_agent.evolution.evolution_strategy import EvolutionStrategy as UCBStrategy
from dgm_agent.evolution.evolution_strategy_baseline import EvolutionStrategy as BaselineStrategy


def node(node_id, cycle, score):
    return ArchiveNode(
        id=node_id,
        cycle=cycle,
        parent_id=None,
        candidate_root="/tmp/candidate",
        score=score,
        runtime_score=score,
        epiplexity_score=1.0,
        self_complexity=1.0,
        goldilocks_status="PASS",
        context_node="agent.py",
        task_preview="test",
        mutation_log={},
        stage1={},
        stage2={},
    )


class ArchiveStrategyTests(unittest.TestCase):
    def test_archive_persists_and_selects_best(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "archive.json"
            archive = EvolutionArchive(path, {"seed": 7})
            archive.add_node(node("a", 0, 0.5))
            archive.add_node(node("b", 1, 0.8))
            archive.increment_children("a")
            reopened = EvolutionArchive(path)
            self.assertEqual(reopened.get("a")["children"], 1)
            self.assertEqual(reopened.best()["id"], "b")

    def test_ucb_returns_archive_id(self):
        archive = [
            {"id": "a", "score": 0.5, "children": 0, "cycle": 0},
            {"id": "b", "score": 0.9, "children": 0, "cycle": 1},
        ]
        self.assertIn(UCBStrategy().select_parent(archive, random.Random(1)), {"a", "b"})

    def test_baseline_strategy_is_seeded(self):
        archive = [
            {"id": "a", "score": 0.5, "children": 0, "cycle": 0},
            {"id": "b", "score": 0.9, "children": 2, "cycle": 1},
        ]
        first = BaselineStrategy().select_parent(archive, random.Random(3))
        second = BaselineStrategy().select_parent(archive, random.Random(3))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
