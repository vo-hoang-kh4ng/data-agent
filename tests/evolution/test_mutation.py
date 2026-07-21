import tempfile
import unittest
from pathlib import Path

from dgm_agent.evolution.mutation import apply_exact_edits, create_candidate_copy, parse_json_object


class MutationTests(unittest.TestCase):
    def test_exact_whitelisted_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "dgm_agent").mkdir()
            target = root / "dgm_agent" / "dacode_orchestrator.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            result = apply_exact_edits(
                root,
                {
                    "target_file": "dgm_agent/dacode_orchestrator.py",
                    "edits": [{"old": "VALUE = 1", "new": "VALUE = 2"}],
                    "rationale": "test",
                    "rule": "change only the bounded value",
                },
                ["dgm_agent/dacode_orchestrator.py"],
                ["eval_official.py"],
                1000,
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 2\n")
            self.assertNotEqual(result.before_sha256, result.after_sha256)

    def test_frozen_and_non_whitelisted_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "eval_official.py").write_text("score = 1\n", encoding="utf-8")
            proposal = {
                "target_file": "eval_official.py",
                "edits": [{"old": "score = 1", "new": "score = 2"}],
            }
            with self.assertRaises(ValueError):
                apply_exact_edits(root, proposal, ["eval_official.py"], ["eval_official.py"], 1000)

    def test_non_unique_old_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "agent.py"
            target.write_text("x = 1\nx = 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                apply_exact_edits(
                    root,
                    {"target_file": "agent.py", "edits": [{"old": "x = 1", "new": "x = 2"}]},
                    ["agent.py"],
                    [],
                    1000,
                )

    def test_candidate_copy_keeps_selected_paths_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, destination = root / "source", root / "candidate"
            (source / "dgm_agent").mkdir(parents=True)
            (source / "dgm_agent" / "agent.py").write_text("x = 1", encoding="utf-8")
            (source / "data").mkdir()
            (source / "data" / "secret.json").write_text("{}", encoding="utf-8")
            create_candidate_copy(source, destination, ["dgm_agent"])
            self.assertTrue((destination / "dgm_agent" / "agent.py").exists())
            self.assertFalse((destination / "data").exists())

    def test_json_fence_parser(self):
        self.assertEqual(parse_json_object("```json\n{\"x\": 1}\n```"), {"x": 1})


if __name__ == "__main__":
    unittest.main()
