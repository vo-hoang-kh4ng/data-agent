import json
import tempfile
import unittest
from pathlib import Path

from dgm_agent.evolution.protocol import validate_protocol, write_fixed_dev_subsets


def write_manifest(path, prefix, count):
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps({"task_id": f"{prefix}-{index:03d}", "question": "q"}) + "\n")


class ProtocolTests(unittest.TestCase):
    def test_valid_disjoint_protocol_and_reproducible_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dev, bench, lake = root / "dev.jsonl", root / "bench.jsonl", root / "lake"
            write_manifest(dev, "dev", 60)
            write_manifest(bench, "bench", 91)
            lake.mkdir()
            for index in range(147):
                (lake / f"file-{index:03d}.txt").write_text("x", encoding="utf-8")
            report = validate_protocol(dev, bench, lake)
            self.assertEqual(report["benchmark_tasks"], 91)
            paths = write_fixed_dev_subsets(dev, root / "split", 7)
            self.assertEqual(len(paths["stage1"].read_text().splitlines()), 10)
            self.assertEqual(len(paths["stage2"].read_text().splitlines()), 50)

    def test_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dev, bench, lake = root / "dev.jsonl", root / "bench.jsonl", root / "lake"
            write_manifest(dev, "same", 60)
            write_manifest(bench, "same", 91)
            lake.mkdir()
            for index in range(147):
                (lake / str(index)).write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_protocol(dev, bench, lake)


if __name__ == "__main__":
    unittest.main()
