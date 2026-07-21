import tempfile
import unittest
from pathlib import Path

from dgm_agent.evolution.frozen_clustering import export_cluster_manifest, load_frozen_clusters


class Agent:
    def __init__(self, name, files):
        self.cluster_name = name
        self.file_paths = files


class FrozenClusteringTests(unittest.TestCase):
    def test_export_load_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lake = root / "lake"
            lake.mkdir()
            paths = []
            for index in range(26):
                path = lake / f"file-{index}.txt"
                path.write_text(f"content-{index}", encoding="utf-8")
                paths.append(path)
            agents = [Agent(f"cluster-{index}", [str(path)]) for index, path in enumerate(paths)]
            manifest = root / "clusters.json"
            export_cluster_manifest(agents, lake, manifest, expected_k=26)
            loaded = load_frozen_clusters(manifest, lake, expected_k=26)
            self.assertEqual(len(loaded), 26)
            paths[0].write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_frozen_clusters(manifest, lake, expected_k=26)

    def test_duplicate_assignment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lake = root / "lake"
            lake.mkdir()
            shared = lake / "shared.txt"
            shared.write_text("x", encoding="utf-8")
            agents = [Agent(f"cluster-{index}", [str(shared)]) for index in range(26)]
            with self.assertRaisesRegex(ValueError, "more than one cluster"):
                export_cluster_manifest(agents, lake, root / "clusters.json", expected_k=26)


if __name__ == "__main__":
    unittest.main()
