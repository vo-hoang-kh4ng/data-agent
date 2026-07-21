import json
import re
import tempfile
import unittest
from pathlib import Path

from dgm_agent.evolution.config import EvolutionConfig
from dgm_agent.evolution.engine import EvolutionEngine
from dgm_agent.evolution.frozen_clustering import export_cluster_manifest
from dgm_agent.evolution.harness import EvaluationResult


class FakeLLM:
    def complete(self, prompt, system, temperature):
        if "You are the Proposer" in prompt:
            return json.dumps({
                "task": "Improve bounded verifier feedback handling with deterministic context and no evaluator changes.",
                "target_file": "dgm_agent/dacode_orchestrator.py",
                "why": "mock dry run",
            })
        if "Implement the proposed" in prompt:
            match = re.search(r"VALUE = (\d+)", prompt)
            current = int(match.group(1))
            return json.dumps({
                "target_file": "dgm_agent/dacode_orchestrator.py",
                "edits": [{"old": f"VALUE = {current}", "new": f"VALUE = {current + 1}"}],
                "rationale": "increment mock bounded state",
                "rule": "Prefer a unique exact replacement for bounded state changes.",
            })
        if "Improve the parent-selection policy" in prompt:
            source = (
                "class EvolutionStrategy:\n"
                "    def select_parent(self, archive, rng):\n"
                "        return archive[0]['id'] if archive else None\n"
            )
            return json.dumps({"source": source, "rationale": "deterministic mock"})
        raise AssertionError("unexpected prompt")


class FakeHarness:
    def evaluate_subset(self, candidate_root, manifest, output_dir, label, timeout=None):
        return EvaluationResult(
            manifest=str(manifest),
            task_count=10 if "stage1" in str(manifest) else 50,
            average_score=1.0,
            pass_rate=1.0,
            passed=10 if "stage1" in str(manifest) else 50,
            results_file=str(output_dir / "official_results.json"),
            sandbox_dir=str(output_dir / "sandbox"),
            log_file=str(output_dir / "execution.log"),
        )

    def evaluate_benchmark(self, candidate_root, output_dir):
        return {"task_count": 91, "average_score": 0.25, "discovery_f1": 0.4}


class FakeFileAgent:
    def __init__(self, name, paths):
        self.cluster_name = name
        self.file_paths = paths


def write_manifest(path, prefix, count):
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps({"task_id": f"{prefix}-{index}", "question": "q"}) + "\n")


class EngineDryRunTests(unittest.TestCase):
    def test_full_30_cycle_control_flow_without_api_or_benchmark(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "project"
            (project / "dgm_agent").mkdir(parents=True)
            orchestrator = project / "dgm_agent" / "dacode_orchestrator.py"
            orchestrator.write_text(
                "VALUE = 0\n" + "def solve(data):\n    return data\n" * 200,
                encoding="utf-8",
            )
            dev, benchmark = root / "dev.jsonl", root / "benchmark.jsonl"
            write_manifest(dev, "dev", 60)
            write_manifest(benchmark, "bench", 91)
            lake = root / "lake"
            lake.mkdir()
            for index in range(147):
                (lake / f"{index}.txt").write_text("x", encoding="utf-8")
            cluster_agents = []
            files = sorted(lake.glob("*.txt"))
            for cluster_index in range(26):
                assigned = [str(path) for offset, path in enumerate(files) if offset % 26 == cluster_index]
                cluster_agents.append(FakeFileAgent(f"cluster-{cluster_index}", assigned))
            frozen_clusters = root / "frozen_clusters.json"
            export_cluster_manifest(cluster_agents, lake, frozen_clusters, expected_k=26)
            config = EvolutionConfig(
                project_root=project,
                output_root=root / "run",
                dev_manifest=dev,
                benchmark_manifest=benchmark,
                data_lake=lake,
                frozen_cluster_file=frozen_clusters,
                python=str(Path(__import__("sys").executable)),
                model="fake",
                mutable_files=["dgm_agent/dacode_orchestrator.py"],
                frozen_files=[],
                copy_paths=["dgm_agent"],
                runner_command=["fake"],
                official_eval_command=["fake"],
            )
            summary = EvolutionEngine(config, llm=FakeLLM(), harness=FakeHarness()).run()
            self.assertEqual(summary["cycles"], 30)
            self.assertEqual(summary["archive_nodes"], 31)
            self.assertEqual(summary["benchmark_91"]["task_count"], 91)
            archive = json.loads((root / "run" / "evolution_archive.json").read_text(encoding="utf-8"))
            meta_events = [e for e in archive["events"] if e["kind"] == "meta_evolution_accepted"]
            self.assertEqual(len(meta_events), 6)


if __name__ == "__main__":
    unittest.main()
