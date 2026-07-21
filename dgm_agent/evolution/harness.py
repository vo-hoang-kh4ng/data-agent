from __future__ import annotations

import json
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .config import EvolutionConfig


@dataclass(frozen=True)
class EvaluationResult:
    manifest: str
    task_count: int
    average_score: float
    pass_rate: float
    passed: int
    results_file: str
    sandbox_dir: str
    log_file: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class CommandHarness:
    def __init__(self, config: EvolutionConfig):
        self.config = config

    def _format(self, template: List[str], values: Mapping[str, str]) -> List[str]:
        return [part.format(**values) for part in template]

    def _environment(self, candidate_root: Path) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(self.config.benchmark_env)
        existing = env.get("PYTHONPATH", "")
        parts = [str(candidate_root), str(self.config.project_root)]
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env["DACODE_MODEL"] = self.config.model
        env["TDGM_MAX_REFLEXION_RETRIES"] = "3"
        env["TDGM_FROZEN_CLUSTER_FILE"] = str(self.config.frozen_cluster_file)
        return env

    @staticmethod
    def _run(command: List[str], cwd: Path, env: Dict[str, str], log_file: Path, timeout: int) -> None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as log:
            log.write("COMMAND: " + " ".join(command) + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"command failed with exit code {completed.returncode}; see {log_file}")

    @staticmethod
    def _parse_scores(results_path: Path, pass_threshold: float) -> tuple[int, float, int, float]:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        rows = payload.get("results", [])
        scores: List[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw = row.get("total_score", row.get("score"))
            if raw is not None:
                scores.append(float(raw))
        if not scores and "average_score" in payload:
            count = int(payload.get("num_results", 0))
            return count, float(payload["average_score"]), 0, 0.0
        if not scores:
            raise ValueError(f"official evaluator produced no task scores: {results_path}")
        passed = sum(score >= pass_threshold for score in scores)
        return len(scores), statistics.fmean(scores), passed, passed / len(scores)

    def evaluate_subset(
        self,
        candidate_root: Path,
        manifest: Path,
        output_dir: Path,
        label: str,
        timeout: Optional[int] = None,
    ) -> EvaluationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        sandbox = output_dir / "sandbox"
        results = output_dir / "official_results.json"
        log = output_dir / "execution.log"
        values = {
            "python": self.config.python,
            "project_root": str(self.config.project_root),
            "candidate_root": str(candidate_root),
            "manifest": str(manifest),
            "sandbox": str(sandbox),
            "results": str(results),
            "discovery_results": str(output_dir / "discovery_f1.json"),
        }
        env = self._environment(candidate_root)
        limit = timeout or self.config.candidate_timeout_seconds
        if not results.exists():
            self._run(self._format(self.config.runner_command, values), self.config.project_root, env, log, limit)
            self._run(self._format(self.config.official_eval_command, values), self.config.project_root, env, log, limit)
        count, average, passed, pass_rate = self._parse_scores(results, self.config.pass_threshold)
        return EvaluationResult(
            manifest=str(manifest),
            task_count=count,
            average_score=average,
            pass_rate=pass_rate,
            passed=passed,
            results_file=str(results),
            sandbox_dir=str(sandbox),
            log_file=str(log),
        )

    def evaluate_benchmark(self, candidate_root: Path, output_dir: Path) -> Dict[str, Any]:
        result = self.evaluate_subset(
            candidate_root,
            self.config.benchmark_manifest,
            output_dir,
            "benchmark-91",
            timeout=self.config.outer_timeout_seconds,
        )
        discovery_path = output_dir / "discovery_f1.json"
        if self.config.discovery_eval_command and not discovery_path.exists():
            values = {
                "python": self.config.python,
                "project_root": str(self.config.project_root),
                "candidate_root": str(candidate_root),
                "manifest": str(self.config.benchmark_manifest),
                "sandbox": result.sandbox_dir,
                "results": result.results_file,
                "discovery_results": str(discovery_path),
            }
            self._run(
                self._format(self.config.discovery_eval_command, values),
                self.config.project_root,
                self._environment(candidate_root),
                output_dir / "execution.log",
                self.config.outer_timeout_seconds,
            )
        payload = result.to_dict()
        if discovery_path.exists():
            discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
            payload["discovery_f1"] = float(discovery.get("macro", {}).get("f1", 0.0))
            payload["discovery_results_file"] = str(discovery_path)
        return payload
