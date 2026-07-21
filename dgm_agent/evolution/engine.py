from __future__ import annotations

import json
import random
import shutil
import traceback
import uuid
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .archive import ArchiveNode, EvolutionArchive, append_jsonl, utc_now
from .config import EvolutionConfig
from .harness import CommandHarness, EvaluationResult
from .frozen_clustering import load_frozen_clusters
from .llm_backend import ExistingRepoLLMBackend, LLMBackend
from .memory import RuleMemory
from .metrics import goldilocks_status, runtime_score, self_complexity, task_solution_epiplexity
from .mutation import MutationResult, apply_exact_edits, create_candidate_copy, parse_json_object
from .protocol import validate_protocol, write_fixed_dev_subsets
from .repository import repository_map, safe_relative, snapshot_hashes
from .strategy_loader import load_strategy
from .strategy_validator import runtime_validate


class FrozenBoundaryViolation(RuntimeError):
    pass


class EvolutionEngine:
    def __init__(
        self,
        config: EvolutionConfig,
        llm: Optional[LLMBackend] = None,
        harness: Optional[CommandHarness] = None,
    ):
        self.config = config
        self.llm = llm or ExistingRepoLLMBackend(config.model)
        self.harness = harness or CommandHarness(config)
        self.rng = random.Random(config.seed)
        self.archive_path = config.output_root / "evolution_archive.json"
        self.attempts_path = config.output_root / "candidate_attempts.jsonl"
        self.control_dir = config.output_root / "control"
        self.candidates_dir = config.output_root / "candidates"
        self.dev_dir = config.output_root / "development_protocol"
        self.strategy_path = self.control_dir / "evolution_strategy.py"
        self.baseline_strategy_path = self.control_dir / "evolution_strategy_baseline.py"
        self.memory = RuleMemory(config.output_root / "rimrule_memory.json")
        self.meta_failures = 0

    def _package_file(self, name: str) -> Path:
        return Path(__file__).resolve().parent / name

    def initialize(self) -> EvolutionArchive:
        if self.config.output_root.exists() and any(self.config.output_root.iterdir()):
            raise FileExistsError(
                f"output_root is not empty: {self.config.output_root}. Use a new run directory; results are immutable."
            )
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        protocol = validate_protocol(
            self.config.dev_manifest, self.config.benchmark_manifest, self.config.data_lake
        )
        frozen_clusters = load_frozen_clusters(
            self.config.frozen_cluster_file, self.config.data_lake, expected_k=26
        )
        protocol["frozen_cluster_file"] = str(self.config.frozen_cluster_file)
        protocol["frozen_cluster_count"] = len(frozen_clusters)
        protocol["frozen_cluster_assigned_files"] = sum(
            len(cluster["file_paths"]) for cluster in frozen_clusters
        )
        if protocol["frozen_cluster_assigned_files"] != 147:
            raise ValueError(
                "frozen cluster manifest must assign all 147 paper-lake files; "
                f"found {protocol['frozen_cluster_assigned_files']}"
            )
        protocol["seed"] = self.config.seed
        protocol["model"] = self.config.model
        protocol["cycles"] = self.config.cycles
        protocol["epiplexity_interval"] = [self.config.epiplexity_min, self.config.epiplexity_max]
        protocol["started_at"] = utc_now()
        (self.config.output_root / "protocol_audit.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        subsets = write_fixed_dev_subsets(
            self.config.dev_manifest,
            self.dev_dir,
            self.config.seed,
            self.config.stage1_tasks,
            self.config.stage2_tasks,
        )
        self.control_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._package_file("evolution_strategy.py"), self.strategy_path)
        shutil.copy2(self._package_file("evolution_strategy_baseline.py"), self.baseline_strategy_path)

        archive = EvolutionArchive(self.archive_path, run_metadata=protocol)
        seed_root = self.candidates_dir / "seed"
        create_candidate_copy(self.config.project_root, seed_root, self.config.copy_paths)
        stage1 = self.harness.evaluate_subset(seed_root, subsets["stage1"], self.dev_dir / "seed_stage1", "seed-stage1")
        stage2: Dict[str, Any] = {"skipped": True, "reason": "stage1 gate not met"}
        if stage1.pass_rate >= self.config.stage1_gate:
            stage2 = self.harness.evaluate_subset(
                seed_root, subsets["stage2"], self.dev_dir / "seed_stage2", "seed-stage2"
            ).to_dict()
        seed_node = ArchiveNode(
            id="seed",
            cycle=0,
            parent_id=None,
            candidate_root=str(seed_root),
            score=runtime_score(stage1.pass_rate),
            runtime_score=runtime_score(stage1.pass_rate),
            epiplexity_score=1.0,
            self_complexity=0.0,
            goldilocks_status="PASS",
            context_node="baseline",
            task_preview="Immutable baseline snapshot evaluated on the fixed development split.",
            mutation_log={"type": "seed", "source_project": str(self.config.project_root)},
            stage1=stage1.to_dict(),
            stage2=stage2,
        )
        archive.add_node(seed_node)
        archive.add_event("initialized", {"seed_node": "seed"})
        return archive

    def _propose_task(self, parent: Dict[str, Any]) -> Dict[str, Any]:
        repo_map = repository_map(Path(parent["candidate_root"]), self.config.mutable_files)
        prompt = f"""You are the Proposer in a Darwin-Goedel-style agent evolution loop.
Propose ONE bounded, testable improvement to the DA-Code agent. Do not modify evaluators,
datasets, scoring, API clients, or the clustering protocol. Prefer a small orchestration,
planning, solver, verification, or repair improvement.

Parent summary:
{json.dumps({k: parent.get(k) for k in ['id', 'cycle', 'score', 'stage1', 'stage2']}, ensure_ascii=False)[:8000]}

Mutable repository surface:
{repo_map}

Return JSON only:
{{"task": "precise mutation objective and acceptance rationale",
  "target_file": "one exact path from the mutable surface",
  "why": "short expected benefit"}}
"""
        value = parse_json_object(self.llm.complete(prompt, "You propose safe repository-grounded agent improvements.", 0.7))
        target = str(value.get("target_file", ""))
        if target not in self.config.mutable_files:
            raise ValueError(f"Proposer selected non-mutable target: {target}")
        task = str(value.get("task", "")).strip()
        if len(task) < 40:
            raise ValueError("Proposer task is too short to be auditable")
        return {"task": task, "target_file": target, "why": str(value.get("why", ""))}

    def _propose_task_with_retries(
        self, archive: EvolutionArchive, parent: Dict[str, Any], cycle: int
    ) -> Dict[str, Any]:
        errors: List[str] = []
        for attempt in range(1, 4):
            try:
                return self._propose_task(parent)
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                archive.add_event(
                    "proposer_retry",
                    {"cycle": cycle, "attempt": attempt, "error": errors[-1]},
                )
        raise RuntimeError("Proposer failed three times: " + " | ".join(errors))

    def _mutation_prompt(
        self,
        parent_root: Path,
        proposal: Dict[str, Any],
        previous_error: str,
    ) -> str:
        target = safe_relative(parent_root, proposal["target_file"])
        content = target.read_text(encoding="utf-8")
        if len(content.encode("utf-8")) > self.config.max_mutable_file_bytes:
            raise ValueError(f"target exceeds max_mutable_file_bytes: {target}")
        return f"""Implement the proposed DA-Code agent mutation using minimal exact replacements.
Never change benchmark semantics, evaluator calls, dataset paths, clustering K, model identity,
or score computation. Preserve public interfaces unless the task explicitly requires a compatible extension.

Mutation objective:
{proposal['task']}

Target file: {proposal['target_file']}

Reusable repair rules:
{self.memory.render(self.config.max_rules_in_prompt)}

Previous failed attempt feedback:
{previous_error[-5000:] if previous_error else '(none)'}

Current file content:
<FILE>
{content}
</FILE>

Return JSON only. Each old string must appear exactly once in the current file.
{{"target_file": "{proposal['target_file']}",
  "edits": [{{"old": "exact existing text", "new": "replacement text"}}],
  "rationale": "what changed and why",
  "rule": "concise reusable correction rule learned from this mutation"}}
Use 1-8 edits and keep the change narrow.
"""

    def _record_attempt(self, record: Dict[str, Any]) -> None:
        record.setdefault("time", utc_now())
        append_jsonl(self.attempts_path, record)

    def _evaluate_candidate(
        self,
        cycle: int,
        candidate_index: int,
        retry: int,
        parent: Dict[str, Any],
        proposal: Dict[str, Any],
        previous_error: str,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        attempt_id = f"c{cycle:02d}-n{candidate_index:02d}-r{retry:02d}-{uuid.uuid4().hex[:8]}"
        candidate_root = self.candidates_dir / attempt_id / "source"
        attempt_dir = self.candidates_dir / attempt_id
        record: Dict[str, Any] = {
            "attempt_id": attempt_id,
            "cycle": cycle,
            "candidate_index": candidate_index,
            "retry": retry,
            "parent_id": parent["id"],
            "proposal": proposal,
            "status": "started",
        }
        base_frozen_before: Optional[Dict[str, str]] = None
        try:
            create_candidate_copy(Path(parent["candidate_root"]), candidate_root, self.config.copy_paths)
            frozen_before = snapshot_hashes(candidate_root, self.config.frozen_files)
            base_protected_paths = list(dict.fromkeys(self.config.copy_paths + self.config.frozen_files))
            base_frozen_before = snapshot_hashes(self.config.project_root, base_protected_paths)
            raw = self.llm.complete(
                self._mutation_prompt(Path(parent["candidate_root"]), proposal, previous_error),
                "You are the Solver. Produce a minimal, valid, safe source-code mutation as JSON.",
                0.65,
            )
            mutation_data = parse_json_object(raw)
            mutation = apply_exact_edits(
                candidate_root,
                mutation_data,
                self.config.mutable_files,
                self.config.frozen_files,
                self.config.max_mutable_file_bytes,
            )
            changed_code = safe_relative(candidate_root, mutation.target_file).read_text(encoding="utf-8")
            epi_self = self_complexity(changed_code)
            epi_task = task_solution_epiplexity(proposal["task"], changed_code)
            gold = goldilocks_status(epi_task, self.config.epiplexity_min, self.config.epiplexity_max)
            record.update({
                "mutation": asdict(mutation),
                "self_complexity": epi_self,
                "epiplexity_score": epi_task,
                "goldilocks_status": gold,
            })
            if gold != "PASS":
                raise ValueError(f"Goldilocks rejection: {gold} ({epi_task:.6f})")

            subsets = {
                "stage1": self.dev_dir / "stage1_manifest.jsonl",
                "stage2": self.dev_dir / "stage2_manifest.jsonl",
            }
            stage1 = self.harness.evaluate_subset(
                candidate_root, subsets["stage1"], attempt_dir / "stage1", f"{attempt_id}-stage1"
            )
            record["stage1"] = stage1.to_dict()
            if stage1.pass_rate < self.config.stage1_gate:
                raise ValueError(
                    f"Stage-1 gate failed: {stage1.pass_rate:.4f} < {self.config.stage1_gate:.4f}"
                )
            stage2 = self.harness.evaluate_subset(
                candidate_root, subsets["stage2"], attempt_dir / "stage2", f"{attempt_id}-stage2"
            )
            record["stage2"] = stage2.to_dict()
            frozen_after = snapshot_hashes(candidate_root, self.config.frozen_files)
            if frozen_before != frozen_after:
                raise RuntimeError("frozen-file hash changed during candidate execution")
            base_frozen_after = snapshot_hashes(self.config.project_root, base_protected_paths)
            if base_frozen_before != base_frozen_after:
                raise FrozenBoundaryViolation("base-project frozen-file hash changed during candidate execution")

            score = runtime_score(stage1.pass_rate)
            record.update({"status": "accepted_candidate", "score": score, "runtime_score": score})
            self._record_attempt(record)
            accepted = {
                "attempt_id": attempt_id,
                "candidate_root": str(candidate_root),
                "score": score,
                "runtime_score": score,
                "epiplexity_score": epi_task,
                "self_complexity": epi_self,
                "goldilocks_status": gold,
                "mutation": asdict(mutation),
                "stage1": stage1.to_dict(),
                "stage2": stage2.to_dict(),
            }
            return accepted, ""
        except FrozenBoundaryViolation as exc:
            record.update({"status": "fatal_frozen_boundary_violation", "error": str(exc)})
            self._record_attempt(record)
            raise
        except Exception as exc:
            if base_frozen_before is not None:
                base_protected_paths = list(dict.fromkeys(self.config.copy_paths + self.config.frozen_files))
                current_base_hashes = snapshot_hashes(self.config.project_root, base_protected_paths)
                if base_frozen_before != current_base_hashes:
                    record.update({
                        "status": "fatal_frozen_boundary_violation",
                        "error": "base-project frozen-file hash changed during a rejected attempt",
                    })
                    self._record_attempt(record)
                    raise FrozenBoundaryViolation(record["error"]) from exc
            error = f"{type(exc).__name__}: {exc}"
            record.update({"status": "rejected", "error": error, "traceback": traceback.format_exc()[-8000:]})
            self._record_attempt(record)
            return None, error

    @staticmethod
    def _candidate_key(candidate: Dict[str, Any]) -> tuple[float, float, float, str]:
        stage2 = candidate.get("stage2", {})
        return (
            float(candidate["score"]),
            float(stage2.get("average_score", 0.0)),
            -abs(float(candidate["epiplexity_score"]) - 1.35),
            candidate["attempt_id"],
        )

    def _meta_evolve(self, archive: EvolutionArchive, cycle: int) -> None:
        current = self.strategy_path.read_text(encoding="utf-8")
        history_path: Optional[Path] = None
        source_sha256: Optional[str] = None
        summary = [
            {"id": n["id"], "cycle": n["cycle"], "score": n["score"], "children": n.get("children", 0)}
            for n in archive.nodes[-30:]
        ]
        prompt = f"""Improve the parent-selection policy for this bounded evolutionary archive.
Return JSON only: {{"source": "complete Python module", "rationale": "short explanation"}}.
The source must define class EvolutionStrategy with method
select_parent(self, archive, rng) returning an archive node id or None.
Allowed imports are limited to safe pure-computation modules such as math, random, statistics,
typing, and collections. Do not use filesystem, process, network, reflection, eval, exec, or open.

Current strategy:
<STRATEGY>
{current}
</STRATEGY>

Recent archive:
{json.dumps(summary, ensure_ascii=False)}
"""
        try:
            value = parse_json_object(
                self.llm.complete(prompt, "You safely meta-evolve a bounded selection policy.", 0.55)
            )
            source = str(value.get("source", ""))
            history_dir = self.control_dir / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            history_path = history_dir / f"cycle_{cycle:02d}_candidate.py"
            history_path.write_text(source, encoding="utf-8")
            source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
            validation = runtime_validate(source, self.config.python, self.config.strategy_timeout_seconds)
            if not validation.valid:
                raise ValueError(validation.reason)
            self.strategy_path.write_text(source, encoding="utf-8")
            self.meta_failures = 0
            archive.add_event(
                "meta_evolution_accepted",
                {
                    "cycle": cycle,
                    "rationale": value.get("rationale", ""),
                    "candidate_source": str(history_path),
                    "source_sha256": source_sha256,
                },
            )
        except Exception as exc:
            self.meta_failures += 1
            archive.add_event(
                "meta_evolution_rejected",
                {
                    "cycle": cycle,
                    "error": f"{type(exc).__name__}: {exc}",
                    "consecutive_failures": self.meta_failures,
                    "candidate_source": str(history_path) if history_path else None,
                    "source_sha256": source_sha256,
                },
            )
            if self.meta_failures >= 3:
                shutil.copy2(self.baseline_strategy_path, self.strategy_path)
                archive.add_event("strategy_circuit_breaker", {"cycle": cycle, "action": "baseline restored"})
                self.meta_failures = 0

    def _resume_archive(self) -> EvolutionArchive:
        if not self.archive_path.is_file():
            raise FileNotFoundError(f"cannot resume without archive: {self.archive_path}")
        current = validate_protocol(
            self.config.dev_manifest, self.config.benchmark_manifest, self.config.data_lake
        )
        resumed_clusters = load_frozen_clusters(
            self.config.frozen_cluster_file, self.config.data_lake, expected_k=26
        )
        if sum(len(cluster["file_paths"]) for cluster in resumed_clusters) != 147:
            raise RuntimeError("resume refused because the frozen cluster manifest does not assign 147 files")
        audit = json.loads((self.config.output_root / "protocol_audit.json").read_text(encoding="utf-8"))
        for key in ("dev_manifest_sha256", "benchmark_manifest_sha256"):
            if current[key] != audit.get(key):
                raise RuntimeError(f"resume refused because {key} changed")
        if not self.strategy_path.is_file() or not self.baseline_strategy_path.is_file():
            raise FileNotFoundError("resume control strategy files are missing")
        archive = EvolutionArchive(self.archive_path)
        failures = 0
        for event in archive.data.get("events", []):
            kind = event.get("kind")
            if kind == "meta_evolution_rejected":
                failures += 1
            elif kind in {"meta_evolution_accepted", "strategy_circuit_breaker"}:
                failures = 0
        self.meta_failures = failures
        return archive

    @staticmethod
    def _event_cycles(archive: EvolutionArchive, kinds: set[str]) -> set[int]:
        values = set()
        for event in archive.data.get("events", []):
            if event.get("kind") in kinds and "cycle" in event.get("payload", {}):
                values.add(int(event["payload"]["cycle"]))
        return values

    def run(self, resume: bool = False) -> Dict[str, Any]:
        if resume and (self.config.output_root / "run_summary.json").is_file():
            return json.loads((self.config.output_root / "run_summary.json").read_text(encoding="utf-8"))
        archive = self._resume_archive() if resume else self.initialize()
        terminal_cycles = self._event_cycles(archive, {"cycle_archived", "cycle_no_admission"})
        completed_cycles = self._event_cycles(archive, {"cycle_completed"})
        meta_cycles = self._event_cycles(archive, {"meta_evolution_accepted", "meta_evolution_rejected"})
        for cycle in range(1, self.config.cycles + 1):
            if cycle in completed_cycles:
                continue
            if cycle in terminal_cycles:
                if cycle % self.config.meta_interval == 0 and cycle not in meta_cycles:
                    self._meta_evolve(archive, cycle)
                archive.add_event("cycle_completed", {"cycle": cycle, "resumed_after_terminal_event": True})
                continue
            strategy = load_strategy(self.strategy_path)
            parent_id = strategy.select_parent(archive.nodes, self.rng)
            if parent_id is None:
                raise RuntimeError("selection strategy returned no parent for a non-empty archive")
            archive.increment_children(str(parent_id))
            parent = archive.get(str(parent_id))
            self.rng = random.Random(self.config.seed + cycle)
            proposal = self._propose_task_with_retries(archive, parent, cycle)
            complexity = self_complexity(proposal["task"])
            budget = self.config.budget_for(complexity)
            archive.add_event(
                "cycle_started",
                {
                    "cycle": cycle,
                    "parent_id": parent_id,
                    "task_complexity": complexity,
                    "candidates": budget.candidates,
                    "retries": budget.retries,
                    "proposal": proposal,
                },
            )
            accepted: List[Dict[str, Any]] = []
            for candidate_index in range(1, budget.candidates + 1):
                previous_error = ""
                for retry in range(1, budget.retries + 1):
                    candidate, error = self._evaluate_candidate(
                        cycle, candidate_index, retry, parent, proposal, previous_error
                    )
                    if candidate is not None:
                        accepted.append(candidate)
                        if previous_error and candidate["mutation"].get("rule"):
                            self.memory.add(candidate["mutation"]["rule"], previous_error, cycle)
                        break
                    previous_error = error

            if accepted:
                winner = max(accepted, key=self._candidate_key)
                node = ArchiveNode(
                    id=winner["attempt_id"],
                    cycle=cycle,
                    parent_id=str(parent_id),
                    candidate_root=winner["candidate_root"],
                    score=winner["score"],
                    runtime_score=winner["runtime_score"],
                    epiplexity_score=winner["epiplexity_score"],
                    self_complexity=winner["self_complexity"],
                    goldilocks_status=winner["goldilocks_status"],
                    context_node=proposal["target_file"],
                    task_preview=proposal["task"][:1000],
                    mutation_log=winner["mutation"],
                    stage1=winner["stage1"],
                    stage2=winner["stage2"],
                )
                archive.add_node(node)
                archive.add_event("cycle_archived", {"cycle": cycle, "node_id": node.id, "score": node.score})
            else:
                archive.add_event("cycle_no_admission", {"cycle": cycle, "parent_id": parent_id})

            if cycle % self.config.meta_interval == 0:
                self._meta_evolve(archive, cycle)
            archive.add_event("cycle_completed", {"cycle": cycle})

        best = archive.best()
        archive.add_event("outer_evaluation_started", {"best_node": best["id"]})
        frozen_agent_root = self.config.output_root / "frozen_agent"
        if not frozen_agent_root.exists():
            create_candidate_copy(Path(best["candidate_root"]), frozen_agent_root, self.config.copy_paths)
            (self.config.output_root / "selected_node.json").write_text(
                json.dumps(best, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
            )
        frozen_source_before = snapshot_hashes(frozen_agent_root, self.config.copy_paths)
        base_protected_paths = list(dict.fromkeys(self.config.copy_paths + self.config.frozen_files))
        base_source_before = snapshot_hashes(self.config.project_root, base_protected_paths)
        benchmark = self.harness.evaluate_benchmark(
            frozen_agent_root, self.config.output_root / "benchmark_91"
        )
        if frozen_source_before != snapshot_hashes(frozen_agent_root, self.config.copy_paths):
            raise FrozenBoundaryViolation("frozen selected agent changed during the 91-task evaluation")
        if base_source_before != snapshot_hashes(self.config.project_root, base_protected_paths):
            raise FrozenBoundaryViolation("base project changed during the 91-task evaluation")
        summary = {
            "completed_at": utc_now(),
            "best_node": best,
            "benchmark_91": benchmark,
            "archive_nodes": len(archive.nodes),
            "cycles": self.config.cycles,
            "attempts_file": str(self.attempts_path),
            "archive_file": str(self.archive_path),
        }
        (self.config.output_root / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        archive.add_event("completed", {"best_node": best["id"], "benchmark": benchmark})
        return summary
