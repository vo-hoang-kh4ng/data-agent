from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class BudgetTier:
    candidates: int
    retries: int


@dataclass(frozen=True)
class EvolutionConfig:
    project_root: Path
    output_root: Path
    dev_manifest: Path
    benchmark_manifest: Path
    data_lake: Path
    frozen_cluster_file: Path
    python: str
    model: str
    cycles: int = 30
    meta_interval: int = 5
    strategy_timeout_seconds: float = 3.0
    candidate_timeout_seconds: int = 7200
    outer_timeout_seconds: int = 86400
    stage1_tasks: int = 10
    stage2_tasks: int = 50
    stage1_gate: float = 0.4
    pass_threshold: float = 0.999999
    epiplexity_min: float = 0.5
    epiplexity_max: float = 2.2
    low_budget: BudgetTier = BudgetTier(1, 1)
    medium_budget: BudgetTier = BudgetTier(3, 2)
    high_budget: BudgetTier = BudgetTier(5, 3)
    mutable_files: List[str] = field(default_factory=list)
    frozen_files: List[str] = field(default_factory=list)
    copy_paths: List[str] = field(default_factory=list)
    runner_command: List[str] = field(default_factory=list)
    official_eval_command: List[str] = field(default_factory=list)
    discovery_eval_command: List[str] = field(default_factory=list)
    benchmark_env: Dict[str, str] = field(default_factory=dict)
    required_env: List[str] = field(default_factory=list)
    seed: int = 20260720
    max_mutable_file_bytes: int = 160_000
    max_rules_in_prompt: int = 8

    def validate(self) -> None:
        errors: List[str] = []
        if self.cycles != 30:
            errors.append("cycles must be 30 for the paper-faithful protocol")
        if self.meta_interval != 5:
            errors.append("meta_interval must be 5 for the paper-faithful protocol")
        if self.stage1_tasks != 10 or self.stage2_tasks != 50:
            errors.append("local evaluation must use 10 Stage-1 and 50 Stage-2 tasks")
        if not (0 <= self.epiplexity_min < self.epiplexity_max):
            errors.append("invalid Epiplexity interval")
        if not self.mutable_files:
            errors.append("mutable_files cannot be empty")
        if not self.runner_command or not self.official_eval_command:
            errors.append("runner_command and official_eval_command are required")
        if errors:
            raise ValueError("; ".join(errors))

    def budget_for(self, task_complexity: float) -> BudgetTier:
        if task_complexity < 0.5:
            return self.low_budget
        if task_complexity <= 1.2:
            return self.medium_budget
        return self.high_budget


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _tier(raw: Dict[str, Any], default: BudgetTier) -> BudgetTier:
    return BudgetTier(int(raw.get("candidates", default.candidates)), int(raw.get("retries", default.retries)))


def load_config(path: str | Path) -> EvolutionConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    cfg = EvolutionConfig(
        project_root=_resolve(base, raw["project_root"]),
        output_root=_resolve(base, raw["output_root"]),
        dev_manifest=_resolve(base, raw["dev_manifest"]),
        benchmark_manifest=_resolve(base, raw["benchmark_manifest"]),
        data_lake=_resolve(base, raw["data_lake"]),
        frozen_cluster_file=_resolve(base, raw["frozen_cluster_file"]),
        python=raw.get("python", "python"),
        model=raw["model"],
        cycles=int(raw.get("cycles", 30)),
        meta_interval=int(raw.get("meta_interval", 5)),
        strategy_timeout_seconds=float(raw.get("strategy_timeout_seconds", 3.0)),
        candidate_timeout_seconds=int(raw.get("candidate_timeout_seconds", 7200)),
        outer_timeout_seconds=int(raw.get("outer_timeout_seconds", 86400)),
        stage1_tasks=int(raw.get("stage1_tasks", 10)),
        stage2_tasks=int(raw.get("stage2_tasks", 50)),
        stage1_gate=float(raw.get("stage1_gate", 0.4)),
        pass_threshold=float(raw.get("pass_threshold", 0.999999)),
        epiplexity_min=float(raw.get("epiplexity_min", 0.5)),
        epiplexity_max=float(raw.get("epiplexity_max", 2.2)),
        low_budget=_tier(raw.get("budgets", {}).get("low", {}), BudgetTier(1, 1)),
        medium_budget=_tier(raw.get("budgets", {}).get("medium", {}), BudgetTier(3, 2)),
        high_budget=_tier(raw.get("budgets", {}).get("high", {}), BudgetTier(5, 3)),
        mutable_files=list(raw.get("mutable_files", [])),
        frozen_files=list(raw.get("frozen_files", [])),
        copy_paths=list(raw.get("copy_paths", [])),
        runner_command=list(raw.get("runner_command", [])),
        official_eval_command=list(raw.get("official_eval_command", [])),
        discovery_eval_command=list(raw.get("discovery_eval_command", [])),
        benchmark_env={str(k): str(v) for k, v in raw.get("benchmark_env", {}).items()},
        required_env=[str(value) for value in raw.get("required_env", [])],
        seed=int(raw.get("seed", 20260720)),
        max_mutable_file_bytes=int(raw.get("max_mutable_file_bytes", 160_000)),
        max_rules_in_prompt=int(raw.get("max_rules_in_prompt", 8)),
    )
    cfg.validate()
    return cfg
