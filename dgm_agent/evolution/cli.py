from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from .config import EvolutionConfig, load_config
from .engine import EvolutionEngine
from .frozen_clustering import load_frozen_clusters
from .protocol import validate_protocol
from .repository import safe_relative


def preflight(config: EvolutionConfig, require_secrets: bool) -> Dict[str, Any]:
    config.validate()
    errors: List[str] = []
    try:
        protocol = validate_protocol(config.dev_manifest, config.benchmark_manifest, config.data_lake)
    except Exception as exc:
        protocol = None
        errors.append(f"protocol validation failed: {type(exc).__name__}: {exc}")
    try:
        frozen_clusters = load_frozen_clusters(config.frozen_cluster_file, config.data_lake, expected_k=26)
        frozen_cluster_report = {
            "path": str(config.frozen_cluster_file),
            "clusters": len(frozen_clusters),
            "assigned_files": sum(len(cluster["file_paths"]) for cluster in frozen_clusters),
        }
        if frozen_cluster_report["assigned_files"] != 147:
            errors.append(
                f"frozen clustering must assign all 147 paper-lake files; found {frozen_cluster_report['assigned_files']}"
            )
    except Exception as exc:
        frozen_cluster_report = None
        errors.append(f"frozen clustering validation failed: {type(exc).__name__}: {exc}")
    missing_mutable = [item for item in config.mutable_files if not safe_relative(config.project_root, item).is_file()]
    missing_copy = [item for item in config.copy_paths if not safe_relative(config.project_root, item).exists()]
    missing_frozen = [item for item in config.frozen_files if not safe_relative(config.project_root, item).exists()]
    python_path = shutil.which(config.python) if not Path(config.python).is_absolute() else config.python
    missing_env = [name for name in config.required_env if not os.environ.get(name)]
    report = {
        "protocol": protocol,
        "frozen_clustering": frozen_cluster_report,
        "project_root": str(config.project_root),
        "output_root": str(config.output_root),
        "python": python_path,
        "mutable_files": config.mutable_files,
        "frozen_files_checked": len(config.frozen_files),
        "required_env_present": {name: bool(os.environ.get(name)) for name in config.required_env},
        "errors": errors,
        "warnings": [],
    }
    if not config.project_root.is_dir():
        report["errors"].append(f"project root does not exist: {config.project_root}")
    if not python_path or not Path(python_path).exists():
        report["errors"].append(f"Python executable not found: {config.python}")
    if missing_mutable:
        report["errors"].append(f"missing mutable files: {missing_mutable}")
    if missing_copy:
        report["errors"].append(f"missing candidate copy paths: {missing_copy}")
    if missing_frozen:
        report["warnings"].append(f"configured frozen paths not present: {missing_frozen}")
    if require_secrets and missing_env:
        report["errors"].append(f"missing required environment variables: {missing_env}")
    elif missing_env:
        report["warnings"].append(f"secrets not checked for execution yet: {missing_env}")
    orchestrator = config.project_root / "dgm_agent" / "dacode_orchestrator.py"
    if orchestrator.exists() and "TDGM_MAX_REFLEXION_RETRIES" not in orchestrator.read_text(encoding="utf-8"):
        report["warnings"].append(
            "dacode_orchestrator.py does not honor TDGM_MAX_REFLEXION_RETRIES; apply the documented integration hook"
        )
    report["ok"] = not report["errors"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="TDGM evolution followed by frozen DA-Code 91-task evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", help="unique immutable output directory for this run")
    parser.add_argument("--preflight", action="store_true", help="validate only; never call an API or run tasks")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted immutable run directory")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_root:
        config = replace(config, output_root=Path(args.output_root).expanduser().resolve())
    report = preflight(config, require_secrets=not args.preflight)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["ok"]:
        return 2
    if args.preflight:
        return 0
    summary = EvolutionEngine(config).run(resume=args.resume)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
