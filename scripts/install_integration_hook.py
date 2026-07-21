#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


REFLEXION_MARKER = "# TDGM_EVOLUTION_REFLEXION_CAP_V1"
CLUSTER_MARKER = "# TDGM_FROZEN_CLUSTER_HOOK_V1"


def write_with_backup(target: Path, patched: str) -> Path:
    backup = target.with_suffix(target.suffix + ".pre_evolution.bak")
    if backup.exists():
        raise SystemExit(f"Backup already exists; refusing to overwrite it: {backup}")
    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")
    return backup


def patch_reflexion(project_root: Path) -> None:
    target = project_root / "dgm_agent" / "dacode_orchestrator.py"
    if not target.is_file():
        raise SystemExit(f"Missing orchestrator: {target}")
    source = target.read_text(encoding="utf-8")
    if REFLEXION_MARKER in source:
        print("Reflexion integration hook already installed.")
        return
    pattern = re.compile(
        r"^(?P<indent>[ \t]*)print\(f?[\"'].*Budget:.*max_retries.*$",
        flags=re.MULTILINE,
    )
    match = pattern.search(source)
    if not match:
        raise SystemExit(
            "Could not find the final Budget/max_retries log line. "
            "No orchestrator file was changed; insert the documented hook manually."
        )
    indent = match.group("indent")
    hook = (
        f"{indent}{REFLEXION_MARKER}\n"
        f"{indent}import os as _tdgm_os\n"
        f"{indent}_tdgm_retry_cap = int(_tdgm_os.environ.get('TDGM_MAX_REFLEXION_RETRIES', '3'))\n"
        f"{indent}max_retries = min(max_retries, 1 + max(0, _tdgm_retry_cap))\n"
    )
    backup = write_with_backup(target, source[: match.start()] + hook + source[match.start() :])
    print(f"Installed Reflexion cap hook. Backup: {backup}")


def patch_frozen_clustering(project_root: Path) -> None:
    target = project_root / "dgm_agent" / "blackboard.py"
    if not target.is_file():
        raise SystemExit(f"Missing blackboard: {target}")
    source = target.read_text(encoding="utf-8")
    if CLUSTER_MARKER in source:
        print("Frozen-clustering integration hook already installed.")
        return
    pattern = re.compile(
        r"^(?P<indent>[ \t]*)clustering_method\s*=\s*os\.environ\.get\([\"']DACODE_CLUSTERING_METHOD[\"'].*$",
        flags=re.MULTILINE,
    )
    match = pattern.search(source)
    if not match:
        raise SystemExit(
            "Could not find DACODE_CLUSTERING_METHOD assignment. "
            "No blackboard file was changed; insert the documented frozen-cluster hook manually."
        )
    indent = match.group("indent")
    hook = (
        f"{indent}{CLUSTER_MARKER}\n"
        f"{indent}_tdgm_frozen_clusters = os.environ.get('TDGM_FROZEN_CLUSTER_FILE', '')\n"
        f"{indent}if _tdgm_frozen_clusters:\n"
        f"{indent}    from dgm_agent.evolution.frozen_clustering import load_frozen_clusters\n"
        f"{indent}    _tdgm_materialized = load_frozen_clusters(\n"
        f"{indent}        Path(_tdgm_frozen_clusters), data_root, expected_k=26\n"
        f"{indent}    )\n"
        f"{indent}    agents = [\n"
        f"{indent}        FileAgent(agent_id=f'file_agent_{{idx:02d}}', cluster_name=item['name'], file_paths=item['file_paths'])\n"
        f"{indent}        for idx, item in enumerate(_tdgm_materialized)\n"
        f"{indent}    ]\n"
        f"{indent}    print(f'  Frozen Hierarchical Clustering: {{len(agents)}} clusters')\n"
        f"{indent}    return agents\n\n"
    )
    backup = write_with_backup(target, source[: match.start()] + hook + source[match.start() :])
    print(f"Installed frozen-clustering hook. Backup: {backup}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the outer-loop Reflexion cap in dacode_orchestrator.py")
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    patch_reflexion(project_root)
    patch_frozen_clustering(project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
