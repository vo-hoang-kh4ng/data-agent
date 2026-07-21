#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dgm_agent"))

from dgm_agent.evolution.frozen_clustering import export_cluster_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hierarchical clustering once and freeze its K=26 mapping")
    parser.add_argument("--data-lake", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--k", type=int, default=26)
    args = parser.parse_args()
    if args.k != 26:
        raise SystemExit("paper protocol requires K=26")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is required only for this offline preprocessing command")
    os.environ.pop("TDGM_FROZEN_CLUSTER_FILE", None)
    os.environ["DACODE_CLUSTERING_METHOD"] = "llm_hierarchical"
    os.environ["DACODE_LLM_CLUSTERING_MODEL"] = args.model
    os.environ["DACODE_LLM_CLUSTERING_K"] = str(args.k)

    from dgm_agent.blackboard import build_file_agents

    data_root = Path(args.data_lake).expanduser().resolve()
    agents = build_file_agents(str(data_root))
    payload = export_cluster_manifest(agents, data_root, Path(args.output).expanduser().resolve(), args.k)
    print(f"Frozen {len(payload['clusters'])} clusters over {payload['assigned_file_count']} assigned files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
