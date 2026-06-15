"""
Diagnostic: dump the EXACT context the solver receives for a task, in BOTH
global and per-subtask lake settings, to find why per-subtask triggers
filename hallucination (data.csv) where global uses the real filename.

Runs ONLY the blackboard discovery (E5 + FileAgents) — no LLM calls.
This is the ground truth of what the planner/solver/fallback actually see.

Usage:
    python scripts/debug_persubtask_context.py di-csv-006
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dgm_agent.dacode_orchestrator import DACodeOrchestrator


def load_task(manifest, task_id):
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["task_id"] == task_id:
                return e
    raise SystemExit(f"task {task_id} not in {manifest}")


def dump(task, label, sandbox_dir):
    print("\n" + "=" * 70)
    print(f"  CONTEXT DUMP — {label}  (task={task['task_id']})")
    print(f"  data_lake_dir = {task.get('data_lake_dir')}")
    print("=" * 70)

    orch = DACodeOrchestrator(
        model=os.environ.get("DACODE_MODEL", "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8"),
        sandbox_dir=sandbox_dir,
    )
    bb = orch._discover_files(task)

    print(f"\n>>> bb.responses: {len(bb.responses)}")
    for r in bb.responses:
        print(f"   cluster={r.cluster_name!r} relevance={r.relevance_score:.3f} "
              f"n_files={len(r.file_contexts)}")
        for fc in r.file_contexts:
            print(f"      file_path={fc.file_path!r}  type={fc.file_type}  "
                  f"err={fc.error!r}")
            if fc.columns:
                print(f"         columns={fc.columns}")

    print("\n>>> _build_compact_context (fed to planner.plan):")
    print("-" * 70)
    print(orch._build_compact_context(bb))
    print("-" * 70)

    print("\n>>> bb.get_context_summary() (fed to FALLBACK prompt):")
    print("-" * 70)
    print(bb.get_context_summary())
    print("-" * 70)


def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else "di-csv-006"

    persubtask = load_task(
        os.path.join(PROJECT_ROOT, "data", "dacode_persubtask_manifest.jsonl"), task_id)
    unified = load_task(
        os.path.join(PROJECT_ROOT, "data", "dacode_unified_manifest.jsonl"), task_id)

    # Per-subtask: use its OWN lake dir + a throwaway sandbox
    dump(persubtask, "PER-SUBTASK",
         os.path.join(PROJECT_ROOT, "data", "_debug_sandbox_persubtask"))
    # Global: the unified lake (the 0.2529 setting)
    dump(unified, "GLOBAL (unified lake)",
         os.path.join(PROJECT_ROOT, "data", "_debug_sandbox_global"))


if __name__ == "__main__":
    main()
