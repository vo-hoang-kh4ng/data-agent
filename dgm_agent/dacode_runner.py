"""
DA-Code Benchmark Runner for Triadic DGM
==========================================
CLI wrapper around DACodeOrchestrator.
Uses the full Triadic DGM pipeline: Planner → Solver → Verifier + CapacityManager + RuleLibrary.

Key improvements over previous version:
- Epiplexity-driven budget (NCD-based, not static hardness)
- RIMRULE rules extracted immediately after each repair (not deferred)
- Goldilocks Zone filtering ([0.5, 2.2])
- Triadic vs Fallback task tracking

Usage:
    python -m dgm_agent.dacode_runner --manifest data/dacode_unified_manifest.jsonl
    python -m dgm_agent.dacode_runner --example_name di-text-001
    python -m dgm_agent.dacode_runner --force_rerun
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dgm_agent.dacode_orchestrator import DACodeOrchestrator


def main():
    parser = argparse.ArgumentParser(description="DA-Code Runner — Triadic DGM Pipeline")
    parser.add_argument("--manifest", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "dacode_unified_manifest.jsonl"))
    parser.add_argument("--model", type=str,
                        default=os.environ.get("DACODE_MODEL", "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8"))
    parser.add_argument("--sandbox_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "dacode_sandbox"))
    parser.add_argument("--max_debug_rounds", type=int, default=3)
    parser.add_argument("--example_name", type=str, default="")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--max_tasks", type=int, default=0,
                        help="Process at most N tasks then exit (0=all). For memory-bounded "
                             "batched runs: each invocation loads E5 fresh and exits after N, "
                             "releasing RAM. Combine with resume-skip to chain batches.")
    args = parser.parse_args()

    # The verifier runs scripts with cwd=<task_sandbox> via subprocess; a RELATIVE sandbox_dir
    # gets resolved relative to that new cwd and path-doubles -> uniform "can't open file"
    # failures. Absolutize against PROJECT_ROOT so a relative --sandbox_dir can never bite.
    if not os.path.isabs(args.sandbox_dir):
        args.sandbox_dir = os.path.join(PROJECT_ROOT, args.sandbox_dir)

    # Load tasks
    tasks = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    if args.example_name:
        tasks = [t for t in tasks if args.example_name in t["task_id"]]

    print(f"🧠 Triadic DGM — DA-Code Benchmark ({len(tasks)} tasks)")
    print(f"   Pipeline: Planner → Solver → Verifier + Epiplexity Budget + RIMRULE + Goldilocks")
    print(f"   Model: {args.model}")
    print(f"   Sandbox: {args.sandbox_dir}")

    orchestrator = DACodeOrchestrator(
        model=args.model,
        sandbox_dir=args.sandbox_dir,
        max_debug_rounds=args.max_debug_rounds,
    )

    # Lake-integrity guard: freeze the lake file set at startup and scrub any stray
    # file the generated code may write into the lake dir after each task. This is the
    # robust guarantee that the benchmark lake stays the verified clean set throughout
    # the run, independent of whatever paths the agent's emitted code writes to.
    lake_dir = ""
    for t in tasks:
        d = t.get("data_lake_dir") or ""
        if d:
            lake_dir = os.path.join(PROJECT_ROOT, d) if not os.path.isabs(d) else d
            break
    LAKE_FROZEN = set(os.listdir(lake_dir)) if (lake_dir and os.path.isdir(lake_dir)) else set()

    def scrub_lake():
        if not (lake_dir and os.path.isdir(lake_dir)):
            return []
        stray = sorted(set(os.listdir(lake_dir)) - LAKE_FROZEN)
        for s in stray:
            try:
                os.remove(os.path.join(lake_dir, s))
            except OSError:
                pass
        return stray

    done = 0
    failed = 0
    triadic_done = 0
    fallback_done = 0
    processed = 0  # tasks actually attempted this invocation (for --max_tasks batching)

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        print(f"\n[{i+1}/{len(tasks)}] {task_id}")

        # Resume support: skip any task that already has a result.json (even an unfinished
        # one) unless --force_rerun. This avoids re-running the 36 already-attempted tasks
        # when resuming a crashed batch.
        existing = os.path.join(orchestrator.sandbox_dir, task_id, "dabench", "result.json")
        if not args.force_rerun and os.path.exists(existing):
            try:
                prev = json.load(open(existing, encoding="utf-8"))
                if prev.get("finished"):
                    done += 1
                else:
                    failed += 1
                print(f"  ⏭️  skipping existing result.json (finished={prev.get('finished')}) — resume")
            except Exception:
                done += 1
                print(f"  ⏭️  skipping existing result.json (unreadable) — resume")
            continue

        # A single transient proxy error (e.g. 502 Bad Gateway) must never abort the whole
        # batch. Wrap each task so one crash is logged and the loop continues.
        processed += 1
        try:
            result = orchestrator.run_task(task, force=args.force_rerun)
        except Exception as e:
            print(f"  💥 CRASHED (logged, continuing): {repr(e)[:200]}")
            failed += 1
            continue

        if result.get("skipped"):
            done += 1
            continue

        # Lake-integrity guard: remove any file the agent's code wrote into the lake dir.
        stray = scrub_lake()
        if stray:
            print(f"  🧹 lake scrubbed {len(stray)} stray write(s): {stray}")

        if result["finished"]:
            done += 1
            if result.get("solved_by") == "triadic_dgm":
                triadic_done += 1
            elif result.get("solved_by") == "fallback_1shot":
                fallback_done += 1
        else:
            failed += 1

        total = done + failed
        if total % 10 == 0 or total == len(tasks):
            rules = len(orchestrator.rule_library.rules)
            print(f"\n📊 Progress: {total}/{len(tasks)} | Done: {done} (Triadic: {triadic_done}, "
                  f"Fallback: {fallback_done}) | Failed: {failed} | RIMRULE rules: {rules} | "
                  f"Goldilocks: {orchestrator.goldilocks_pass}P/{orchestrator.goldilocks_fail}F")

        if args.max_tasks and processed >= args.max_tasks:
            print(f"\n🛑 --max_tasks={args.max_tasks} reached; exiting batch to release memory "
                  f"(resume to continue).")
            break

    rules = len(orchestrator.rule_library.rules)
    print(f"\n{'='*60}")
    print(f"📊 FINAL: {done}/{len(tasks)} done ({done*100/max(1,len(tasks)):.1f}%)")
    print(f"   Triadic DGM solved: {triadic_done}")
    print(f"   Fallback solved:    {fallback_done}")
    print(f"   Failed:             {failed}")
    print(f"   Goldilocks PASS:    {orchestrator.goldilocks_pass}")
    print(f"   Goldilocks FAIL:    {orchestrator.goldilocks_fail}")
    print(f"   RIMRULE rules:      {rules}")
    print(f"{'='*60}")

    # Print top rules
    if rules > 0:
        print(f"\n📚 Top RIMRULE Rules Learned:")
        top = orchestrator.rule_library.get_top_rules(5)
        if top:
            print(top)

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
