"""Top-level CLI to run Triadic DGM evaluation on DABench and/or ScienceAgentBench.

Examples:
    # smoke (5 tasks each), fast: thinking off
    python -m triadic_dgm.benchmark.run_eval --benchmark both --limit 5 --no-thinking

    # full run, faithful accuracy: thinking on
    python -m triadic_dgm.benchmark.run_eval --benchmark both --limit -1 --thinking

    # resume an interrupted run
    python -m triadic_dgm.benchmark.run_eval --benchmark dabench --resume
"""
import argparse
import json
import os
import sys

# Keep the eval process TF-free and unbuffered regardless of how it's launched.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _set_thinking(enable: bool) -> None:
    os.environ["QWEN_ENABLE_THINKING"] = "true" if enable else "false"
    os.environ["QWEN_DEBUG"] = "0"


def _repo_root() -> str:
    """Walk up from this file to the dir containing .env / config.yaml (the repo root,
    = data-agent/). Robust to being run from any cwd; avoids the level-counting pitfall
    that previously sent results to the PARENT of data-agent (3 `..` from benchmark/)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(d, "config.yaml")) or os.path.exists(os.path.join(d, ".env")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.getcwd()


def run_dabench(args):
    _set_thinking(args.thinking)
    from triadic_dgm.benchmark.harness.dabench.harness import run as dabench_run
    return dabench_run(
        limit=args.limit, offset=args.offset, max_workers=args.max_workers,
        max_inner_retries=args.max_inner_retries, exec_timeout=args.exec_timeout,
        resume=args.resume, results_dir=os.path.join(args.results_root, "dabench"),
    )


def run_sab(args):
    _set_thinking(args.thinking)
    from triadic_dgm.benchmark.harness.scienceagentbench.harness import run as sab_run
    return sab_run(
        limit=args.limit, offset_id=args.offset, max_workers=args.max_workers,
        max_inner_retries=args.max_inner_retries, exec_timeout=args.exec_timeout,
        eval_timeout=args.eval_timeout, resume=args.resume, prefer_csv=args.prefer_csv,
        gradeable_only=args.gradeable_only, feasible_only=args.feasible_only,
        results_dir=os.path.join(args.results_root, "scienceagentbench"),
    )


def write_report(summaries: dict, out_path: str):
    """Markdown report with a table vs the EvoDS paper reference numbers."""
    lines = ["# Triadic DGM — Evaluation Report", ""]
    lines += [
        "| Benchmark | Tasks | Passed | Pass rate | Env-feasible pass | Env-blocked | EvoDS (paper) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bench, s in summaries.items():
        if not s:
            continue
        ref = s.get("reference_evods_paper", {})
        ref_str = next(iter(ref.values())) if ref else "—"
        # DABench has by_level but no env subset; SAB has pass_rate_envfeasible (the
        # fair solver metric, excluding tasks blocked by missing scientific deps).
        if "pass_rate_envfeasible" in s:
            ef = f"{s.get('pass_rate_envfeasible')} ({s.get('env_feasible_passed')}/{s.get('env_feasible_total')})"
        else:
            ef = "—"
        env_b = s.get("env_blocked", "—")
        lines.append(f"| {bench} | {s.get('total','—')} | {s.get('passed','—')} | "
                     f"{s.get('pass_rate','—')} | {ef} | {env_b} | {ref_str} |")
    lines += ["", "## Notes / caveats", ""]
    lines += [
        "- **LLM**: Qwen3.5-35B-A3B-FP8 via the proxy in `.env` (hybrid reasoning; "
        f"thinking={os.environ.get('QWEN_ENABLE_THINKING')}). The paper's numbers use frontier "
        "models, so absolute pass-rates are expected to differ.",
        "- **DABench grading**: programmatic `@field[value]` parse + numeric/string match, with an "
        "LLM-judge fallback only when the agent ignores the format.",
        "- **DABench Goldilocks window**: the verifier's epiplexity gate (default [0.5,1.8]) is "
        "calibrated for code-evolution and rejects correct DA solutions; it is widened for this eval.",
        "- **ScienceAgentBench grading**: official per-task `eval_program`s run locally. "
        "**Env-feasible pass** is the pass-rate over tasks whose gold program needs only installed "
        "deps — the fairest solver metric (tasks blocked by missing scientific deps such as "
        "deepchem/rdkit/geopandas/mne, or the GPT-4o visual judge, are excluded, not counted as "
        "solver failures).",
        "- **Sandbox**: local subprocess (cwd = per-task workspace), not Docker.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[run_eval] report written to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Run Triadic DGM eval on DABench / ScienceAgentBench")
    p.add_argument("--benchmark", choices=["dabench", "scienceagentbench", "both"], default="both")
    p.add_argument("--limit", type=int, default=5, help="tasks per benchmark (-1 = all)")
    p.add_argument("--offset", type=int, default=0, help="skip tasks with id < offset")
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--max-inner-retries", type=int, default=3)
    p.add_argument("--exec-timeout", type=int, default=600)
    p.add_argument("--eval-timeout", type=int, default=600)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--thinking", dest="thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="thinking", action="store_false")
    p.add_argument("--prefer-csv", dest="prefer_csv", action="store_true", default=True,
                   help="SAB: order CSV-gradeable tasks first")
    p.add_argument("--no-prefer-csv", dest="prefer_csv", action="store_false")
    p.add_argument("--gradeable-only", dest="gradeable_only", action="store_true", default=False,
                   help="SAB: skip the 64 plot tasks (need GPT-4o visual judge, ungradeable locally)")
    p.add_argument("--feasible-only", dest="feasible_only", action="store_true", default=False,
                   help="SAB: skip tasks whose gold program needs uninstalled scientific deps")
    p.add_argument("--results-root", default=os.path.join(_repo_root(), "results"))
    args = p.parse_args()

    summaries = {}
    if args.benchmark in ("dabench", "both"):
        try:
            summaries["dabench"] = run_dabench(args)
        except Exception as e:  # noqa: BLE001
            print(f"[run_eval] DABench failed: {e}", file=sys.stderr)
            summaries["dabench"] = None
    if args.benchmark in ("scienceagentbench", "both"):
        try:
            summaries["scienceagentbench"] = run_sab(args)
        except Exception as e:  # noqa: BLE001
            print(f"[run_eval] ScienceAgentBench failed: {e}", file=sys.stderr)
            summaries["scienceagentbench"] = None

    report_path = os.path.join(args.results_root, "report.md")
    os.makedirs(args.results_root, exist_ok=True)
    write_report(summaries, report_path)
    print("[run_eval] summaries:")
    print(json.dumps({k: v for k, v in summaries.items() if v}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
