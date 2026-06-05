#!/usr/bin/env python3
"""
LAMBDA Paper Benchmark Runner
==============================
Evaluates the data-agent (LAMBDA) system on all benchmarks from the LAMBDA paper
(arxiv 2407.17535) and generates comparison tables.

Usage:
    cd data-agent
    python -m benchmarks.run_benchmarks                          # Run all benchmarks
    python -m benchmarks.run_benchmarks --group group1_classification   # Run specific group
    python -m benchmarks.run_benchmarks --download-only           # Only download datasets
    python -m benchmarks.run_benchmarks --retry-failed           # Retry failed tasks
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

# Ensure repo root is in path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Also add parent of benchmarks for module imports
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from benchmarks.datasets.registry import ALL_DATASETS, get_datasets_by_group, get_all_groups, DatasetInfo
from benchmarks.datasets.download import download_all, download_dataset
from benchmarks.runners.lambda_runner import BenchmarkRunner, build_instruction
from benchmarks.report.paper_results import get_paper_result


CHECKPOINT_DIR = os.path.join(BENCH_DIR, "checkpoints")


def get_checkpoint_path() -> str:
    return os.path.join(CHECKPOINT_DIR, "benchmark_state.json")


def load_checkpoint() -> dict:
    path = get_checkpoint_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"run_id": "", "results": {}, "config": {}}


def save_checkpoint(state: dict):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(get_checkpoint_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def make_task_key(ds: DatasetInfo, model: str) -> str:
    return f"{ds.group}__{ds.name}__{model}"


def run_benchmark(datasets=None, groups=None, retry_failed=False, max_attempts=5):
    """Main benchmark execution loop."""
    print("=" * 70)
    print("  LAMBDA Paper Benchmark Suite")
    print("  Model: Qwen 3.5 35B (via proxy)")
    print("=" * 70)

    # Download datasets
    print("\n[1/4] Downloading datasets...")
    download_all()

    # Load checkpoint
    state = load_checkpoint()
    if not state.get("run_id"):
        state["run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        state["config"] = {
            "llm_model": "Qwen 3.5 35B",
            "max_attempts": max_attempts,
            "started_at": datetime.now().isoformat(),
        }
    results = state.get("results", {})

    # Filter datasets
    if groups:
        run_datasets = []
        for g in groups:
            run_datasets.extend(get_datasets_by_group(g))
    elif datasets:
        run_datasets = [d for d in ALL_DATASETS if d.name in datasets]
    else:
        run_datasets = ALL_DATASETS

    # Count tasks — only count tasks relevant to the selected datasets
    run_task_keys = set()
    for ds in run_datasets:
        for model in ds.models:
            run_task_keys.add(make_task_key(ds, model))

    total_tasks = len(run_task_keys)
    completed = sum(1 for k in run_task_keys if k in results and results[k].get("status") == "completed")
    remaining = total_tasks - completed

    print(f"\n[2/4] Task Overview:")
    print(f"  Total tasks:     {total_tasks}")
    print(f"  Already done:    {completed}")
    print(f"  To run now:      {remaining}")

    if remaining == 0 and not retry_failed:
        print("\n  All tasks already completed! Use --retry-failed to retry failures.")
        print("  Generating report from existing results...")
        state["results"] = results
        save_checkpoint(state)
        generate_report(state)
        return state

    # Initialize runner
    print(f"\n[3/4] Initializing LAMBDA BenchmarkRunner...")
    runner = BenchmarkRunner(max_attempts=max_attempts)

    # Run evaluations
    print(f"\n[4/4] Running evaluations...\n")
    task_count = 0
    start_time = time.time()

    for ds in run_datasets:
        print(f"\n{'─' * 60}")
        print(f"  Dataset: {ds.name} ({ds.group})")
        print(f"  Type: {ds.task_type} | Metric: {ds.metric}")
        print(f"  Models to test: {len(ds.models)}")
        print(f"{'─' * 60}")

        # For file-based datasets, upload once
        needs_upload = ds.source not in ("torch", "synthetic") or ds.group in ("group6_text",)
        data_path = None
        if ds.source not in ("torch", "synthetic"):
            data_path = os.path.join(os.path.dirname(__file__), "datasets", "data", ds.filename)
            if not os.path.exists(data_path):
                print(f"  ⚠ Dataset file not found: {data_path}")
                print(f"    Skipping all models for this dataset.")
                for model in ds.models:
                    key = make_task_key(ds, model)
                    results[key] = {
                        "status": "skipped", "dataset": ds.name, "model": model,
                        "group": ds.group, "metric_type": ds.metric,
                        "error": f"Dataset file not found: {data_path}",
                        "timestamp": datetime.now().isoformat(),
                    }
                continue

        for model in ds.models:
            key = make_task_key(ds, model)

            # Skip if already completed (unless retry_failed)
            if key in results and results[key].get("status") == "completed" and not retry_failed:
                val = results[key].get("metric_value")
                print(f"  [skip] {model}: {val}")
                continue

            # Skip failed tasks unless retry_failed
            if (key in results and results[key].get("status") == "failed"
                    and not retry_failed):
                print(f"  [skip-failed] {model}")
                continue

            task_count += 1
            print(f"\n  [{task_count}/{remaining}] {ds.name} × {model}")

            # Reset session for each task
            runner._reset_session()

            # Upload dataset if needed
            if needs_upload and data_path and os.path.exists(data_path):
                try:
                    runner.upload_dataset(data_path)
                except Exception as e:
                    print(f"    ✗ Upload failed: {e}")
                    results[key] = {
                        "status": "failed", "dataset": ds.name, "model": model,
                        "group": ds.group, "metric_type": ds.metric,
                        "metric_value": None, "error": f"Upload failed: {e}",
                        "attempts": 0, "timestamp": datetime.now().isoformat(),
                    }
                    continue

            # Build instruction and run
            instruction = build_instruction(ds, model)
            print(f"    → Running ({ds.metric})...")
            task_start = time.time()

            try:
                result = runner.run_task(instruction, metric_type=ds.metric,
                                         max_attempts=max_attempts)
            except Exception as e:
                print(f"    ✗ Exception: {e}")
                results[key] = {
                    "status": "failed", "dataset": ds.name, "model": model,
                    "group": ds.group, "metric_type": ds.metric,
                    "metric_value": None, "error": str(e),
                    "attempts": 0, "timestamp": datetime.now().isoformat(),
                }
                save_checkpoint({**state, "results": results})
                continue

            elapsed = time.time() - task_start

            # Store result
            paper_val = get_paper_result(ds.name, model)
            our_val = result.get("metric_value")

            status = "completed" if result["success"] else "failed"
            if not result["success"] and our_val is not None:
                status = "completed"  # Had errors but still got a metric

            results[key] = {
                "status": status,
                "dataset": ds.name,
                "model": model,
                "group": ds.group,
                "metric_type": ds.metric,
                "metric_value": our_val,
                "paper_value": paper_val,
                "attempts": result["attempts"],
                "elapsed_seconds": round(elapsed, 1),
                "error": result.get("error"),
                "timestamp": datetime.now().isoformat(),
            }

            # Print result
            if our_val is not None:
                paper_str = f" (paper: {paper_val})" if paper_val is not None else ""
                print(f"    ✓ {ds.metric} = {our_val:.4f}{paper_str} [{result['attempts']} attempt(s), {elapsed:.0f}s]")
            else:
                print(f"    ✗ Could not parse {ds.metric} [{result['attempts']} attempt(s), {elapsed:.0f}s]")
                if result.get("error"):
                    err_preview = result["error"][:150].replace("\n", " ")
                    print(f"      Error: {err_preview}...")

            # Save checkpoint after each task
            save_checkpoint({**state, "results": results})

    # Cleanup
    runner.shutdown()
    total_time = time.time() - start_time

    # Final save
    state["results"] = results
    state["config"]["completed_at"] = datetime.now().isoformat()
    state["config"]["total_time_seconds"] = round(total_time, 1)
    state["config"]["tasks_run"] = task_count
    save_checkpoint(state)

    # Generate report
    generate_report(state)

    return state


def generate_report(state: dict):
    """Generate markdown comparison report."""
    results = state.get("results", {})
    if not results:
        print("No results to report.")
        return

    report_path = os.path.join(REPO_ROOT, "BENCHMARK_RESULTS.md")
    lines = []
    lines.append("# LAMBDA Benchmark Results — Paper Comparison\n")
    lines.append(f"**Run ID:** {state.get('run_id', 'N/A')}")
    lines.append(f"**LLM:** Qwen 3.5 35B (via proxy.onebot.meobeo.ai)")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Max self-correction attempts:** {state.get('config', {}).get('max_attempts', 5)}")
    lines.append("")

    # Group results
    groups = {}
    for key, r in results.items():
        g = r.get("group", "unknown")
        if g not in groups:
            groups[g] = []
        groups[g].append(r)

    # Summary stats
    total = len(results)
    completed = sum(1 for r in results.values() if r.get("status") == "completed")
    failed = sum(1 for r in results.values() if r.get("status") == "failed")
    skipped = sum(1 for r in results.values() if r.get("status") == "skipped")

    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total tasks | {total} |")
    lines.append(f"| Completed | {completed} |")
    lines.append(f"| Failed | {failed} |")
    lines.append(f"| Skipped | {skipped} |")
    lines.append(f"| Total time | {state.get('config', {}).get('total_time_seconds', 'N/A')}s |")
    lines.append("")

    # Per-group tables
    group_titles = {
        "group1_classification": "Group 1: Classical Tabular Classification (Accuracy %)",
        "group2_regression": "Group 2: Classical Tabular Regression (MSE)",
        "group3_genomic": "Group 3: High-Dimensional Genomic Data (Accuracy %)",
        "group4_missing": "Group 4: Missing Data (Accuracy %)",
        "group5_image": "Group 5: Image Data — MNIST (Accuracy %)",
        "group6_text": "Group 6: Text Data — SMS Spam (Accuracy %)",
        "group7_knowledge": "Group 7: Knowledge Integration (Score 0-1)",
    }

    for gname, title in group_titles.items():
        if gname not in groups:
            continue

        lines.append(f"## {title}\n")

        # Group by dataset
        datasets = {}
        for r in groups[gname]:
            dname = r.get("dataset", "Unknown")
            if dname not in datasets:
                datasets[dname] = []
            datasets[dname].append(r)

        for dname, rows in datasets.items():
            lines.append(f"### {dname}\n")
            lines.append("| Model | Paper (GPT-4) | Ours (Qwen 3.5) | Delta | Status |")
            lines.append("|-------|--------------|-----------------|-------|--------|")

            for r in sorted(rows, key=lambda x: x.get("model", "")):
                model = r.get("model", "?")
                ours = r.get("metric_value")
                paper = r.get("paper_value")
                status = r.get("status", "?")
                metric_type = r.get("metric_type", "accuracy")

                # Normalize: paper values for accuracy are in % (0-100), ours are 0-1
                # Convert ours to % for comparison
                ours_pct = ours * 100 if ours is not None and ours <= 1.0 and metric_type == "accuracy" else ours
                paper_pct = paper if paper is not None else None

                ours_str = f"{ours_pct:.2f}" if ours_pct is not None else "N/A"
                paper_str = f"{paper_pct:.2f}" if paper_pct is not None else "N/A"

                # Calculate delta
                if ours_pct is not None and paper_pct is not None:
                    if metric_type == "mse":
                        # MSE: lower is better. Delta = paper - ours (positive = we're better)
                        delta = paper_pct - ours_pct
                        delta_str = f"{delta:+.4f}"
                        if delta > 0.001:
                            delta_str += " ✓"
                        elif delta < -0.001:
                            delta_str += " ✗"
                    elif metric_type == "accuracy":
                        # Accuracy %: higher is better. Delta = ours - paper
                        delta = ours_pct - paper_pct
                        delta_str = f"{delta:+.2f}"
                        if delta > 0.01:
                            delta_str += " ✓"
                        elif delta < -0.01:
                            delta_str += " ✗"
                    else:
                        delta_str = "—"
                else:
                    delta_str = "—"

                status_icon = "✅" if status == "completed" else "❌" if status == "failed" else "⏭️"
                lines.append(f"| {model} | {paper_str} | {ours_str} | {delta_str} | {status_icon} |")
            lines.append("")

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'=' * 70}")
    print(f"  Report saved to: {report_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="LAMBDA Paper Benchmark Runner")
    parser.add_argument("--group", nargs="*", help="Run specific group(s)")
    parser.add_argument("--dataset", nargs="*", help="Run specific dataset(s)")
    parser.add_argument("--max-attempts", type=int, default=5, help="Max self-correction attempts")
    parser.add_argument("--download-only", action="store_true", help="Only download datasets")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed tasks")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    args = parser.parse_args()

    if args.download_only:
        download_all()
        return

    if args.report_only:
        state = load_checkpoint()
        generate_report(state)
        return

    run_benchmark(
        datasets=args.dataset,
        groups=args.group,
        retry_failed=args.retry_failed,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
