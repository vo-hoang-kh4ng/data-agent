#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_tokens(paths):
    tokens = Counter()
    seen = 0
    for path in paths:
        seen += 1
        try:
            usage = load_json(path).get("token_usage", {})
            tokens["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            tokens["completion_tokens"] += int(usage.get("completion_tokens", 0))
            tokens["total_tokens"] += int(usage.get("total_tokens", 0))
        except (OSError, ValueError, TypeError):
            tokens["unreadable_result_files"] += 1
    return dict(tokens), seen


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize one immutable TDGM evolution run")
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.run_dir).expanduser().resolve()
    archive = load_json(root / "evolution_archive.json")
    summary = load_json(root / "run_summary.json") if (root / "run_summary.json").exists() else {}
    attempts = []
    attempts_path = root / "candidate_attempts.jsonl"
    if attempts_path.exists():
        attempts = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    result_files = list((root / "benchmark_91" / "sandbox").glob("*/dabench/result.json"))
    benchmark_tokens, benchmark_seen = collect_tokens(result_files)
    dev_result_files = list((root / "candidates").glob("*/stage*/sandbox/*/dabench/result.json"))
    dev_result_files += list((root / "development_protocol").glob("seed_stage*/sandbox/*/dabench/result.json"))
    development_tokens, development_seen = collect_tokens(dev_result_files)

    nodes = archive.get("nodes", [])
    event_counts = Counter(event.get("kind", "unknown") for event in archive.get("events", []))
    attempt_counts = Counter(attempt.get("status", "unknown") for attempt in attempts)
    report = {
        "run_dir": str(root),
        "cycles_requested": archive.get("run_metadata", {}).get("cycles"),
        "archive_nodes": len(nodes),
        "admitted_mutations": max(0, len(nodes) - 1),
        "attempt_counts": dict(attempt_counts),
        "event_counts": dict(event_counts),
        "goldilocks_rate_over_attempts": (
            sum(a.get("goldilocks_status") == "PASS" for a in attempts) / len(attempts) if attempts else 0.0
        ),
        "best_inner_node": summary.get("best_node"),
        "benchmark_91": summary.get("benchmark_91"),
        "benchmark_token_usage": benchmark_tokens,
        "benchmark_result_files_seen": benchmark_seen,
        "development_task_token_usage": development_tokens,
        "development_result_files_seen": development_seen,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
