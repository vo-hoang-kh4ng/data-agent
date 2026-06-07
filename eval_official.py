"""
Official DA-Code Evaluator Wrapper
====================================
Uses the EXACT evaluator from da-code-repo to score results.
Ensures score calculation is 100% identical to the official DA-Code paper.

Usage:
    python eval_official.py
    python eval_official.py --example_name di-text-001
"""

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List

# Use local eval metrics (copied from da-code-repo)
from eval.text import compare_text
from eval.table import compare_csv

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFAULT_SANDBOX_DIR = os.path.join(DATA_DIR, "dacode_sandbox")
DEFAULT_GOLD_DIR = os.path.join(DATA_DIR, "dacode_gold", "gold")
DEFAULT_EVAL_CONFIG = os.path.join(PROJECT_ROOT, "eval", "configs", "eval_all.jsonl")
DEFAULT_RESULTS_FILE = os.path.join(DATA_DIR, "dacode_official_eval_results.json")


def _load_eval_configs(path: str) -> Dict[str, dict]:
    """Load eval_all.jsonl into a dict keyed by task_id."""
    configs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                configs[c["id"]] = c
    return configs


def _get_gold_result(eval_result_cfg, gold_dir: str, task_id: str):
    """Extract gold data from eval config's 'result' field.

    For text tasks: returns list of gold dicts
    For CSV tasks: returns list of CSV file paths
    """
    results = eval_result_cfg if isinstance(eval_result_cfg, list) else [eval_result_cfg]

    gold_data = []
    for entry in results:
        if "number" in entry:
            # Text task: gold is the number dict
            gold_data.append(entry["number"])
        elif "file" in entry:
            # CSV task: gold is file path(s)
            files = entry["file"] if isinstance(entry["file"], list) else [entry["file"]]
            csv_paths = []
            for f in files:
                # Use basename for gold dir
                csv_path = os.path.join(gold_dir, task_id, os.path.basename(f))
                csv_paths.append(csv_path)
            gold_data.append(csv_paths)

    return gold_data


def _get_output_result(task_id: str, sandbox_dir: str, eval_result_cfg: list, is_text: bool):
    """Get agent output in format expected by official evaluator.

    For text tasks: returns the result string
    For CSV tasks: returns path to output CSV
    """
    task_sandbox = os.path.join(sandbox_dir, task_id)
    result_json = os.path.join(task_sandbox, "dabench", "result.json")

    if not os.path.exists(result_json):
        return None, False

    data = json.load(open(result_json, "r", encoding="utf-8"))
    finished = data.get("finished", False)
    agent_result = data.get("result", "")

    if is_text:
        # For text tasks, return the result string directly
        return agent_result, finished
    else:
        # For CSV tasks, find the output CSV file
        # First try: look for CSV files mentioned in eval config
        results = eval_result_cfg if isinstance(eval_result_cfg, list) else [eval_result_cfg]
        csv_paths = []
        for entry in results:
            if "file" in entry:
                files = entry["file"] if isinstance(entry["file"], list) else [entry["file"]]
                for f in files:
                    basename = os.path.basename(f)
                    # Check various locations
                    for candidate in [
                        os.path.join(task_sandbox, basename),
                        os.path.join(task_sandbox, "dabench", basename),
                    ]:
                        if os.path.exists(candidate):
                            csv_paths.append(candidate)
                            break

        if csv_paths:
            return csv_paths[0] if len(csv_paths) == 1 else csv_paths, finished

        # Fallback: find any CSV in sandbox
        if os.path.exists(task_sandbox):
            for f in os.listdir(task_sandbox):
                if f.endswith(".csv") and not f.startswith("_"):
                    return os.path.join(task_sandbox, f), finished

        # Also check result_files field
        result_files = data.get("result_files", [])
        if isinstance(result_files, list):
            for rf in result_files:
                if isinstance(rf, str) and rf.endswith(".csv"):
                    rf_path = os.path.join(task_sandbox, rf)
                    if os.path.exists(rf_path):
                        return rf_path, finished

        return None, finished


def evaluate_task(task_id: str, eval_config: dict, sandbox_dir: str, gold_dir: str) -> Dict:
    """Evaluate a single task using official DA-Code metrics."""
    config = eval_config.get("config", {})
    funcs = eval_config["func"]
    if isinstance(funcs, str):
        funcs = [funcs]

    results_cfg = eval_config["result"]
    if not isinstance(results_cfg, list):
        results_cfg = [results_cfg]

    options = eval_config.get("options", [{}])
    if not isinstance(options, list):
        options = [options]
    # Pad options
    while len(options) < len(funcs):
        options.append({})

    conj = eval_config.get("conj", "avg")
    hardness = config.get("hardness", "unknown")
    task_type = config.get("task", "unknown")
    result_type = config.get("type", "unknown")

    # Get gold data
    gold_data = _get_gold_result(results_cfg, gold_dir, task_id)

    # Get output
    is_text = funcs[0] in ("compare_text", "compare_number")
    output, finished = _get_output_result(task_id, sandbox_dir, results_cfg, is_text)

    if not finished:
        return {
            "task_id": task_id,
            "score": 0.0,
            "finished": False,
            "category": task_type,
            "hardness": hardness,
            "eval_type": result_type,
            "error": "Task not finished",
        }

    if output is None:
        return {
            "task_id": task_id,
            "score": 0.0,
            "finished": True,
            "category": task_type,
            "hardness": hardness,
            "eval_type": result_type,
            "error": "No output file found",
        }

    # Evaluate
    scores = []
    for idx, func_name in enumerate(funcs):
        opt = options[idx] if idx < len(options) else {}
        gold = gold_data[idx] if idx < len(gold_data) else gold_data[0]

        try:
            if func_name == "compare_text":
                # compare_text expects: result=str, expected=list of dicts
                result_str = output if isinstance(output, str) else str(output)
                # gold is list of dicts from "number" field
                if isinstance(gold, list):
                    # Each element is a list of dicts
                    expected = gold
                else:
                    expected = [gold]

                result = compare_text(result_str, expected, **opt)
                if isinstance(result, dict):
                    scores.append(result.get("score", 0.0))
                else:
                    scores.append(float(result) if result else 0.0)

            elif func_name == "compare_csv":
                # compare_csv expects: result=path(str), expected=str or list[str]
                # Official evaluator passes expected as str (single gold path)
                # so compare_csv enters the `isinstance(expected, str)` branch
                # which wraps ignore_order into a list properly
                output_path = output if isinstance(output, str) else output[0]
                if isinstance(gold, list) and len(gold) == 1:
                    # Pass as single string to avoid bool subscript error
                    result = compare_csv(output_path, gold[0], **opt)
                elif isinstance(gold, str):
                    result = compare_csv(output_path, gold, **opt)
                elif isinstance(gold, list):
                    result = compare_csv(output_path, gold, **opt)
                else:
                    result = compare_csv(output_path, str(gold), **opt)

                if isinstance(result, dict):
                    scores.append(result.get("score", 0.0))
                else:
                    scores.append(float(result) if result else 0.0)
            else:
                scores.append(0.0)

        except Exception as e:
            print(f"      ⚠️ {func_name} error: {e}")
            scores.append(0.0)

    # Apply conjunction rule
    if conj == "avg":
        total_score = sum(scores) / len(scores) if scores else 0.0
    elif conj == "max":
        total_score = max(scores) if scores else 0.0
    elif conj == "min":
        total_score = min(scores) if scores else 0.0
    elif conj == "and":
        total_score = float(all(s != 0 for s in scores))
    elif conj == "or":
        total_score = float(any(s != 0 for s in scores))
    else:
        total_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "task_id": task_id,
        "score": total_score,
        "finished": True,
        "category": task_type,
        "hardness": hardness,
        "eval_type": result_type,
        "metric_scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Official DA-Code Evaluation")
    parser.add_argument("--sandbox_dir", type=str, default=DEFAULT_SANDBOX_DIR)
    parser.add_argument("--gold_dir", type=str, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--eval_config", type=str, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--results_file", type=str, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--example_name", type=str, default="")
    args = parser.parse_args()

    # Use the EXACT official evaluator from da-code-repo
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "da-code-repo"))
    from da_agent.evaluators.evaluation import Evaluator

    # Load eval configs
    eval_configs = _load_eval_configs(args.eval_config)

    # Load our manifest to get task list
    manifest_path = os.path.join(DATA_DIR, "dacode_unified_manifest.jsonl")
    task_ids = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                task_ids.append(json.loads(line)["task_id"])

    if args.example_name:
        task_ids = [t for t in task_ids if args.example_name in t]

    print(f"📊 Official DA-Code Evaluation ({len(task_ids)} tasks)")
    print(f"   Sandbox: {args.sandbox_dir}")
    print(f"   Gold:    {args.gold_dir}")
    print(f"   Eval:    {args.eval_config}")
    print(f"   Evaluator: da-code-repo/da_agent/evaluators/evaluation.py (OFFICIAL)")

    # Filter eval configs to our task list
    eval_list = [eval_configs[tid] for tid in task_ids if tid in eval_configs]

    # Run official evaluator (bypass buggy evaluate() method, call _get_eval_config_info directly)
    evaluator = Evaluator(output_dir=args.sandbox_dir, gold_dir=args.gold_dir, timeout_seconds=30)

    results = []
    for i, eval_config in enumerate(eval_list):
        task_id = eval_config["id"]
        print(f"  [{i+1}/{len(eval_list)}] {task_id}...", end=" ", flush=True)

        try:
            id, exist, trajectory_info, eval_info = evaluator._get_eval_config_info(eval_config)
        except Exception as e:
            print(f"❌ parse error: {e}")
            results.append({"task_id": task_id, "score": 0.0, "finished": False,
                           "category": "?", "hardness": "?", "eval_type": "?"})
            continue

        if not exist:
            print(f"⏭️ no result")
            continue

        config = eval_config.get("config", {})
        task_type = config.get("task", "unknown")
        hardness = config.get("hardness", "unknown")
        result_type = config.get("type", "unknown")

        if not trajectory_info.get("finished", False):
            print(f"❌ not finished")
            results.append({"task_id": task_id, "score": 0.0, "finished": False,
                           "category": task_type, "hardness": hardness, "eval_type": result_type})
            continue

        # Run metrics
        _, metric_list, metric_conj, metric_options, output_results, gold_results = eval_info
        scores = []
        for idx, metric in enumerate(metric_list):
            try:
                output_result = output_results[idx]
                gold_result = gold_results[idx]
                if config:
                    metric_options[idx].update({"config": config})
                result = metric(output_result, gold_result, **metric_options[idx])
            except Exception as e:
                scores.append(0.0)
                continue
            if isinstance(result, dict):
                scores.append(result.get("score", 0.0))
            else:
                scores.append(float(result) if isinstance(result, (float, int)) else 0.0)

        scores = [s if isinstance(s, (float, int)) else 0.0 for s in scores]
        if metric_conj == "avg":
            total_score = sum(scores) / len(scores) if scores else 0.0
        elif metric_conj == "max":
            total_score = max(scores) if scores else 0.0
        elif metric_conj == "min":
            total_score = min(scores) if scores else 0.0
        elif metric_conj == "and":
            total_score = float(all(s != 0 for s in scores))
        elif metric_conj == "or":
            total_score = float(any(s != 0 for s in scores))
        else:
            total_score = sum(scores) / len(scores) if scores else 0.0

        if total_score >= 0.999:
            print(f"✅ 1.0")
        elif total_score > 0:
            print(f"🔶 {total_score:.3f}")
        else:
            print(f"❌ 0.0")

        results.append({"task_id": task_id, "score": total_score, "finished": True,
                        "category": task_type, "hardness": hardness, "eval_type": result_type,
                        "metric_scores": scores})

    # Report
    total = len(results)
    if total == 0:
        print("No results")
        return

    scores = [r["score"] for r in results]
    avg_score = sum(scores) / total
    finished = sum(1 for r in results if r.get("finished", False))
    perfect = sum(1 for s in scores if s >= 0.999)

    print(f"\n{'='*60}")
    print(f"📊 Official DA-Code Evaluation Report")
    print(f"{'='*60}")
    print(f"  Total tasks:          {total}")
    print(f"  Tasks finished:       {finished} ({finished/total*100:.1f}%)")
    print(f"  Average Score:        {avg_score:.4f}")
    print(f"  Perfect (score=1.0):  {perfect} ({perfect/total*100:.1f}%)")

    # By category
    print(f"\n  By Category:")
    for cat in ["data insight", "data manipulation", "statistical analysis"]:
        cat_results = [r for r in results if r.get("category") == cat]
        if cat_results:
            cs = sum(r["score"] for r in cat_results) / len(cat_results)
            cf = sum(1 for r in cat_results if r.get("finished"))
            cp = sum(1 for r in cat_results if r["score"] >= 0.999)
            print(f"    {cat:25s}: score={cs:.4f}  finished={cf}/{len(cat_results)}  perfect={cp}")

    # By hardness
    print(f"\n  By Hardness:")
    for h in ["Easy", "Medium", "Hard"]:
        h_results = [r for r in results if r.get("hardness") == h]
        if h_results:
            hs = sum(r["score"] for r in h_results) / len(h_results)
            hf = sum(1 for r in h_results if r.get("finished"))
            print(f"    {h:10s}: score={hs:.4f}  finished={hf}/{len(h_results)}")

    # By eval type
    print(f"\n  By Eval Type:")
    for t in ["text", "csv"]:
        t_results = [r for r in results if r.get("eval_type") == t]
        if t_results:
            ts = sum(r["score"] for r in t_results) / len(t_results)
            tf = sum(1 for r in t_results if r.get("finished"))
            tp = sum(1 for r in t_results if r["score"] >= 0.999)
            print(f"    {t:10s}: score={ts:.4f}  finished={tf}/{len(t_results)}  perfect={tp}")

    # Finished but score 0
    finished_zero = [r["task_id"] for r in results if r.get("finished") and r["score"] < 0.01]
    if finished_zero:
        print(f"\n  ❌ Finished but score=0 ({len(finished_zero)}):")
        for tid in finished_zero[:20]:
            print(f"      {tid}")
        if len(finished_zero) > 20:
            print(f"      ... and {len(finished_zero)-20} more")

    print(f"\n{'='*60}")

    # Save
    output = {
        "num_results": total,
        "average_score": avg_score,
        "average_finished": finished / total if total else 0,
        "perfect": perfect,
        "evaluator": "official_da_code_repo",
        "results": results,
    }
    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to: {args.results_file}")


if __name__ == "__main__":
    main()
