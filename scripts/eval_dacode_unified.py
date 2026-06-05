"""
DA-Code Unified Evaluation Script
===================================
Evaluates agent outputs against gold answers for the 91 retained DA-Code tasks.
Faithful to the official DA-Code evaluator (github.com/yiyihum/da-code).

Usage:
    python eval_dacode_unified.py
    python eval_dacode_unified.py --example_name di-text-001
    python eval_dacode_unified.py --retry_failed
"""

import argparse
import json
import math
import os
import sys
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFAULT_SANDBOX_DIR = os.path.join(DATA_DIR, "dacode_sandbox")
DEFAULT_GOLD_DIR = os.path.join(DATA_DIR, "dacode_gold", "gold")
DEFAULT_MANIFEST = os.path.join(DATA_DIR, "dacode_unified_manifest.jsonl")
DEFAULT_RESULTS_FILE = os.path.join(DATA_DIR, "dacode_eval_results.json")


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def compare_text(output_str: str, gold_entries: list, options: dict) -> float:
    """
    Compare text/JSON output against gold.

    gold_entries format: [{"number": [{"key1": ["val1"], "key2": ["val2"], ...}]}]
    output_str: the agent's text output (should be JSON or contain JSON)

    Returns: score (0.0-1.0)
    """
    # Parse gold
    gold_dict = {}
    for entry in gold_entries:
        if "number" in entry:
            for item in entry["number"]:
                gold_dict.update(item)

    if not gold_dict:
        return 0.0

    # Parse output
    output_dict = _parse_json_output(output_str)
    if output_dict is None:
        return 0.0

    # Compare each key
    ignore_order = options.get("ignore_order", False)
    total_keys = len(gold_dict)
    matched_keys = 0

    for key, gold_values in gold_dict.items():
        # Try exact key match first, then case-insensitive
        pred_values = output_dict.get(key)
        if pred_values is None:
            # Case-insensitive search
            for ok, ov in output_dict.items():
                if ok.strip().lower() == key.strip().lower():
                    pred_values = ov
                    break

        if pred_values is None:
            continue

        # Normalize to lists
        if not isinstance(gold_values, list):
            gold_values = [gold_values]
        if not isinstance(pred_values, list):
            pred_values = [pred_values]

        # Flatten nested dicts in pred_values (agent may output [{"Country": "Niger"}, ...])
        flat_pred = []
        for v in pred_values:
            if isinstance(v, dict):
                # Take the first value from the dict
                for dv in v.values():
                    flat_pred.append(dv)
                    break  # only first value
            else:
                flat_pred.append(v)
        pred_values = flat_pred

        if _values_match(pred_values, gold_values, ignore_order):
            matched_keys += 1

    if total_keys == 0:
        return 0.0

    return matched_keys / total_keys


def compare_csv(output_path: str, gold_path: str, options: dict) -> float:
    """
    Compare CSV output against gold CSV.

    Returns: score (0.0-1.0)
    """
    import pandas as pd
    import numpy as np

    if not os.path.exists(output_path):
        return 0.0
    if not os.path.exists(gold_path):
        return 0.0

    try:
        df_out = pd.read_csv(output_path)
        df_gold = pd.read_csv(gold_path)
    except Exception as e:
        print(f"      CSV read error: {e}")
        return 0.0

    # Shape check
    if df_out.shape != df_gold.shape:
        # Allow extra columns but require same rows
        if len(df_out) != len(df_gold):
            return 0.0

    # Handle condition_cols: only compare specific columns
    condition_cols = options.get("condition_cols", None)
    if condition_cols:
        # Convert 1-indexed to column names
        if isinstance(condition_cols, list) and all(isinstance(c, int) for c in condition_cols):
            col_names = [df_gold.columns[i] for i in condition_cols if i < len(df_gold.columns)]
        else:
            col_names = condition_cols
        # Filter to condition columns
        common_cols = [c for c in col_names if c in df_out.columns and c in df_gold.columns]
        if not common_cols:
            return 0.0
        df_out_cmp = df_out[common_cols]
        df_gold_cmp = df_gold[common_cols]
    else:
        # Use common columns
        common_cols = [c for c in df_gold.columns if c in df_out.columns]
        if not common_cols:
            return 0.0
        df_out_cmp = df_out[common_cols]
        df_gold_cmp = df_gold[common_cols]

    # Sort if ignore_order
    if options.get("ignore_order", False):
        sort_cols = list(df_gold_cmp.columns)
        try:
            df_out_cmp = df_out_cmp.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
            df_gold_cmp = df_gold_cmp.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        except Exception:
            pass

    # Compare values
    total_cells = 0
    matched_cells = 0

    for col in df_gold_cmp.columns:
        for i in range(min(len(df_out_cmp), len(df_gold_cmp))):
            total_cells += 1
            gold_val = df_gold_cmp.iloc[i][col]
            pred_val = df_out_cmp.iloc[i][col]

            if _cell_match(pred_val, gold_val):
                matched_cells += 1

    if total_cells == 0:
        return 0.0

    return matched_cells / total_cells


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _parse_json_output(text: str) -> Optional[Dict]:
    """Parse JSON from agent output text."""
    if not text:
        return None

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from ```json block
    import re
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Try finding first { ... } in text
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i+1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict):
                        return result
                except (json.JSONDecodeError, TypeError):
                    pass
                start = -1

    return None


def _values_match(pred_values: list, gold_values: list, ignore_order: bool = False) -> bool:
    """Check if predicted values match gold values."""
    if ignore_order:
        pred_set = set(str(v).strip().lower() for v in pred_values)
        gold_set = set(str(v).strip().lower() for v in gold_values)
        return pred_set == gold_set

    if len(pred_values) != len(gold_values):
        return False

    for pv, gv in zip(pred_values, gold_values):
        if not _single_value_match(pv, gv):
            return False
    return True


def _single_value_match(pred, gold) -> bool:
    """Match a single value (fuzzy for numbers, exact for strings)."""
    # Try numeric comparison
    try:
        pred_num = float(pred)
        gold_num = float(gold)
        if abs(pred_num - gold_num) < 0.01:
            return True
        # Relative tolerance for large numbers
        if gold_num != 0 and abs(pred_num - gold_num) / abs(gold_num) < 0.01:
            return True
        return False
    except (ValueError, TypeError):
        pass

    # String comparison (case-insensitive)
    pred_str = str(pred).strip().lower()
    gold_str = str(gold).strip().lower()
    return pred_str == gold_str


def _cell_match(pred, gold) -> bool:
    """Match a single cell value."""
    return _single_value_match(pred, gold)


def _load_manifest(path: str) -> List[Dict]:
    """Load manifest JSONL."""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_task(task: dict, sandbox_dir: str, gold_dir: str) -> Dict:
    """
    Evaluate a single task.

    Returns: {
        "task_id": str,
        "score": float (0.0-1.0),
        "finished": bool,
        "has_output": bool,
        "has_gold": bool,
        "error": str or None,
    }
    """
    task_id = task["task_id"]
    eval_funcs = task.get("eval_func", [])
    eval_results_cfg = task.get("eval_result", [])
    eval_options = task.get("eval_options", [{}])

    # Normalize to lists
    if isinstance(eval_funcs, str):
        eval_funcs = [eval_funcs]
    if not isinstance(eval_results_cfg, list):
        eval_results_cfg = [eval_results_cfg]
    if not isinstance(eval_options, list):
        eval_options = [eval_options]
    # Pad options
    while len(eval_options) < len(eval_funcs):
        eval_options.append({})

    # Check gold exists
    task_gold_dir = os.path.join(gold_dir, task_id)
    if not os.path.exists(task_gold_dir):
        return {
            "task_id": task_id,
            "score": 0.0,
            "finished": False,
            "has_output": False,
            "has_gold": False,
            "error": "Gold dir not found",
        }

    # Check agent output — support multiple formats
    task_sandbox = os.path.join(sandbox_dir, task_id.replace("/", "_"))

    # Format 1: dabench/result.json (runner.py format)
    result_json_path = os.path.join(task_sandbox, "dabench", "result.json")
    # Format 2: result.json directly in sandbox
    if not os.path.exists(result_json_path):
        result_json_path = os.path.join(task_sandbox, "result.json")

    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            finished = result_data.get("finished", True)
            agent_result = result_data.get("result", "")
        except Exception as e:
            return {
                "task_id": task_id,
                "score": 0.0,
                "finished": False,
                "has_output": True,
                "has_gold": True,
                "error": f"Result parse error: {e}",
            }
    else:
        # Format 3: raw files in sandbox (no result.json)
        # Check if sandbox has any output files
        has_files = False
        agent_result = ""
        finished = False

        if os.path.exists(task_sandbox):
            # Look for JSON output files
            json_files = []
            for f in sorted(os.listdir(task_sandbox)):
                if f.startswith("_") or f == "dabench":
                    continue
                fpath = os.path.join(task_sandbox, f)
                if os.path.isfile(fpath):
                    has_files = True
                    if f.endswith(".json"):
                        json_files.append(fpath)

            # Read the last JSON file as output (or first if only one)
            if json_files:
                try:
                    with open(json_files[-1], "r", encoding="utf-8") as fh:
                        agent_result = fh.read().strip()
                except Exception:
                    pass
            # Also try _sandbox_run.py stdout capture
            elif os.path.exists(os.path.join(task_sandbox, "_output.txt")):
                try:
                    with open(os.path.join(task_sandbox, "_output.txt"), "r", encoding="utf-8") as fh:
                        agent_result = fh.read().strip()
                except Exception:
                    pass

        if not has_files:
            return {
                "task_id": task_id,
                "score": 0.0,
                "finished": False,
                "has_output": False,
                "has_gold": True,
                "error": "No output yet",
            }
        # Sandbox has files but no result.json — treat as finished for CSV tasks
        finished = True

    if not finished and not agent_result:
        return {
            "task_id": task_id,
            "score": 0.0,
            "finished": False,
            "has_output": True,
            "has_gold": True,
            "error": "Task not finished",
        }

    # Evaluate based on eval_funcs
    scores = []

    for idx, func_name in enumerate(eval_funcs):
        options = eval_options[idx] if idx < len(eval_options) else {}
        gold_cfg = eval_results_cfg[idx] if idx < len(eval_results_cfg) else {}

        try:
            if func_name == "compare_text":
                score = _eval_text_task(agent_result, gold_cfg, task_gold_dir, options)
            elif func_name == "compare_csv":
                score = _eval_csv_task(agent_result, gold_cfg, task_sandbox, task_gold_dir, options)
            elif func_name == "compare_number":
                score = _eval_text_task(agent_result, gold_cfg, task_gold_dir, options)
            else:
                # Unknown metric — try text comparison as fallback
                score = _eval_text_task(agent_result, gold_cfg, task_gold_dir, options)

            scores.append(max(0.0, min(1.0, score)))
        except Exception as e:
            print(f"      ⚠️ Eval error for {func_name}: {e}")
            scores.append(0.0)

    # Aggregate: average of all metric scores
    final_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "task_id": task_id,
        "score": final_score,
        "finished": finished,
        "has_output": True,
        "has_gold": True,
        "error": None,
        "metric_scores": scores,
    }


def _eval_text_task(agent_result: str, gold_cfg: dict, gold_dir: str, options: dict) -> float:
    """Evaluate a text/number task."""
    # gold_cfg: {"number": [{"key": ["val"], ...}]}
    # or {"number": [{"answer": ["Yes"]}, {"answer": ["B"]}]}
    if "number" not in gold_cfg:
        return 0.0
    return compare_text(agent_result, [gold_cfg], options)


def _eval_csv_task(agent_result, gold_cfg: dict, task_sandbox: str, gold_dir: str, options: dict) -> float:
    """Evaluate a CSV task."""
    # Find gold CSV
    gold_files = gold_cfg.get("file", [])
    if isinstance(gold_files, str):
        gold_files = [gold_files]

    if not gold_files:
        return 0.0

    gold_csv_name = gold_files[0]
    gold_csv_path = os.path.join(gold_dir, gold_csv_name)

    # Find output CSV — check multiple locations
    output_csv_path = None

    # Location 1: directly in sandbox
    candidate = os.path.join(task_sandbox, gold_csv_name)
    if os.path.exists(candidate):
        output_csv_path = candidate

    # Location 2: in dabench dir
    if not output_csv_path:
        candidate = os.path.join(task_sandbox, "dabench", gold_csv_name)
        if os.path.exists(candidate):
            output_csv_path = candidate

    # Location 3: any CSV in sandbox
    if not output_csv_path:
        for f in os.listdir(task_sandbox):
            if f.endswith(".csv") and not f.startswith("_"):
                output_csv_path = os.path.join(task_sandbox, f)
                break

    if not output_csv_path:
        return 0.0

    return compare_csv(output_csv_path, gold_csv_path, options)


# ═══════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════

def print_report(results: List[Dict]):
    """Print evaluation report matching paper's format."""
    total = len(results)
    if total == 0:
        print("No results to report.")
        return

    scores = [r["score"] for r in results]
    finished = [r["finished"] for r in results]
    avg_score = sum(scores) / total
    avg_finished = sum(finished) / total
    perfect = sum(1 for s in scores if s >= 0.999)

    print(f"\n{'='*60}")
    print(f"📊 DA-Code Evaluation Report (91 Retained Tasks)")
    print(f"{'='*60}")
    print(f"  Total tasks:          {total}")
    print(f"  Tasks with output:    {sum(r['has_output'] for r in results)}")
    print(f"  Tasks finished:       {sum(finished)} ({avg_finished*100:.1f}%)")
    print(f"  Average Score:        {avg_score:.4f}")
    print(f"  Perfect (score=1.0):  {perfect} ({perfect/total*100:.1f}%)")
    print()

    # By task category
    print("─" * 40)
    print("  By Category:")
    categories = {}
    for r in results:
        # Get category from manifest
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r)

    for cat in ["data insight", "data manipulation", "statistical analysis"]:
        cat_results = categories.get(cat, [])
        if cat_results:
            cat_scores = [r["score"] for r in cat_results]
            cat_finished = sum(r["finished"] for r in cat_results)
            cat_perfect = sum(1 for s in cat_scores if s >= 0.999)
            print(f"    {cat:25s}: score={sum(cat_scores)/len(cat_scores):.4f}  "
                  f"finished={cat_finished}/{len(cat_results)}  "
                  f"perfect={cat_perfect}")
    print()

    # By hardness
    print("─" * 40)
    print("  By Hardness:")
    hardness_groups = {}
    for r in results:
        h = r.get("hardness", "unknown")
        hardness_groups.setdefault(h, []).append(r)

    for h in ["Easy", "Medium", "Hard"]:
        h_results = hardness_groups.get(h, [])
        if h_results:
            h_scores = [r["score"] for r in h_results]
            print(f"    {h:10s}: score={sum(h_scores)/len(h_scores):.4f}  "
                  f"finished={sum(r['finished'] for r in h_results)}/{len(h_results)}")
    print()

    # By eval type
    print("─" * 40)
    print("  By Eval Type:")
    type_groups = {}
    for r in results:
        t = r.get("eval_type", "unknown")
        type_groups.setdefault(t, []).append(r)

    for t in sorted(type_groups.keys()):
        t_results = type_groups[t]
        t_scores = [r["score"] for r in t_results]
        print(f"    {t:10s}: score={sum(t_scores)/len(t_scores):.4f}  "
              f"finished={sum(r['finished'] for r in t_results)}/{len(t_results)}")

    # Tasks with no output
    no_output = [r["task_id"] for r in results if not r["has_output"]]
    if no_output:
        print(f"\n  ⚠️ No output ({len(no_output)} tasks):")
        for tid in no_output[:10]:
            print(f"      {tid}")
        if len(no_output) > 10:
            print(f"      ... and {len(no_output)-10} more")

    # Failed tasks (finished but score=0)
    failed = [r for r in results if r["has_output"] and r["finished"] and r["score"] < 0.01]
    if failed:
        print(f"\n  ❌ Finished but score=0 ({len(failed)} tasks):")
        for r in failed[:10]:
            print(f"      {r['task_id']}")

    print(f"\n{'='*60}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate DA-Code results")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--sandbox_dir", type=str, default=DEFAULT_SANDBOX_DIR)
    parser.add_argument("--gold_dir", type=str, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--results_file", type=str, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--example_name", type=str, default="", help="Evaluate specific task")
    parser.add_argument("--retry_failed", action="store_true", help="Re-evaluate failed tasks")
    args = parser.parse_args()

    # Load manifest
    tasks = _load_manifest(args.manifest)

    # Filter tasks
    if args.example_name:
        tasks = [t for t in tasks if args.example_name in t["task_id"]]

    print(f"📊 Evaluating {len(tasks)} tasks")
    print(f"   Sandbox: {args.sandbox_dir}")
    print(f"   Gold:    {args.gold_dir}")

    results = []
    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        print(f"  [{i+1}/{len(tasks)}] {task_id}...", end=" ")

        result = evaluate_task(task, args.sandbox_dir, args.gold_dir)

        # Enrich with metadata
        result["category"] = task.get("task_category", "unknown")
        result["hardness"] = task.get("hardness", "unknown")
        result["eval_type"] = task.get("task_type", "unknown")

        if result["score"] >= 0.999:
            print(f"✅ 1.0")
        elif result["score"] > 0:
            print(f"🔶 {result['score']:.3f}")
        elif result["has_output"]:
            print(f"❌ 0.0")
        else:
            print(f"⏭️ no output")

        results.append(result)

    # Print report
    print_report(results)

    # Save results
    output = {
        "num_results": len(results),
        "average_score": sum(r["score"] for r in results) / len(results) if results else 0,
        "average_finished": sum(r["finished"] for r in results) / len(results) if results else 0,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {args.results_file}")


if __name__ == "__main__":
    main()
