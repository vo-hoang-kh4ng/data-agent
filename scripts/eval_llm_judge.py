"""
LLM-Judge Evaluation for DA-Code
=================================
Uses DeepSeek V4 to re-evaluate agent outputs against gold answers.
Handles spelling, formatting, and semantic equivalence that exact-miss misses.

Usage:
    python eval_llm_judge.py
    python eval_llm_judge.py --example_name di-text-001
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFAULT_SANDBOX_DIR = os.path.join(DATA_DIR, "dacode_sandbox")
DEFAULT_GOLD_DIR = os.path.join(DATA_DIR, "dacode_gold", "gold")
DEFAULT_MANIFEST = os.path.join(DATA_DIR, "dacode_unified_manifest.jsonl")
DEFAULT_RESULTS_FILE = os.path.join(DATA_DIR, "dacode_llm_judge_results.json")

# ── DeepSeek Config ──
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"


def _call_deepseek(prompt: str, system: str = "", max_retries: int = 3) -> str:
    """Call DeepSeek V4 API (OpenAI-compatible)."""
    import openai

    client = openai.OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                # Disable thinking mode for faster/cheaper responses
                extra_body={"thinking": {"type": "disabled"}},
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"      ⚠️ API error (attempt {attempt+1}): {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


JUDGE_SYSTEM = """You are an expert evaluation judge for data science tasks. Your job is to compare a model's prediction against the gold answer and determine if they are semantically equivalent.

Rules:
1. Ignore minor spelling differences (e.g., "Chad" vs "Tchad" vs "TCHAD")
2. Ignore formatting differences (e.g., ["Yes"] vs "Yes" vs {"answer": "Yes"})
3. Ignore case differences
4. For numbers: allow ±1% relative difference or ±0.01 absolute difference
5. For lists: consider them matching if they contain the same elements (ignore order if not specified)
6. For country/person names: accept different language variants (e.g., "Germany" = "Allemagne" = "Deutschland")
7. For scientific notation: 1.5e-3 = 0.0015
8. If the prediction contains extra information but the required fields match, still count as correct
9. Pay attention to the EXACT question being asked - make sure the answer addresses it

You must respond with a JSON object:
{
  "score": 0.0 to 1.0 (fraction of correct key-value pairs),
  "reasoning": "brief explanation",
  "key_matches": {"key_name": true/false, ...}
}"""


JUDGE_PROMPT_TEXT = """## Task
Evaluate whether the model's prediction matches the gold answer for this data science task.

## Question
{question}

## Gold Answer
{gold}

## Model Prediction
{prediction}

## Instructions
Compare the gold answer with the model prediction. The gold answer has specific keys and values.
For each key in the gold, check if the prediction has a matching value (using the rules above).
Calculate score = (number of matched keys) / (total keys in gold).

Respond with JSON only:
{{"score": <float>, "reasoning": "<brief explanation>", "key_matches": {{"<key>": <bool>, ...}}}}"""


JUDGE_PROMPT_CSV = """## Task
Evaluate whether the model's CSV output matches the gold CSV for this data science task.

## Question
{question}

## Gold CSV (first 20 rows)
{gold}

## Model CSV (first 20 rows)
{prediction}

## Eval Options
{options}

## Instructions
Compare the CSV outputs. Check if they contain the same data (allowing for formatting, rounding, and order differences as specified in options).
If ignore_order is true, the row order doesn't matter.

Respond with JSON only:
{{"score": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def _load_manifest(path: str) -> List[Dict]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def _parse_json_output(text: str) -> Optional[Dict]:
    """Parse JSON from agent output text."""
    if not text:
        return None
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    import re
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
    # Find first { ... }
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
                try:
                    result = json.loads(text[start:i+1])
                    if isinstance(result, dict):
                        return result
                except (json.JSONDecodeError, TypeError):
                    pass
                start = -1
    return None


def _get_agent_result(task_id: str, sandbox_dir: str) -> Dict:
    """Get agent output for a task."""
    task_sandbox = os.path.join(sandbox_dir, task_id)
    result_json = os.path.join(task_sandbox, "dabench", "result.json")

    if not os.path.exists(result_json):
        result_json = os.path.join(task_sandbox, "result.json")

    if os.path.exists(result_json):
        try:
            data = json.load(open(result_json, "r", encoding="utf-8"))
            return {
                "finished": data.get("finished", True),
                "result": data.get("result", ""),
                "result_files": data.get("result_files", []),
            }
        except:
            pass

    return {"finished": False, "result": "", "result_files": []}


def evaluate_text_with_llm(task: dict, prediction: str, gold_dir: str) -> Dict:
    """Use LLM to judge text output vs gold."""
    task_id = task["task_id"]
    question = task.get("question", "")
    eval_results_cfg = task.get("eval_result", [])
    if not isinstance(eval_results_cfg, list):
        eval_results_cfg = [eval_results_cfg]

    # Load gold
    gold_path = os.path.join(gold_dir, task_id, "result.json")
    if os.path.exists(gold_path):
        gold_data = json.load(open(gold_path, "r", encoding="utf-8"))
    else:
        return {"score": 0.0, "reasoning": "No gold file", "key_matches": {}}

    # Format gold and prediction for judge
    gold_str = json.dumps(gold_data, indent=2, ensure_ascii=False)
    pred_str = prediction[:3000] if prediction else "(empty)"

    prompt = JUDGE_PROMPT_TEXT.format(
        question=question,
        gold=gold_str,
        prediction=pred_str,
    )

    try:
        response = _call_deepseek(prompt, system=JUDGE_SYSTEM)
        # Parse JSON from response
        response = response.strip()
        # Remove markdown wrapper if present
        if response.startswith("```"):
            import re
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if m:
                response = m.group(1)

        result = json.loads(response)
        score = float(result.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return {
            "score": score,
            "reasoning": result.get("reasoning", ""),
            "key_matches": result.get("key_matches", {}),
        }
    except Exception as e:
        print(f"      ⚠️ LLM judge error: {e}")
        return {"score": 0.0, "reasoning": f"Error: {e}", "key_matches": {}}


def evaluate_csv_with_llm(task: dict, task_sandbox: str, gold_dir: str, options: dict) -> Dict:
    """Use LLM to judge CSV output vs gold."""
    task_id = task["task_id"]
    question = task.get("question", "")

    # Find gold CSV
    eval_results_cfg = task.get("eval_result", [])
    if not isinstance(eval_results_cfg, list):
        eval_results_cfg = [eval_results_cfg]
    gold_cfg = eval_results_cfg[0] if eval_results_cfg else {}
    gold_files = gold_cfg.get("file", [])
    if isinstance(gold_files, str):
        gold_files = [gold_files]
    if not gold_files:
        return {"score": 0.0, "reasoning": "No gold CSV specified"}

    gold_csv_name = gold_files[0]
    gold_csv_path = os.path.join(gold_dir, task_id, gold_csv_name)

    # Find output CSV
    output_csv_path = None
    for candidate in [
        os.path.join(task_sandbox, gold_csv_name),
        os.path.join(task_sandbox, "dabench", gold_csv_name),
    ]:
        if os.path.exists(candidate):
            output_csv_path = candidate
            break
    if not output_csv_path:
        # Find any CSV
        if os.path.exists(task_sandbox):
            for f in os.listdir(task_sandbox):
                if f.endswith(".csv") and not f.startswith("_"):
                    output_csv_path = os.path.join(task_sandbox, f)
                    break
        if not output_csv_path:
            # Check result_files
            agent_data = _get_agent_result(task_id, os.path.dirname(task_sandbox))
            for rf in agent_data.get("result_files", []):
                if rf.endswith(".csv"):
                    rf_path = os.path.join(task_sandbox, rf)
                    if os.path.exists(rf_path):
                        output_csv_path = rf_path
                        break

    if not output_csv_path or not os.path.exists(gold_csv_path):
        return {"score": 0.0, "reasoning": "Missing CSV files"}

    # Read first 20 rows of each
    try:
        with open(gold_csv_path, "r", encoding="utf-8", errors="replace") as f:
            gold_lines = f.readlines()[:21]  # header + 20 rows
            gold_csv_text = "".join(gold_lines)
        with open(output_csv_path, "r", encoding="utf-8", errors="replace") as f:
            pred_lines = f.readlines()[:21]
            pred_csv_text = "".join(pred_lines)
    except Exception as e:
        return {"score": 0.0, "reasoning": f"CSV read error: {e}"}

    prompt = JUDGE_PROMPT_CSV.format(
        question=question,
        gold=gold_csv_text,
        prediction=pred_csv_text,
        options=json.dumps(options, indent=2),
    )

    try:
        response = _call_deepseek(prompt, system=JUDGE_SYSTEM)
        response = response.strip()
        if response.startswith("```"):
            import re
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if m:
                response = m.group(1)
        result = json.loads(response)
        score = float(result.get("score", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Error: {e}"}


def main():
    parser = argparse.ArgumentParser(description="LLM-Judge Evaluation for DA-Code")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--sandbox_dir", type=str, default=DEFAULT_SANDBOX_DIR)
    parser.add_argument("--gold_dir", type=str, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--results_file", type=str, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--example_name", type=str, default="")
    parser.add_argument("--only_scored_zero", action="store_true", help="Only re-evaluate tasks that scored 0")
    parser.add_argument("--previous_results", type=str, default=os.path.join(DATA_DIR, "dacode_eval_results.json"),
                        help="Previous eval results to identify score=0 tasks")
    args = parser.parse_args()

    tasks = _load_manifest(args.manifest)
    if args.example_name:
        tasks = [t for t in tasks if args.example_name in t["task_id"]]

    # If only_scored_zero, filter to tasks that scored 0 in previous eval
    zero_tasks = set()
    if args.only_scored_zero and os.path.exists(args.previous_results):
        prev = json.load(open(args.previous_results, "r", encoding="utf-8"))
        zero_tasks = {r["task_id"] for r in prev.get("results", []) if r.get("score", 1.0) < 0.01}
        tasks = [t for t in tasks if t["task_id"] in zero_tasks]
        print(f"🎯 Re-evaluating {len(tasks)} tasks that scored 0")
    else:
        print(f"📊 LLM-Judge evaluating {len(tasks)} tasks")

    print(f"   Model: {DEEPSEEK_MODEL}")
    print(f"   Sandbox: {args.sandbox_dir}")
    print(f"   Gold:    {args.gold_dir}")

    results = []
    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        eval_funcs = task.get("eval_func", [])
        eval_options = task.get("eval_options", [{}])
        if isinstance(eval_funcs, str):
            eval_funcs = [eval_funcs]
        if not isinstance(eval_options, list):
            eval_options = [eval_options]

        print(f"  [{i+1}/{len(tasks)}] {task_id}...", end=" ", flush=True)

        # Get agent result
        agent = _get_agent_result(task_id, args.sandbox_dir)
        if not agent["finished"]:
            print("⏭️ not finished")
            results.append({
                "task_id": task_id,
                "score": 0.0,
                "finished": False,
                "judge_reasoning": "Task not finished",
                "category": task.get("task_category", "unknown"),
                "hardness": task.get("hardness", "unknown"),
                "eval_type": task.get("task_type", "unknown"),
            })
            continue

        # Evaluate based on type
        all_scores = []
        all_reasoning = []

        for idx, func_name in enumerate(eval_funcs):
            options = eval_options[idx] if idx < len(eval_options) else {}

            if func_name in ("compare_text", "compare_number"):
                judge_result = evaluate_text_with_llm(task, agent["result"], args.gold_dir)
            elif func_name == "compare_csv":
                task_sandbox = os.path.join(args.sandbox_dir, task_id)
                judge_result = evaluate_csv_with_llm(task, task_sandbox, args.gold_dir, options)
            else:
                judge_result = evaluate_text_with_llm(task, agent["result"], args.gold_dir)

            all_scores.append(judge_result.get("score", 0.0))
            all_reasoning.append(judge_result.get("reasoning", ""))

        final_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        if final_score >= 0.999:
            icon = "✅"
        elif final_score > 0:
            icon = "🔶"
        else:
            icon = "❌"

        print(f"{icon} {final_score:.3f}")
        if final_score > 0 and final_score < 1.0:
            print(f"      💡 {all_reasoning[0][:100]}")

        results.append({
            "task_id": task_id,
            "score": final_score,
            "finished": True,
            "judge_reasoning": " | ".join(all_reasoning),
            "category": task.get("task_category", "unknown"),
            "hardness": task.get("hardness", "unknown"),
            "eval_type": task.get("task_type", "unknown"),
            "metric_scores": all_scores,
        })

    # Report
    total = len(results)
    avg_score = sum(r["score"] for r in results) / total if total else 0
    finished = sum(1 for r in results if r["finished"])
    perfect = sum(1 for r in results if r["score"] >= 0.999)
    improved = sum(1 for r in results if r["score"] > 0 and r["task_id"] in zero_tasks)

    print(f"\n{'='*60}")
    print(f"📊 LLM-Judge Evaluation Report")
    print(f"{'='*60}")
    print(f"  Total tasks:     {total}")
    print(f"  Finished:        {finished} ({finished/total*100:.1f}%)")
    print(f"  Average Score:   {avg_score:.4f}")
    print(f"  Perfect (1.0):   {perfect} ({perfect/total*100:.1f}%)")
    if args.only_scored_zero:
        print(f"  Improved (was 0, now >0): {improved}/{len(zero_tasks)}")

    # By category
    print(f"\n  By Category:")
    for cat in ["data insight", "data manipulation", "statistical analysis"]:
        cat_results = [r for r in results if r.get("category") == cat]
        if cat_results:
            cs = sum(r["score"] for r in cat_results) / len(cat_results)
            cf = sum(1 for r in cat_results if r["finished"])
            cp = sum(1 for r in cat_results if r["score"] >= 0.999)
            print(f"    {cat:25s}: score={cs:.4f}  finished={cf}/{len(cat_results)}  perfect={cp}")

    # Save
    output = {
        "num_results": total,
        "average_score": avg_score,
        "average_finished": finished / total if total else 0,
        "perfect": perfect,
        "evaluator": DEEPSEEK_MODEL,
        "results": results,
    }

    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {args.results_file}")


if __name__ == "__main__":
    main()
