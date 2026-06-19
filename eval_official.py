"""
Official DA-Code Evaluator Wrapper
====================================
Thin wrapper that calls the OFFICIAL da-code-repo evaluator *directly* and
reports EXACTLY what da-code-repo/evaluate.py reports — same metrics, same
group-by breakdowns, same JSON schema. Nothing of ours is added on top.

Scoring   → da-code-repo Evaluator.evaluate()              (the official entrypoint)
Reporting → mirrors da-code-repo/evaluate.py run_evaluation (verbatim logic)

da-code-repo must sit one level up, as a sibling of this project:
    <parent>/
      da-code-repo/          # official repo (provides da_agent.evaluators)
      data-agent/            # this project

Usage:
    python eval_official.py --sandbox_dir data/dacode_sandbox_paper_k8_think
    python eval_official.py --sandbox_dir ... --example_name dm-csv-034
"""

import argparse
import json
import os
import sys
import tempfile

import pandas as pd

# ── Make da-code-repo importable (sibling dir) ──
DA_CODE_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "da-code-repo")
if DA_CODE_REPO not in sys.path:
    sys.path.insert(0, DA_CODE_REPO)
from da_agent.evaluators.evaluation import Evaluator  # noqa: E402  (official, unmodified)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFAULT_SANDBOX_DIR = os.path.join(DATA_DIR, "dacode_sandbox")
DEFAULT_GOLD_DIR = os.path.join(DATA_DIR, "dacode_gold", "gold")
DEFAULT_EVAL_CONFIG = os.path.join(PROJECT_ROOT, "eval", "configs", "eval_all.jsonl")
DEFAULT_RESULTS_FILE = os.path.join(DATA_DIR, "dacode_official_eval_results.json")
TIMEOUT_SECONDS = 30


def _select_config(eval_config: str, sandbox_dir: str, example_name: str) -> str:
    """Keep only eval configs whose task has a result in the sandbox (and matches --example_name).

    This is purely to avoid the official Evaluator's 'does not exist' skip-print noise;
    the reported numbers are identical to passing the full config (the official skips
    missing tasks itself). Selecting only present tasks yields the same num_results/averages.
    """
    keep = []
    with open(eval_config, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cfg = json.loads(line)
            tid = cfg["id"]
            if example_name and example_name not in tid:
                continue
            if not os.path.exists(os.path.join(sandbox_dir, tid, "dabench", "result.json")):
                continue
            keep.append(cfg)
    if not keep:
        sys.exit("No task with an existing dabench/result.json matched the selection.")
    if example_name:
        print(f"   (filtered to {len(keep)} task(s) matching '{example_name}')")
    tmp = os.path.join(tempfile.gettempdir(), "_dacode_eval_present.jsonl")
    with open(tmp, "w", encoding="utf-8") as w:
        for cfg in keep:
            w.write(json.dumps(cfg) + "\n")
    return tmp


def main():
    parser = argparse.ArgumentParser(
        description="Official DA-Code Evaluation (mirrors da-code-repo/evaluate.py output)")
    parser.add_argument("--sandbox_dir", type=str, default=DEFAULT_SANDBOX_DIR,
                        help="Agent output dir (one subdir per task_id, each with dabench/result.json)")
    parser.add_argument("--gold_dir", type=str, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--eval_config", type=str, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--results_file", type=str, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--example_name", type=str, default="",
                        help="Evaluate only tasks whose id contains this substring")
    args = parser.parse_args()

    if not os.path.exists(args.eval_config):
        sys.exit(f"Eval config not found: {args.eval_config}")
    if not os.path.isdir(args.sandbox_dir):
        sys.exit(f"Sandbox dir not found: {args.sandbox_dir}")

    config_path = _select_config(args.eval_config, args.sandbox_dir, args.example_name)

    print("📊 Official DA-Code Evaluation")
    print(f"   sandbox:   {args.sandbox_dir}")
    print(f"   gold:      {args.gold_dir}")
    print(f"   eval cfg:  {config_path}")
    print(f"   evaluator: da-code-repo/da_agent/evaluators/evaluation.py (OFFICIAL, unmodified)")

    # ── THE official call: nothing of ours touches the scoring ──
    evaluator = Evaluator(output_dir=args.sandbox_dir, gold_dir=args.gold_dir,
                          timeout_seconds=TIMEOUT_SECONDS)
    results_infos = evaluator.evaluate(env_config=config_path)

    # ── Report: verbatim logic from da-code-repo/evaluate.py run_evaluation ──
    num_results = len(results_infos)
    if num_results == 0:
        print("No results.")
        return

    scores = [result['total_score'] for result in results_infos]
    finished = [result['finished'] for result in results_infos]
    result_type = [result['result_type'] for result in results_infos]
    types = [result['task'] for result in results_infos]
    types = ["machine learning" if "machine learning" in t else t for t in types]
    hardness = [result['hardness'] for result in results_infos]
    eda = ["data insight", "data manipulation", "data visualization", "statistical analysis"]
    big_types = ["EDA" if t in eda else t for t in types]
    plot = ["line", "pie", "bar", "scatter"]
    result_type = ["plot" if t in plot else t for t in result_type]
    df = pd.DataFrame({"type": types, "score": scores, "finished": finished,
                       "hardness": hardness, "big_type": big_types, "result_type": result_type})

    average_score = sum(scores) / num_results
    average_finished = sum(finished) / num_results
    results_json = {"num_results": num_results, "average_score": average_score,
                    "results": results_infos, "average_finished": average_finished}
    print(f"Number of results: {num_results}")
    print(f"Average score: {average_score}")
    print(f"Average finished: {average_finished}")

    print("====================================")
    print(df.groupby("type").agg({"score": "mean", "finished": "mean"}))
    print("-------------------------------")
    print(df.groupby("hardness").agg({"score": "mean", "finished": "mean"}))
    print("-------------------------------")
    print(df.groupby("big_type").agg({"score": "mean", "finished": "mean"}))
    print("-------------------------------")
    print(df.groupby("result_type").agg({"score": "mean", "finished": "mean"}))

    os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)
    print(f"\n💾 Results saved to: {args.results_file}")


if __name__ == "__main__":
    main()
