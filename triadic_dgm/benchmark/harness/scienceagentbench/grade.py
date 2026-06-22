"""ScienceAgentBench grading (local, official eval_programs).

Each SAB task ships an official grader `benchmark/eval_programs/<eval_script_name>`
that defines `eval()` -> (score:int(0/1), info:str). It reads:
  - pred_results/<output_fname>   (the agent's produced output)
  - benchmark/eval_programs/gold_results/<task>_gold.*  (the reference)
both relative to the benchmark *parent* (data/sab/). So grading:
  1. confirm the agent produced <workspace>/<output_fname>;
  2. copy it to <data_dir>/<output_fname>;
  3. run `python benchmark/eval_programs/<eval_script>` with cwd=<data_dir> and
     PYTHONPATH including data_dir + the eval_programs dir (for gpt4_visual_judge);
  4. parse the printed (score, info).

Outcome categories: success | no_output (agent ran but didn't save the file) |
eval_error (grader crashed, e.g. missing dep / visual-judge unavailable) |
eval_fail (grader ran, score 0). Environment-blocked tasks are reported separately
so the solver pass-rate isn't unfairly penalised.
"""
import ast
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

_FIRST_INT_RE = re.compile(r"\b([01])\b")


def _parse_eval_out(out: str) -> Tuple[Optional[int], str]:
    out = (out or "").strip()
    if not out:
        return None, ""
    first = out.splitlines()[0].strip()
    try:
        val = ast.literal_eval(first)
        if isinstance(val, tuple) and val:
            return int(val[0]), str(val[1])
        return int(val), ""
    except Exception:  # noqa: BLE001
        m = _FIRST_INT_RE.search(out)
        if m:
            return int(m.group(1)), out[:200]
        return None, out[:300]


def grade_one(
    result: Dict[str, Any],
    task: Dict[str, Any],
    data_dir: str,
    eval_timeout: int = 600,
) -> Dict[str, Any]:
    bench_root = os.path.join(data_dir, "benchmark")
    output_fname = task.get("output_fname", "")  # e.g. pred_results/clintox_test_pred.csv
    eval_script = task.get("eval_script_name", "")
    ws = result.get("workspace") or os.path.join(data_dir, "workspaces", str(task.get("instance_id", "")))

    produced_src = os.path.join(ws, output_fname)
    produced = os.path.exists(produced_src)
    rec: Dict[str, Any] = {
        "produced": produced, "method": "sab_eval",
        "output_fname": output_fname, "eval_script": eval_script,
    }
    if not produced:
        rec.update(success=False, category="no_output",
                   reason=f"agent did not produce {output_fname}")
        return rec

    # copy agent output to the shared root the eval reads from
    dst = os.path.join(data_dir, output_fname)
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(produced_src, dst)
    except Exception as e:  # noqa: BLE001
        rec.update(success=False, category="copy_error", reason=str(e))
        return rec

    eval_path = os.path.join(bench_root, "eval_programs", eval_script)
    if not os.path.exists(eval_path):
        rec.update(success=False, category="eval_missing", reason=f"no eval script {eval_script}")
        return rec

    env = os.environ.copy()
    # eval imports gpt4_visual_judge from its own dir; also expose data_dir for any helpers
    pp = [os.path.join(bench_root, "eval_programs"), data_dir]
    env["PYTHONPATH"] = os.pathsep.join(pp + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        run = subprocess.run(
            [sys.executable, eval_path],
            capture_output=True, text=True, cwd=data_dir,
            env=env, timeout=eval_timeout,
        )
    except subprocess.TimeoutExpired:
        rec.update(success=False, category="eval_error", reason="eval timeout")
        return rec
    except Exception as e:  # noqa: BLE001
        rec.update(success=False, category="eval_error", reason=f"eval spawn failed: {e}")
        return rec

    score, info = _parse_eval_out(run.stdout)
    rec["eval_stdout"] = (run.stdout or "")[:500]
    if score is None:
        rec.update(success=False, category="eval_error",
                   reason=(run.stderr or "")[:300] or "could not parse eval output")
        return rec
    rec["score"] = score
    rec["eval_info"] = info
    rec.update(success=bool(score), category="success" if score else "eval_fail")
    return rec


def grade_results(results: List[Dict[str, Any]], tasks_by_id: Dict[str, Dict[str, Any]],
                  data_dir: str, eval_timeout: int = 600) -> List[Dict[str, Any]]:
    graded = []
    for r in results:
        t = tasks_by_id.get(r.get("task_id"), {})
        g = grade_one(r, t, data_dir, eval_timeout=eval_timeout)
        rr = dict(r)
        rr["grade"] = g
        graded.append(rr)
    return graded


def summarize(graded: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(graded)
    by_cat: Dict[str, int] = {}
    passed = 0
    env_blocked = 0
    # env-feasible subset: tasks whose gold program needs only deps that are
    # actually installed. pass_rate_envfeasible is the fairest measure of solver
    # ability (excludes tasks blocked by missing scientific deps, not by the solver).
    feasible_total = feasible_passed = 0
    by_cat_feasible: Dict[str, int] = {}
    for r in graded:
        cat = r.get("grade", {}).get("category", "unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1
        if r.get("grade", {}).get("success"):
            passed += 1
        if cat in ("eval_error", "eval_missing", "copy_error"):
            env_blocked += 1
        if r.get("env_feasible", True):
            feasible_total += 1
            by_cat_feasible[cat] = by_cat_feasible.get(cat, 0) + 1
            if r.get("grade", {}).get("success"):
                feasible_passed += 1
    gradeable = total - env_blocked
    return {
        "benchmark": "scienceagentbench",
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "pass_rate_gradeable": round(passed / gradeable, 4) if gradeable else 0.0,
        "gradeable": gradeable,
        "env_blocked": env_blocked,
        "by_category": by_cat,
        "env_feasible_total": feasible_total,
        "env_feasible_passed": feasible_passed,
        "pass_rate_envfeasible": round(feasible_passed / feasible_total, 4) if feasible_total else 0.0,
        "by_category_envfeasible": by_cat_feasible,
    }
