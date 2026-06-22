"""ScienceAgentBench harness: prepare -> manifest -> TriadicRunner -> official eval -> summary."""
import json
import os
from typing import Any, Dict

from triadic_dgm.benchmark.implementations.qwen_llm import OpenAICompatibleClient
from triadic_dgm.benchmark.harness.common.runner import TriadicRunner
from triadic_dgm.benchmark.harness.scienceagentbench.prepare import prepare, DEFAULT_DATA_DIR
from triadic_dgm.benchmark.harness.scienceagentbench.build_manifest import build_manifest
from triadic_dgm.benchmark.harness.scienceagentbench.grade import grade_results, summarize

DEFAULT_RESULTS = os.path.join(DEFAULT_DATA_DIR, "..", "results", "scienceagentbench")

SOLVER_INSTRUCTION = (
    "TARGET PROGRAMMING LANGUAGE IS PYTHON. "
    "Input data is under benchmark/datasets/<dir>/ in the current working directory (see the folder tree). "
    "Load it with the exact relative paths shown. Save the final result to the specified output file path "
    "(creating the directory if needed). Do not modify files under benchmark/."
)


def run(
    limit: int = -1,
    offset_id: int = 0,
    max_workers: int = 1,
    max_inner_retries: int = 2,
    exec_timeout: int = 600,
    eval_timeout: int = 600,
    results_dir: str = "",
    data_dir: str = "",
    resume: bool = True,
    prefer_csv: bool = True,
    gradeable_only: bool = False,
    feasible_only: bool = False,
) -> Dict[str, Any]:
    data_dir = data_dir or os.path.join(DEFAULT_DATA_DIR, "sab")
    results_dir = results_dir or os.path.normpath(DEFAULT_RESULTS)
    os.makedirs(results_dir, exist_ok=True)

    print(f"[sab] data_dir={data_dir}  results_dir={results_dir}")
    prepare(data_dir)
    tasks = build_manifest(data_dir, limit=limit, offset_id=offset_id, prefer_csv=prefer_csv,
                           gradeable_only=gradeable_only, feasible_only=feasible_only)
    print(f"[sab] {len(tasks)} tasks (limit={limit}, prefer_csv={prefer_csv}, "
          f"gradeable_only={gradeable_only}, feasible_only={feasible_only})")

    llm = OpenAICompatibleClient()
    runner = TriadicRunner(
        llm_client=llm,
        max_inner_retries=max_inner_retries,
        exec_timeout=exec_timeout,
        num_workers=max_workers,
        solver_instruction=SOLVER_INSTRUCTION,
    )
    runner.run_all(tasks, results_dir=results_dir, resume=resume)

    run_jsonl = os.path.join(results_dir, "run.jsonl")
    results = [json.loads(l) for l in open(run_jsonl, encoding="utf-8") if l.strip()]
    tasks_by_id = {t["task_id"]: t for t in tasks}
    # merge in manifest fields (output_fname etc.) for any resumed results lacking them
    for r in results:
        t = tasks_by_id.get(r.get("task_id"))
        if t:
            for k in ("output_fname", "eval_script_name", "instance_id", "task_inst",
                      "env_feasible", "needed_deps", "domain"):
                r.setdefault(k, t.get(k))
            r.setdefault("workspace", os.path.join(data_dir, "workspaces", str(t.get("instance_id", ""))))
    graded = grade_results(results, tasks_by_id, data_dir, eval_timeout=eval_timeout)

    with open(os.path.join(results_dir, "graded.jsonl"), "w", encoding="utf-8") as f:
        for r in graded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(graded)
    summary["results_dir"] = results_dir
    summary["reference_evods_paper"] = {"sab_pass_rate": "0.10-0.12"}
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[sab] DONE  pass_rate={summary['pass_rate']}  envfeasible={summary['pass_rate_envfeasible']}  "
          f"({summary['passed']}/{summary['total']}, env_feasible {summary['env_feasible_passed']}/{summary['env_feasible_total']}, "
          f"env_blocked={summary['env_blocked']})  by_category={summary['by_category']}")
    return summary


if __name__ == "__main__":
    run(limit=5)
