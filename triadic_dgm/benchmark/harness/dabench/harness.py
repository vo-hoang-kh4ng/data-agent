"""DABench harness: prepare -> manifest -> TriadicRunner -> grade -> summary."""
import json
import os
from typing import Any, Dict

from triadic_dgm.benchmark.implementations.qwen_llm import OpenAICompatibleClient
from triadic_dgm.benchmark.harness.common.runner import TriadicRunner
from triadic_dgm.benchmark.harness.dabench.prepare import prepare, DEFAULT_DATA_DIR
from triadic_dgm.benchmark.harness.dabench.build_manifest import build_manifest
from triadic_dgm.benchmark.harness.dabench.grade import grade_results, summarize

DEFAULT_RESULTS = os.path.join(DEFAULT_DATA_DIR, "..", "results", "dabench")

SOLVER_INSTRUCTION = (
    "TARGET PROGRAMMING LANGUAGE IS PYTHON. "
    "The current working directory already contains the input CSV file(s); load them by "
    "their exact filename(s) with pandas (e.g. pd.read_csv('filename.csv')). "
    "Print the final answer in the exact @field[value] format specified. "
    "Do not read or write any other files. Do not print anything except the final answer."
)


def run(
    limit: int = -1,
    offset: int = 0,
    max_workers: int = 1,
    max_inner_retries: int = 3,
    exec_timeout: int = 300,
    results_dir: str = "",
    data_dir: str = "",
    resume: bool = True,
    enable_judge: bool = True,
) -> Dict[str, Any]:
    data_dir = data_dir or os.path.join(DEFAULT_DATA_DIR, "dabench")
    results_dir = results_dir or os.path.normpath(DEFAULT_RESULTS)
    os.makedirs(results_dir, exist_ok=True)

    print(f"[dabench] data_dir={data_dir}  results_dir={results_dir}")
    prepare(data_dir)
    tasks = build_manifest(data_dir, limit=limit, offset=offset)
    print(f"[dabench] {len(tasks)} tasks (limit={limit}, offset={offset})")

    llm = OpenAICompatibleClient()
    runner = TriadicRunner(
        llm_client=llm,
        max_inner_retries=max_inner_retries,
        exec_timeout=exec_timeout,
        num_workers=max_workers,
        solver_instruction=SOLVER_INSTRUCTION,
    )
    runner.run_all(tasks, results_dir=results_dir, resume=resume)

    # Load all results (including resumed ones) and grade.
    run_jsonl = os.path.join(results_dir, "run.jsonl")
    results = [json.loads(l) for l in open(run_jsonl, encoding="utf-8") if l.strip()]
    graded = grade_results(results, llm_client=llm if enable_judge else None)

    with open(os.path.join(results_dir, "graded.jsonl"), "w", encoding="utf-8") as f:
        for r in graded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(graded)
    summary["results_dir"] = results_dir
    summary["reference_evods_paper"] = {"dabench_pass_rate": "0.89-0.91"}
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[dabench] DONE  pass_rate={summary['pass_rate']}  "
          f"({summary['passed']}/{summary['total']})  by_level={summary['by_level']}")
    return summary


if __name__ == "__main__":
    run(limit=5)
