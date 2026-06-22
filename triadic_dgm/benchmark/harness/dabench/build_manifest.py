"""Build the DABench task manifest for the runner.

For each DABench question we produce a runner task:
    {
      task_id:    "dabench-<id>",
      description: <question + constraints + DABench format-prompting contract>,
      workspace:  <abs dir containing the question's CSV>,
      gold:       [[field, value], ...]   (carried through for the grader),
      question, level,
    }

DABench uses *format prompting*: the agent is told the exact answer schema
`@field[value]` and must print it, so the grader can parse stdout programmatically
(no judge needed in the common case).
"""
import json
import os
import shutil
from typing import Dict, List


def build_description(q: Dict) -> str:
    """Question text + DABench format-prompting contract."""
    file_name = q.get("file_name", "")
    constraints = q.get("constraints", "")
    fmt = q.get("format", "")
    return "\n".join([
        "DATA ANALYSIS TASK (DABench).",
        "",
        f"Question: {q.get('question','')}",
        "",
        f"Constraints: {constraints}" if constraints else "",
        "",
        f"You have access to a CSV file named '{file_name}' in the current working directory.",
        "Write a SELF-CONTAINED Python program that loads it with pandas, computes the",
        "answer, and PRINTS the final answer(s) in the exact format below. Do not save any",
        "files; do not print explanations or intermediate values.",
        "",
        f"Required output format: {fmt}",
        "",
        "CRITICAL ANSWER-FORMAT RULES:",
        "  - Print ONLY the final answer, one per line, in the form @field_name[value].",
        "  - Use the exact field names from the format spec above.",
        "  - Round every numeric value to 2 decimal places (e.g. 34.65, not 34.6521).",
        "  - Example: @mean_fare[34.65]",
        "  - Do not wrap the answer in code fences. Print it as plain text to stdout.",
    ])


def build_manifest(
    data_dir: str,
    workspaces_root: str = "",
    limit: int = -1,
    offset: int = 0,
) -> List[Dict]:
    qpath = os.path.join(data_dir, "questions.jsonl")
    lpath = os.path.join(data_dir, "labels.jsonl")
    tables_dir = os.path.join(data_dir, "tables")
    questions = [json.loads(l) for l in open(qpath, encoding="utf-8") if l.strip()]
    labels = {}
    for line in open(lpath, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        labels[obj["id"]] = obj.get("common_answers", [])

    workspaces_root = workspaces_root or os.path.join(data_dir, "workspaces")
    os.makedirs(workspaces_root, exist_ok=True)

    tasks: List[Dict] = []
    # Stable iteration by id so --limit/--offset are reproducible.
    for q in sorted(questions, key=lambda x: x["id"]):
        if q["id"] < offset:
            continue
        ws = os.path.abspath(os.path.join(workspaces_root, str(q["id"])))
        os.makedirs(ws, exist_ok=True)
        fn = q.get("file_name", "")
        src = os.path.join(tables_dir, fn)
        if fn and os.path.exists(src):
            dst = os.path.join(ws, fn)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
        else:
            # If the referenced table is missing, seed the workspace with ALL tables
            # so the agent still has data to work with (matches DABench's data-lake setting).
            for tf in os.listdir(tables_dir):
                if tf.endswith(".csv"):
                    tdst = os.path.join(ws, tf)
                    if not os.path.exists(tdst):
                        shutil.copy(os.path.join(tables_dir, tf), tdst)
        tasks.append({
            "task_id": f"dabench-{q['id']}",
            "description": build_description(q),
            "workspace": ws,
            "gold": labels.get(q["id"], []),
            "question": q.get("question", ""),
            "level": q.get("level", ""),
        })
        if limit >= 0 and len(tasks) >= limit:
            break
    return tasks


if __name__ == "__main__":
    import sys
    from triadic_dgm.benchmark.harness.dabench.prepare import prepare
    dd = prepare()
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ts = build_manifest(dd, limit=lim)
    print(f"built {len(ts)} tasks; sample:\n{json.dumps(ts[0], ensure_ascii=False)[:600]}")
