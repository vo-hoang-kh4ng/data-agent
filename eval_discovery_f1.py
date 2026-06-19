"""
Discovery (μ_retrieval) for DA-Code — mirrors the Blackboard paper's Table 2.
=============================================================================
The paper (arXiv:2510.01285, Table 2) reports μ_retrieval = recall / precision /
F1 of file discovery vs the gold relevant files. We measure discovery as the set
of LAKE FILES actually referenced in the agent's generated program (quoted
data-file strings in the code) and report MACRO recall / precision / F1 —
exactly the paper's metric.

Only the paper's three numbers are reported. (Earlier versions also printed
micro scores and a hallucination rate; those are not in the paper and have been
removed.)

Usage:
    python eval_discovery_f1.py --sandbox_dir data/dacode_sandbox_paper_k8_think
"""

import argparse
import glob
import json
import os
import re
from typing import Set

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_SANDBOX_DIR = os.path.join(DATA_DIR, "dacode_sandbox")
DEFAULT_LAKE_DIR = os.path.join(DATA_DIR, "dacode_lake_paper")
DEFAULT_GOLD_FILE = os.path.join(DATA_DIR, "_paper", "task_gold_files.json")
DEFAULT_MANIFEST = os.path.join(DATA_DIR, "dacode_unified_manifest.jsonl")
DEFAULT_RESULTS_FILE = os.path.join(DATA_DIR, "discovery_f1.json")

# quoted strings ending in a data extension — captures read_csv / read_excel / open / path args
_FILE_RE = re.compile(r'''["\']([^"\']+\.(?:csv|xlsx?|json|tsv|txt|ods))["\']''', re.IGNORECASE)

CATEGORY_OF = {}  # task_id -> category, filled from manifest


def _basename(name: str) -> str:
    return re.split(r"[\\/]", name)[-1]


def _load_code(task_id: str, sandbox_dir: str) -> str:
    """Concatenate every generated program we can find for the task."""
    chunks = []
    for pat in (f"{task_id}/_sandbox_run.py", f"{task_id}/sandbox_run.py"):
        for p in glob.glob(os.path.join(sandbox_dir, pat)):
            try:
                chunks.append(open(p, encoding="utf-8", errors="ignore").read())
            except Exception:
                pass
    rj = os.path.join(sandbox_dir, task_id, "dabench", "result.json")
    if os.path.exists(rj):
        try:
            traj = json.load(open(rj, encoding="utf-8")).get("trajectory", [])
            for step in traj:
                if isinstance(step, dict) and isinstance(step.get("code"), str):
                    if "read_csv" in step["code"] or "read_excel" in step["code"]:
                        chunks.append(step["code"])
        except Exception:
            pass
    return "\n".join(chunks)


def _discovered(code: str, lake_lower: dict) -> Set[str]:
    """Lake files whose basename is referenced in the code."""
    found = set()
    for raw in _FILE_RE.findall(code):
        base = _basename(raw)
        if base.lower() in lake_lower:
            found.add(lake_lower[base.lower()])  # canonical lake name
    return found


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox_dir", default=DEFAULT_SANDBOX_DIR)
    ap.add_argument("--lake_dir", default=DEFAULT_LAKE_DIR)
    ap.add_argument("--gold_file", default=DEFAULT_GOLD_FILE)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--results_file", default=DEFAULT_RESULTS_FILE)
    args = ap.parse_args()

    task_ids = [json.loads(l)["task_id"] for l in open(args.manifest, encoding="utf-8") if l.strip()]
    for l in open(args.manifest, encoding="utf-8"):
        if l.strip():
            c = json.loads(l)
            CATEGORY_OF[c["task_id"]] = c.get("task_category") or c.get("task_type", "?")

    lake_lower = {f.lower(): f for f in os.listdir(args.lake_dir)}
    gold_map = json.load(open(args.gold_file, encoding="utf-8"))

    print("📊 Discovery (μ_retrieval) — paper Table 2 metric")
    print(f"   sandbox: {args.sandbox_dir}")
    print(f"   lake:    {args.lake_dir} ({len(lake_lower)} files)")
    print(f"   gold:    {args.gold_file}")

    per_task = []
    macro_p = macro_r = macro_f = 0.0
    n_eval = 0
    for tid in task_ids:
        gold = set(gold_map.get(tid, []))
        code = _load_code(tid, args.sandbox_dir)
        if not code:
            per_task.append({"task_id": tid, "category": CATEGORY_OF.get(tid, "?"),
                             "gold": sorted(gold), "discovered": [],
                             "precision": 0.0, "recall": 0.0, "f1": 0.0,
                             "note": "no generated code found"})
            continue
        disc = _discovered(code, lake_lower)
        tp = len(disc & gold)
        fp = len(disc - gold)
        fn = len(gold - disc)
        p, r, f = _prf(tp, fp, fn)
        per_task.append({"task_id": tid, "category": CATEGORY_OF.get(tid, "?"),
                         "gold": sorted(gold), "discovered": sorted(disc),
                         "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                         "tp": tp, "fp": fp, "fn": fn})
        macro_p += p
        macro_r += r
        macro_f += f
        n_eval += 1

    n = max(n_eval, 1)
    macro = {"precision": macro_p / n, "recall": macro_r / n, "f1": macro_f / n}

    print(f"\n{'=' * 60}")
    print(f"📊 μ_retrieval (macro, {n_eval} tasks with code)")
    print(f"{'=' * 60}")
    print(f"  precision = {macro['precision']:.4f}")
    print(f"  recall    = {macro['recall']:.4f}")
    print(f"  F1        = {macro['f1']:.4f}")
    print(f"\n  Paper Blackboard (Table 2, DA-Code, Gemini-2.5-Pro): "
          f"precision 0.837  recall 0.600  F1 0.643")
    print(f"{'=' * 60}")

    out = {
        "metric": "macro_recall_precision_f1 (paper Table 2 μ_retrieval)",
        "sandbox_dir": args.sandbox_dir,
        "lake_dir": args.lake_dir,
        "num_tasks": len(task_ids),
        "num_with_code": n_eval,
        "macro": {k: round(v, 4) for k, v in macro.items()},
        "paper_reference": {"precision": 0.837, "recall": 0.600, "f1": 0.643},
        "per_task": per_task,
    }
    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {args.results_file}")


if __name__ == "__main__":
    main()
