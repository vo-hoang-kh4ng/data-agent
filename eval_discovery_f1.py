"""
Discovery F1 (μ_retrieval) for DA-Code — mirrors the Blackboard paper's Table 2.
=============================================================================

The paper reports μ_retrieval = recall / precision / F1 of file discovery vs the
gold relevant files. We measure discovery as the set of LAKE FILES actually
referenced in the agent's generated program (read_csv / read_excel / open /
os.path targets, captured by scanning quoted data-file strings in the code).
This is the downstream, reproducible signal of whether the agent found usable
data — robust across the global (real discovery) and per-subtask (preload)
settings, and independent of which read API the code uses.

For each task:
  discovered   = lake files whose basename appears in the generated code
  hallucinated = data-file names in the code that resolve to NO lake file
                 (filename hallucination — the global failure mode)
  gold         = this task's input data files in the lake (task_gold_files.json)
  precision    = |discovered ∩ gold| / |discovered|
  recall       = |discovered ∩ gold| / |gold|
  F1           = harmonic mean

Macro = mean of per-task scores; Micro = pooled TP/FP/FN. Hallucination rate =
fraction of tasks with ≥1 hallucinated name.

Usage:
    python eval_discovery_f1.py --sandbox_dir data/dacode_sandbox
    python eval_discovery_f1.py --sandbox_dir data/dacode_sandbox_persubtask \
        --results_file data/discovery_f1_persubtask.json
"""

import argparse
import glob
import json
import os
import re
from typing import List, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFAULT_SANDBOX_DIR = os.path.join(DATA_DIR, "dacode_sandbox")
DEFAULT_LAKE_DIR = os.path.join(DATA_DIR, "dacode_lake_paper")
DEFAULT_GOLD_FILE = os.path.join(DATA_DIR, "_paper", "task_gold_files.json")
DEFAULT_MANIFEST = os.path.join(DATA_DIR, "dacode_unified_manifest.jsonl")
DEFAULT_RESULTS_FILE = os.path.join(DATA_DIR, "discovery_f1.json")

DATA_EXTS = (".csv", ".xls", ".xlsx", ".json", ".tsv", ".txt", ".ods")
# quoted strings ending in a data extension — captures read_csv / open / path args
_FILE_RE = re.compile(r'''["\']([^"\']+\.(?:csv|xlsx?|json|tsv|txt|ods))["\']''', re.IGNORECASE)

CATEGORY_OF = {}  # task_id -> category, filled from manifest


def _basename(name: str) -> str:
    return re.split(r"[\\/]", name)[-1]


def _load_code(task_id: str, sandbox_dir: str) -> str:
    """Concatenate every generated program we can find for the task."""
    chunks: List[str] = []
    for pat in (f"{task_id}/_sandbox_run.py", f"{task_id}/sandbox_run.py"):
        for p in glob.glob(os.path.join(sandbox_dir, pat)):
            try:
                chunks.append(open(p, encoding="utf-8", errors="ignore").read())
            except Exception:
                pass
    # Fallback: pull code blobs from the trajectory in result.json
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


def _extract_referenced(code: str, lake_lower: dict) -> Tuple[Set[str], Set[str]]:
    """Return (discovered lake names, hallucinated raw names)."""
    discovered: Set[str] = set()
    hallucinated: Set[str] = set()
    for raw in _FILE_RE.findall(code):
        base = _basename(raw)
        if base.lower() in lake_lower:
            discovered.add(lake_lower[base.lower()])  # canonical lake name
        else:
            # ignore obviously non-data noise, keep real-looking hallucinations
            if any(base.lower().endswith(ext) for ext in DATA_EXTS):
                hallucinated.add(base)
    return discovered, hallucinated


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
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

    lake_files = set(os.listdir(args.lake_dir))
    lake_lower = {f.lower(): f for f in lake_files}
    gold_map = json.load(open(args.gold_file, encoding="utf-8"))

    print(f"📊 Discovery F1 (μ_retrieval)")
    print(f"   sandbox: {args.sandbox_dir}")
    print(f"   lake:    {args.lake_dir} ({len(lake_files)} files)")
    print(f"   gold:    {args.gold_file}")

    per_task = []
    # micro pools
    tp = fp = fn = 0
    macro_p = macro_r = macro_f = 0.0
    n_eval = 0
    n_halluc = 0

    for tid in task_ids:
        gold = set(gold_map.get(tid, []))
        code = _load_code(tid, args.sandbox_dir)
        if not code:
            # no code at all → nothing discovered
            per_task.append({"task_id": tid, "category": CATEGORY_OF.get(tid, "?"),
                             "gold": sorted(gold), "discovered": [], "hallucinated": [],
                             "precision": 0.0, "recall": 0.0, "f1": 0.0,
                             "note": "no generated code found"})
            continue
        discovered, halluc = _extract_referenced(code, lake_lower)
        tp_t = len(discovered & gold)
        fp_t = len(discovered - gold)
        fn_t = len(gold - discovered)
        p, r, f = _prf(tp_t, fp_t, fn_t)
        per_task.append({"task_id": tid, "category": CATEGORY_OF.get(tid, "?"),
                         "gold": sorted(gold), "discovered": sorted(discovered),
                         "hallucinated": sorted(halluc),
                         "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                         "tp": tp_t, "fp": fp_t, "fn": fn_t})
        tp += tp_t; fp += fp_t; fn += fn_t
        macro_p += p; macro_r += r; macro_f += f
        n_eval += 1
        if halluc:
            n_halluc += 1

    n = max(n_eval, 1)
    micro_p, micro_r, micro_f = _prf(tp, fp, fn)
    macro = {"precision": macro_p / n, "recall": macro_r / n, "f1": macro_f / n}

    print(f"\n{'='*60}")
    print(f"📊 Discovery F1 Report ({len(task_ids)} tasks, {n_eval} with code)")
    print(f"{'='*60}")
    print(f"  MACRO  P={macro['precision']:.4f}  R={macro['recall']:.4f}  F1={macro['f1']:.4f}")
    print(f"  MICRO  P={micro_p:.4f}  R={micro_r:.4f}  F1={micro_f:.4f}   (TP={tp} FP={fp} FN={fn})")
    print(f"  Hallucination: {n_halluc}/{n_eval} tasks ({n_halluc/n*100:.1f}%) reference ≥1 non-lake file")
    # paper reference
    print(f"\n  Paper Blackboard (Table 2, DA-Code): recall≈0.60  precision≈0.84  F1≈0.64")

    # By category
    print(f"\n  By Category (macro F1):")
    by_cat = {}
    for t in per_task:
        by_cat.setdefault(t["category"], []).append(t)
    for cat, items in sorted(by_cat.items()):
        cf = sum(i.get("f1", 0) for i in items) / max(len(items), 1)
        ch = sum(1 for i in items if i.get("hallucinated")) / max(len(items), 1)
        print(f"    {cat:25s}: F1={cf:.4f}  n={len(items)}  halluc={ch*100:.0f}%")

    out = {
        "sandbox_dir": args.sandbox_dir,
        "lake_dir": args.lake_dir,
        "num_tasks": len(task_ids),
        "num_with_code": n_eval,
        "macro": {k: round(v, 4) for k, v in macro.items()},
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f, 4),
                  "tp": tp, "fp": fp, "fn": fn},
        "hallucination_rate": round(n_halluc / n, 4),
        "paper_reference": {"recall": 0.60, "precision": 0.84, "f1": 0.64},
        "per_task": per_task,
    }
    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: {args.results_file}")


if __name__ == "__main__":
    main()
