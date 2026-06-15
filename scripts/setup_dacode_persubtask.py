"""
Build a COMPLETE per-subtask DA-Code data lake (91 subdirs) + per-subtask manifest.

Paper-faithful protocol (arXiv:2510.01285, Footnote 13 — per-subtask lake construction):
  Each task receives ONLY its own data files (the original DA-Code per-task source dir),
  NOT a unified global pool. This is the "easy discovery" setting and is the counterpart
  to the repo's unified 172-file global lake (dacode_lake), which is the "real discovery"
  setting behind the 0.2529 headline.

Difference vs setup_dacode_unified.py:
  - Unified: ALL files from 91 tasks aggregated into ONE flat pool (172 files), data_lake_dir
    is the SAME for every task. Agent must DISCOVER relevant files among 172.
  - Per-subtask (THIS script): each task gets its OWN subdir as data_lake_dir. Discovery is
    trivial (lake == the task's files). Used to measure the discovery-difficulty delta.

Hint/gold integrity:
  Applies the SAME exclusion filter as dgm_agent/blackboard.py (HINT_EXACT_NAMES +
  HINT_STEM_SUBSTRINGS + GOLD_TEMPLATE_NAMES) so solution methodology / gold templates never
  reach the agent — the discovery benchmark stays valid in both settings.

This script creates a FRESH data/dacode_lake_persubtask/ (does NOT touch the user-provided
data/dacode_lake_145, which covers only 56/91 tasks).

Usage:
    python scripts/setup_dacode_persubtask.py
"""

import json
import os
import shutil
import hashlib

# ── Paths ──
DATA_AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data-agent/
DATA_DIR = os.path.join(DATA_AGENT_ROOT, "data")

IDS_FILE = os.path.join(os.path.dirname(DATA_AGENT_ROOT), "da_code_retained_example_ids.txt")
SOURCE_DIR = os.path.join(DATA_DIR, "dacode_source", "source")

LAKE_DIR = os.path.join(DATA_DIR, "dacode_lake_persubtask")            # NEW per-subtask lake
UNIFIED_MANIFEST = os.path.join(DATA_DIR, "dacode_unified_manifest.jsonl")
PERSUBTASK_MANIFEST = os.path.join(DATA_DIR, "dacode_persubtask_manifest.jsonl")

# ── Hint / gold exclusion (MUST match dgm_agent/blackboard.py exactly) ──
HINT_EXACT_NAMES = {
    "README.md", ".DS_Store",
    "result.csv", "sample_result.csv",            # gold output templates
    "tips.txt", "tips.md", "guidance.txt", "step.md", "workflow.md",
    "data_standard.md", "weight_class.md",
    "playerposition.txt", "BMI.txt", "age.txt", "iqr.txt",
    "relevant_avocado_categories.txt", "relevant_olive_oil_categories.txt",
    "relevant_sourdough_categories.txt",
}
HINT_STEM_SUBSTRINGS = ("tips", "guidance", "step", "workflow", "data_standard",
                        "playerposition", "relevant_", "weight_class")
DATA_EXTS = {".csv", ".xlsx", ".xls", ".ods", ".json", ".jsonl", ".tsv", ".txt", ".sqlite", ".db"}


def _is_hint_or_gold(fname: str) -> bool:
    if fname in HINT_EXACT_NAMES or fname.startswith('.'):
        return True
    stem = os.path.splitext(fname)[0].lower()
    return any(s in stem for s in HINT_STEM_SUBSTRINGS)


def load_retained_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("Source") \
                    and not line.startswith("Total") and not line.startswith("Retained"):
                ids.append(line)
    return ids


def build_persubtask_lake(retained_ids, source_dir, lake_dir):
    """For each retained task, create <lake_dir>/<task_id>/ with hint-filtered data files."""
    if os.path.exists(lake_dir):
        shutil.rmtree(lake_dir)
    os.makedirs(lake_dir, exist_ok=True)

    total_copied = 0
    total_filtered = 0
    tasks_with_data = 0
    tasks_empty = []

    for task_id in retained_ids:
        task_source = os.path.join(source_dir, task_id)
        task_dir = os.path.join(lake_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        if not os.path.exists(task_source):
            tasks_empty.append(task_id)
            continue

        copied_here = 0
        for fname in os.listdir(task_source):
            src_file = os.path.join(task_source, fname)
            if not os.path.isfile(src_file):
                continue
            if _is_hint_or_gold(fname):
                total_filtered += 1
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in DATA_EXTS:
                total_filtered += 1
                continue
            shutil.copy2(src_file, os.path.join(task_dir, fname))
            copied_here += 1
            total_copied += 1

        if copied_here > 0:
            tasks_with_data += 1
        else:
            tasks_empty.append(task_id)

    print(f"📂 Per-subtask lake: {total_copied} files across {len(retained_ids)} task subdirs "
          f"({tasks_with_data} with data), {total_filtered} hint/non-data filtered")
    if tasks_empty:
        print(f"   ⚠️ {len(tasks_empty)} tasks ended with 0 data files: {tasks_empty}")
    return total_copied, tasks_with_data


def verify_no_hints(lake_dir):
    """Fail-fast: assert no hint/gold file survived in any subdir."""
    leaks = []
    for root, _dirs, files in os.walk(lake_dir):
        for fname in files:
            if _is_hint_or_gold(fname):
                leaks.append(os.path.join(root, fname))
    if leaks:
        print(f"  🚨 HINT LEAK — {len(leaks)} hint/gold file(s) present:")
        for p in leaks[:20]:
            print(f"     {p}")
        raise SystemExit("Aborting: hint/gold files present in per-subtask lake.")
    print(f"  🛡️  Verified: 0 hint/gold files in per-subtask lake.")


def generate_manifest(unified_manifest_path, lake_dir, out_path):
    """Rewrite unified manifest's data_lake_dir → per-subtask subdir; keep all eval fields."""
    entries = []
    with open(unified_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            tid = e["task_id"]
            e["data_lake_dir"] = os.path.abspath(os.path.join(lake_dir, tid))
            entries.append(e)

    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"📝 Per-subtask manifest: {len(entries)} entries → {out_path}")
    print(f"   (each task's data_lake_dir → its own subdir; eval configs identical to unified)")
    return entries


def main():
    print("=" * 64)
    print("🔧 Building COMPLETE per-subtask DA-Code lake (91 tasks, Footnote-13 protocol)")
    print("=" * 64)

    retained_ids = load_retained_ids(IDS_FILE)
    print(f"📋 {len(retained_ids)} retained task IDs")

    print("\n📦 Building per-subtask lake from source...")
    n_files, n_with_data = build_persubtask_lake(retained_ids, SOURCE_DIR, LAKE_DIR)

    print("\n🛡️  Verifying hint/gold integrity...")
    verify_no_hints(LAKE_DIR)

    print("\n📝 Generating per-subtask manifest...")
    generate_manifest(UNIFIED_MANIFEST, LAKE_DIR, PERSUBTASK_MANIFEST)

    print("\n" + "=" * 64)
    print(f"✅ Done. Per-subtask lake: {LAKE_DIR} ({n_files} files, {n_with_data}/91 tasks with data)")
    print(f"   Manifest: {PERSUBTASK_MANIFEST}")
    print(f"\n   Run inference (per-subtask):")
    print(f"     python -m dgm_agent.dacode_runner \\")
    print(f"       --manifest {os.path.relpath(PERSUBTASK_MANIFEST, DATA_AGENT_ROOT)} \\")
    print(f"       --sandbox_dir data/dacode_sandbox_persubtask --force_rerun")
    print("=" * 64)


if __name__ == "__main__":
    main()
