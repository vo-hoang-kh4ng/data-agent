"""
Setup Unified DA-Code Data Lake for Blackboard Evaluation
==========================================================
Theo paper "LLM-based Multi-Agent Blackboard System" (arxiv 2510.01285):
- Lọc 91 retained example IDs từ Appendix G
- Gom tất cả source files thành 1 unified data lake
- Tạo manifest cho runner

Usage:
    python setup_dacode_unified.py
"""

import json
import os
import shutil
from collections import OrderedDict

# ── Paths ──
DATA_AGENT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DATA_AGENT_ROOT, "data")

IDS_FILE = os.path.join(os.path.dirname(DATA_AGENT_ROOT), "da_code_retained_example_ids.txt")
SOURCE_DIR = os.path.join(DATA_DIR, "dacode_source", "source")
TASK_CONFIG = os.path.join(DATA_DIR, "dacode_configs", "_raw", "task_cfg", "all.jsonl")
EVAL_CONFIG = os.path.join(DATA_DIR, "dacode_configs", "_raw", "eval_cfg", "eval_all.jsonl")
GOLD_DIR = os.path.join(DATA_DIR, "dacode_gold", "gold")

LAKE_DIR = os.path.join(DATA_DIR, "dacode_lake")
MANIFEST_PATH = os.path.join(DATA_DIR, "dacode_unified_manifest.jsonl")
FILTERED_TASK_PATH = os.path.join(DATA_DIR, "dacode_configs", "retained_91_task.jsonl")
FILTERED_EVAL_PATH = os.path.join(DATA_DIR, "dacode_configs", "retained_91_eval.jsonl")


def load_retained_ids(path):
    """Load 91 retained IDs from text file."""
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("Source") and not line.startswith("Total") and not line.startswith("Retained"):
                ids.append(line)
    print(f"📋 Loaded {len(ids)} retained IDs")
    return ids


def load_jsonl(path):
    """Load JSONL file into dict keyed by id. Handles concatenated JSON on same line."""
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Use json.JSONDecoder to extract multiple objects
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        # Skip whitespace
        while idx < len(content) and content[idx] in ' \t\n\r':
            idx += 1
        if idx >= len(content):
            break
        try:
            data, end = decoder.raw_decode(content, idx)
            items[data["id"]] = data
            idx = end
        except json.JSONDecodeError:
            # Skip to next line
            next_nl = content.find('\n', idx)
            if next_nl == -1:
                break
            idx = next_nl + 1
    return items


def build_unified_lake(retained_ids, source_dir, lake_dir):
    """Copy all source files from retained tasks into unified lake, dedup.

    Paper-faithful setup (arXiv:2510.01285, Table 3):
    - Aggregate all data files from 91 retained tasks into one lake
    - Exclude gold templates (result.csv, sample_result.csv) — these are output hints
    - Exclude non-data files (README, tips, .png, .sql, etc.)
    - Dedup by (filename, content_hash): same name + same content = true duplicate
    - Same name + different content → prefix with task_id
    """
    import hashlib

    # Files to skip entirely
    SKIP_NAMES = {
        "README.md", "tips.md", ".DS_Store", "tips.txt", "guidance.txt",
        "step.md", "workflow.md", "analysis.py", "playerposition.txt",
        "BMI.txt", "age.txt", "iqr.txt", "data_standard.md", "weight_class.md",
        "relevant_avocado_categories.txt", "relevant_olive_oil_categories.txt",
        "relevant_sourdough_categories.txt",
    }
    # Gold template files — output format hints, not input data
    GOLD_TEMPLATE_NAMES = {"result.csv", "sample_result.csv"}
    # Only include these extensions (real data files)
    DATA_EXTS = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".tsv", ".txt", ".sqlite", ".db"}

    if os.path.exists(lake_dir):
        shutil.rmtree(lake_dir)
    os.makedirs(lake_dir, exist_ok=True)

    # Track by (filename, content_hash) for correct dedup
    seen_by_name_hash = {}  # (fname, hash) -> True
    seen_by_name = {}       # fname -> first_hash (to detect name collisions with diff content)
    total_copied = 0
    total_skipped = 0
    total_filtered = 0
    file_task_map = {}  # filename -> list of task_ids

    for task_id in retained_ids:
        task_source = os.path.join(source_dir, task_id)
        if not os.path.exists(task_source):
            print(f"  ⚠️ Source dir not found: {task_id}")
            continue

        for fname in os.listdir(task_source):
            src_file = os.path.join(task_source, fname)
            if not os.path.isfile(src_file):
                continue

            # Filter: skip non-data files (exact names + hint/gold patterns).
            # Substring check catches task-prefixed hint variants from the dedup
            # collision path (e.g. "dm-csv-020_guidance.txt"); only safe, hint-specific
            # substrings are used to avoid false positives on real data files.
            stem = os.path.splitext(fname)[0].lower()
            hint_substrings = ("tips", "guidance", "data_standard", "weight_class",
                               "playerposition", "relevant_")
            if (fname in SKIP_NAMES or fname in GOLD_TEMPLATE_NAMES or fname.startswith('.')
                    or any(s in stem for s in hint_substrings)):
                total_filtered += 1
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext not in DATA_EXTS:
                total_filtered += 1
                continue

            # Compute content hash for dedup
            with open(src_file, "rb") as f:
                content_hash = hashlib.md5(f.read()).hexdigest()

            dedup_key = (fname, content_hash)

            if dedup_key in seen_by_name_hash:
                # Same filename + same content = true duplicate
                total_skipped += 1
                file_task_map.setdefault(fname, []).append(task_id)
                continue

            # Determine destination filename
            if fname in seen_by_name and seen_by_name[fname] != content_hash:
                # Same filename but DIFFERENT content → prefix with task_id
                dst_name = f"{task_id}_{fname}"
            else:
                dst_name = fname

            dst_file = os.path.join(lake_dir, dst_name)
            shutil.copy2(src_file, dst_file)
            seen_by_name_hash[dedup_key] = True
            seen_by_name[fname] = content_hash
            total_copied += 1
            file_task_map.setdefault(fname, []).append(task_id)

    print(f"📂 Unified Data Lake: {total_copied} files copied, {total_skipped} duplicates skipped, {total_filtered} non-data filtered")
    print(f"   Lake dir: {lake_dir}")
    return total_copied


def generate_manifest(retained_ids, task_configs, eval_configs, lake_dir, manifest_path):
    """Generate manifest JSONL for the unified runner."""
    manifest = []
    skipped = 0

    for task_id in retained_ids:
        if task_id not in task_configs:
            print(f"  ⚠️ Task config not found: {task_id}")
            skipped += 1
            continue
        if task_id not in eval_configs:
            print(f"  ⚠️ Eval config not found: {task_id}")
            skipped += 1
            continue

        task_cfg = task_configs[task_id]
        eval_cfg = eval_configs[task_id]

        instruction = task_cfg["instruction"]
        eval_func = eval_cfg.get("func", [])
        eval_result = eval_cfg.get("result", [])
        eval_options = eval_cfg.get("options", [])
        task_type = eval_cfg.get("config", {}).get("type", "unknown")
        hardness = eval_cfg.get("config", {}).get("hardness", "unknown")
        task_category = eval_cfg.get("config", {}).get("task", "unknown")

        # Generate test_code based on eval type
        test_code = generate_test_code(task_id, eval_func, eval_result, eval_options, task_type)

        entry = {
            "task_id": task_id,
            "question": instruction,
            "task_type": task_type,
            "task_category": task_category,
            "hardness": hardness,
            "data_lake_dir": os.path.abspath(lake_dir),
            "gold_dir": os.path.abspath(os.path.join(GOLD_DIR, task_id)),
            "eval_func": eval_func,
            "eval_result": eval_result,
            "eval_options": eval_options,
            "test_code": test_code,
        }
        manifest.append(entry)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"📝 Manifest: {len(manifest)} entries ({skipped} skipped) → {manifest_path}")

    # Stats by type
    from collections import Counter
    type_counts = Counter(e["task_category"] for e in manifest)
    for t, c in type_counts.items():
        print(f"   {t}: {c}")

    return manifest


def generate_test_code(task_id, eval_func, eval_result, eval_options, task_type):
    """Generate test/verification code based on eval type."""

    if "compare_csv" in eval_func:
        files_to_check = []
        for res in eval_result:
            if "file" in res:
                files_to_check.extend(res["file"])

        ignore_order = any(opt.get("ignore_order", False) for opt in eval_options)
        condition_cols = None
        for opt in eval_options:
            if "condition_cols" in opt:
                condition_cols = opt["condition_cols"]

        sort_stmt = ".sort_values(by=list(df.columns)).reset_index(drop=True)" if ignore_order else ""

        code = f"""import os, pandas as pd
files = {files_to_check}
for f in files:
    assert os.path.exists(f), f"Missing: {{f}}"
    df = pd.read_csv(f)
    assert len(df) > 0, f"Empty: {{f}}"
"""
        return code

    elif "compare_text" in eval_func:
        return "print('Text output — will be compared by evaluator')"

    else:
        return "print('Manual evaluation required')"


def write_filtered_configs(retained_ids, task_configs, eval_configs, task_path, eval_path):
    """Write filtered JSONL with only the 91 retained IDs."""
    os.makedirs(os.path.dirname(task_path), exist_ok=True)

    task_count = 0
    with open(task_path, "w", encoding="utf-8") as f:
        for tid in retained_ids:
            if tid in task_configs:
                f.write(json.dumps(task_configs[tid], ensure_ascii=False) + "\n")
                task_count += 1

    eval_count = 0
    with open(eval_path, "w", encoding="utf-8") as f:
        for tid in retained_ids:
            if tid in eval_configs:
                f.write(json.dumps(eval_configs[tid], ensure_ascii=False) + "\n")
                eval_count += 1

    print(f"📄 Filtered configs: {task_count} tasks, {eval_count} evals")
    print(f"   Tasks: {task_path}")
    print(f"   Evals: {eval_path}")


def main():
    print("=" * 60)
    print("🔧 Setting up Unified DA-Code Data Lake (Blackboard Paper)")
    print("=" * 60)

    # 1. Load retained IDs
    retained_ids = load_retained_ids(IDS_FILE)

    # 2. Load configs
    print("\n📥 Loading task & eval configs...")
    task_configs = load_jsonl(TASK_CONFIG)
    eval_configs = load_jsonl(EVAL_CONFIG)
    print(f"   {len(task_configs)} task configs, {len(eval_configs)} eval configs")

    # 3. Build unified data lake
    print("\n📦 Building unified data lake...")
    num_files = build_unified_lake(retained_ids, SOURCE_DIR, LAKE_DIR)

    # 4. Generate manifest
    print("\n📝 Generating manifest...")
    manifest = generate_manifest(
        retained_ids, task_configs, eval_configs,
        LAKE_DIR, MANIFEST_PATH
    )

    # 5. Write filtered configs for official evaluator
    print("\n📄 Writing filtered configs...")
    write_filtered_configs(
        retained_ids, task_configs, eval_configs,
        FILTERED_TASK_PATH, FILTERED_EVAL_PATH
    )

    print("\n" + "=" * 60)
    print(f"✅ Setup complete!")
    print(f"   Data Lake: {LAKE_DIR} ({num_files} files)")
    print(f"   Manifest:  {MANIFEST_PATH} ({len(manifest)} tasks)")
    print(f"\n   Next: python run_dacode_unified.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
