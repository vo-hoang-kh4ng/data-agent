"""
DS-Bench Dataset Downloader
============================
Downloads tasks from liqiang888/DSBench on HuggingFace.

Dataset structure:
  data_analysis/data.json   - JSONL with tasks (id, questions, answers)
  data_analysis/data.zip    - ZIP with Excel files per task
  data_modeling/data.json   - JSONL with modeling tasks
  data_modeling/data.zip    - ZIP with data files

Usage:
    python download_dsbench.py
    python download_dsbench.py --num_tasks 20 --output_dir data/local_test
    python download_dsbench.py --subset data_analysis --num_tasks 10
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path


def download_dsbench(
    num_tasks: int = 10,
    output_dir: str = "data/local_test",
    subset: str = "data_analysis",   # 'data_analysis' or 'data_modeling'
):
    """Download num_tasks tasks from DS-Bench and extract associated data files."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tasks_manifest_path = output_path / "tasks_manifest.jsonl"
    cache_dir = output_path / "_hf_cache"

    # Step 1: Download repo
    print(f"Downloading liqiang888/DSBench from HuggingFace...")
    try:
        local_dir = snapshot_download(
            repo_id="liqiang888/DSBench",
            repo_type="dataset",
            local_dir=str(cache_dir),
            ignore_patterns=["*.git*"],
        )
        print(f"  Cached at: {local_dir}")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    local_root = Path(local_dir)
    subset_dir = local_root / subset

    if not subset_dir.exists():
        available = [d.name for d in local_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        print(f"Subset '{subset}' not found. Available: {available}")
        sys.exit(1)

    # Step 2: Load task metadata from JSONL
    data_json = subset_dir / "data.json"
    if not data_json.exists():
        print(f"data.json not found in {subset_dir}")
        sys.exit(1)

    rows = []
    with open(data_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Skipping malformed line: {e}")

    print(f"  Found {len(rows)} tasks in {subset}/data.json")

    # Step 3: Extract data.zip if not already done
    data_zip = subset_dir / "data.zip"
    extract_dir = subset_dir / "extracted"
    if data_zip.exists() and not extract_dir.exists():
        print(f"  Extracting {data_zip.name} ({data_zip.stat().st_size // 1024 // 1024} MB)...")
        with zipfile.ZipFile(data_zip, "r") as zf:
            zf.extractall(extract_dir)
        print(f"  Extracted to: {extract_dir}")
    elif extract_dir.exists():
        print(f"  Already extracted: {extract_dir}")
    else:
        print(f"  No data.zip found, proceeding without extracted files.")
        extract_dir = None

    # Step 4: Process tasks and build manifest
    num_tasks = min(num_tasks, len(rows))
    print(f"\nProcessing {num_tasks} tasks from subset '{subset}'...")
    print(f"  Sample task keys: {list(rows[0].keys())}")

    manifest = []
    with open(tasks_manifest_path, "w", encoding="utf-8") as mf:
        for i in range(num_tasks):
            row = rows[i]
            task_id = str(row.get("id", f"task_{i:04d}"))
            task_name = row.get("name", task_id)

            task_dir = output_path / task_id
            task_dir.mkdir(exist_ok=True)

            # Build question string from questions list
            questions_list = row.get("questions", [])
            answers_list = row.get("answers", [])
            question_text = row.get("txt", "") or ""
            if questions_list:
                question_text += "\n\nQuestions:\n"
                for qi, (q, a) in enumerate(zip(questions_list, answers_list)):
                    question_text += f"  {qi+1}. {q}\n"

            task_info = {
                "task_id": task_id,
                "task_name": task_name,
                "task_dir": str(task_dir),
                "subset": subset,
                "question": question_text.strip(),
                "questions_list": questions_list,
                "answers": answers_list,
                "url": row.get("url", ""),
                "data_files": [],
            }

            # Find associated Excel/CSV files from extracted zip
            if extract_dir and extract_dir.exists():
                # Look for folder matching task name or id
                DATA_EXTS = {".xlsx", ".xls", ".csv", ".json", ".txt", ".tsv"}
                # Search by task name (common pattern in DS-Bench)
                candidates = []
                for search_name in [task_name, task_id, task_name.replace("-", "_")]:
                    found = list(extract_dir.rglob(f"*{search_name}*"))
                    candidates.extend([f for f in found if f.is_file() and f.suffix.lower() in DATA_EXTS])
                # Also check for a directory with matching name
                for d in extract_dir.rglob("*"):
                    if d.is_dir() and (task_name in d.name or task_id in d.name):
                        for f in d.iterdir():
                            if f.is_file() and f.suffix.lower() in DATA_EXTS:
                                candidates.append(f)

                # Deduplicate and copy to task_dir
                seen = set()
                for src_file in candidates:
                    if src_file not in seen:
                        seen.add(src_file)
                        dst = task_dir / src_file.name
                        if not dst.exists():
                            import shutil
                            shutil.copy2(src_file, dst)
                        task_info["data_files"].append(str(dst))

            # Save full task metadata
            serialisable = {k: v for k, v in row.items()
                            if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            serialisable["data_files"] = task_info["data_files"]
            serialisable["task_dir"] = str(task_dir)

            task_meta_path = task_dir / "task_meta.json"
            with open(task_meta_path, "w", encoding="utf-8") as tmf:
                json.dump(serialisable, tmf, indent=2, ensure_ascii=False)

            mf.write(json.dumps(task_info, ensure_ascii=False) + "\n")
            print(f"  [{i+1:02d}/{num_tasks}] {task_id}: {task_name[:50]} | files={len(task_info['data_files'])}")
            manifest.append(task_info)

    print(f"\nDone! {num_tasks} tasks saved to: {output_path.resolve()}")
    print(f"Manifest: {tasks_manifest_path}")

    # Quick summary
    with_files = sum(1 for t in manifest if t["data_files"])
    print(f"  Tasks with data files: {with_files}/{num_tasks}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DS-Bench tasks from HuggingFace")
    parser.add_argument("--num_tasks", type=int, default=10,
                        help="Number of tasks to download (default: 10)")
    parser.add_argument("--output_dir", type=str, default="data/local_test",
                        help="Output directory")
    parser.add_argument("--subset", type=str, default="data_analysis",
                        choices=["data_analysis", "data_modeling"],
                        help="Which DS-Bench subset to use")
    args = parser.parse_args()

    download_dsbench(
        num_tasks=args.num_tasks,
        output_dir=args.output_dir,
        subset=args.subset,
    )
