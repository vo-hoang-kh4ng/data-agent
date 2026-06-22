"""DABench data preparation.

DABench (InfiAgent/DABench) ships three pieces on the HF hub (repo_type=dataset):
  - da-dev-questions.jsonl  ({id, question, concepts, constraints, format, file_name, level})
  - da-dev-labels.jsonl     ({id, common_answers: [[field, value], ...]})
  - da-dev-tables/*.csv      (52 CSV data files)

`prepare()` materialises them into a self-contained `data/dabench/` dir as
questions.jsonl / labels.jsonl / tables/*.csv so the rest of the harness never
touches the HF cache layout. It uses `snapshot_download`, which is a no-op
re-download when the snapshot is already cached locally.
"""
import os
import shutil


def _repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(d, "config.yaml")) or os.path.exists(os.path.join(d, ".env")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.getcwd()


REPO_ROOT = _repo_root()
DEFAULT_DATA_DIR = os.environ.get("TRIADIC_EVAL_DATA", os.path.join(REPO_ROOT, "data"))
REPO_ID = "InfiAgent/DABench"


def prepare(data_dir: str = "") -> str:
    data_dir = data_dir or os.path.join(DEFAULT_DATA_DIR, "dabench")
    questions_path = os.path.join(data_dir, "questions.jsonl")
    tables_dir = os.path.join(data_dir, "tables")
    if os.path.exists(questions_path) and os.path.isdir(tables_dir):
        print(f"[dabench] already prepared at {data_dir}")
        return data_dir

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"[dabench] resolving DABench snapshot (cache or download)...")
    snap = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    # Copy the two jsonl files (handle either name present in the snapshot).
    for src_name, dst_name in (("da-dev-questions.jsonl", "questions.jsonl"),
                               ("da-dev-labels.jsonl", "labels.jsonl")):
        src = os.path.join(snap, src_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(data_dir, dst_name))
    # Copy the CSV tables.
    src_tables = os.path.join(snap, "da-dev-tables")
    n = 0
    if os.path.isdir(src_tables):
        for fn in os.listdir(src_tables):
            if fn.endswith(".csv"):
                shutil.copy(os.path.join(src_tables, fn), os.path.join(tables_dir, fn))
                n += 1
    # Fallback: some snapshots flatten tables into the root.
    if n == 0:
        for fn in os.listdir(snap):
            if fn.endswith(".csv"):
                shutil.copy(os.path.join(snap, fn), os.path.join(tables_dir, fn))
                n += 1

    if not os.path.exists(questions_path):
        raise RuntimeError(f"DABench prepare failed: questions.jsonl missing at {data_dir}")
    print(f"[dabench] prepared {n} tables + questions/labels at {data_dir}")
    return data_dir


if __name__ == "__main__":
    prepare()
