from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def task_id(record: Dict[str, Any]) -> str:
    value = record.get("task_id", record.get("id"))
    if value is None:
        raise ValueError("manifest record is missing task_id/id")
    return str(value)


def manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_protocol(dev_manifest: Path, benchmark_manifest: Path, data_lake: Path) -> Dict[str, Any]:
    dev = read_jsonl(dev_manifest)
    benchmark = read_jsonl(benchmark_manifest)
    if len(benchmark) != 91:
        raise ValueError(f"benchmark manifest must contain exactly 91 tasks; found {len(benchmark)}")
    if len(dev) < 60:
        raise ValueError(f"development manifest needs at least 60 tasks; found {len(dev)}")
    dev_ids = [task_id(item) for item in dev]
    benchmark_ids = [task_id(item) for item in benchmark]
    if len(dev_ids) != len(set(dev_ids)):
        raise ValueError("development manifest contains duplicate task IDs")
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("benchmark manifest contains duplicate task IDs")
    overlap = sorted(set(dev_ids) & set(benchmark_ids))
    if overlap:
        raise ValueError(f"development/benchmark leakage: {len(overlap)} overlapping IDs; examples={overlap[:10]}")
    if not data_lake.is_dir():
        raise FileNotFoundError(data_lake)
    lake_files = [path for path in data_lake.rglob("*") if path.is_file()]
    if len(lake_files) != 147:
        raise ValueError(f"paper-faithful data lake must contain 147 files; found {len(lake_files)}")
    return {
        "dev_tasks": len(dev),
        "benchmark_tasks": len(benchmark),
        "lake_files": len(lake_files),
        "dev_manifest_sha256": manifest_hash(dev_manifest),
        "benchmark_manifest_sha256": manifest_hash(benchmark_manifest),
    }


def write_fixed_dev_subsets(
    dev_manifest: Path,
    output_dir: Path,
    seed: int,
    stage1_size: int = 10,
    stage2_size: int = 50,
) -> Dict[str, Path]:
    records = read_jsonl(dev_manifest)
    if len(records) < stage1_size + stage2_size:
        raise ValueError("development manifest is too small for disjoint Stage-1/Stage-2 subsets")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    selected = {
        "stage1": shuffled[:stage1_size],
        "stage2": shuffled[stage1_size : stage1_size + stage2_size],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, subset in selected.items():
        path = output_dir / f"{name}_manifest.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in subset:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        paths[name] = path
    audit = {
        "seed": seed,
        "stage1_ids": [task_id(record) for record in selected["stage1"]],
        "stage2_ids": [task_id(record) for record in selected["stage2"]],
    }
    (output_dir / "development_split.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return paths
