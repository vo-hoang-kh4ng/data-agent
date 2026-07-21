from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_cluster_manifest(agents: Iterable[Any], data_root: Path, output: Path, expected_k: int = 26) -> Dict[str, Any]:
    data_root = data_root.resolve()
    clusters = []
    assigned: List[str] = []
    for agent in agents:
        relative_files = []
        for raw in agent.file_paths:
            path = Path(raw).resolve()
            relative = path.relative_to(data_root).as_posix()
            relative_files.append(relative)
            assigned.append(relative)
        clusters.append({
            "name": str(agent.cluster_name),
            "files": relative_files,
        })
    if len(clusters) != expected_k:
        raise ValueError(f"expected {expected_k} clusters; got {len(clusters)}")
    if len(assigned) != len(set(assigned)):
        raise ValueError("a file was assigned to more than one cluster")
    file_hashes = {relative: sha256_file(data_root / relative) for relative in sorted(assigned)}
    payload = {
        "schema_version": 1,
        "method": "llm_hierarchical_offline",
        "k": expected_k,
        "data_root_name": data_root.name,
        "assigned_file_count": len(assigned),
        "file_sha256": file_hashes,
        "clusters": clusters,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen cluster manifest: {output}")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_frozen_clusters(manifest: Path, data_root: Path, expected_k: int = 26) -> List[Dict[str, Any]]:
    manifest = manifest.resolve()
    data_root = data_root.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported frozen cluster schema")
    if int(payload.get("k", -1)) != expected_k:
        raise ValueError(f"frozen cluster manifest K mismatch: {payload.get('k')} != {expected_k}")
    clusters = payload.get("clusters")
    if not isinstance(clusters, list) or len(clusters) != expected_k:
        raise ValueError(f"frozen cluster manifest must contain exactly {expected_k} clusters")
    assigned: List[str] = []
    materialized: List[Dict[str, Any]] = []
    hashes = payload.get("file_sha256", {})
    for index, cluster in enumerate(clusters):
        name = str(cluster.get("name", f"cluster_{index:02d}"))
        relatives = cluster.get("files")
        if not isinstance(relatives, list) or not relatives:
            raise ValueError(f"cluster {name} is empty or invalid")
        paths = []
        for raw in relatives:
            relative = Path(str(raw)).as_posix().lstrip("./")
            if relative.startswith("../") or Path(relative).is_absolute():
                raise ValueError(f"unsafe frozen cluster path: {raw}")
            path = (data_root / relative).resolve()
            try:
                path.relative_to(data_root)
            except ValueError as exc:
                raise ValueError(f"frozen cluster path escapes data root: {raw}") from exc
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_hash = hashes.get(relative)
            if not expected_hash or sha256_file(path) != expected_hash:
                raise ValueError(f"data-lake hash mismatch for {relative}")
            assigned.append(relative)
            paths.append(str(path))
        materialized.append({"name": name, "file_paths": paths})
    if len(assigned) != len(set(assigned)):
        raise ValueError("frozen cluster manifest contains duplicate file assignments")
    if len(assigned) != int(payload.get("assigned_file_count", -1)):
        raise ValueError("frozen cluster assigned-file count mismatch")
    return materialized
