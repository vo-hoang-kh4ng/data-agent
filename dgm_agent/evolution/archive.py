from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ArchiveNode:
    id: str
    cycle: int
    parent_id: Optional[str]
    candidate_root: str
    score: float
    runtime_score: float
    epiplexity_score: float
    self_complexity: float
    goldilocks_status: str
    context_node: str
    task_preview: str
    mutation_log: Dict[str, Any]
    stage1: Dict[str, Any]
    stage2: Dict[str, Any]
    children: int = 0
    created_at: str = field(default_factory=utc_now)


class EvolutionArchive:
    def __init__(self, path: Path, run_metadata: Optional[Dict[str, Any]] = None):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "schema_version": 1,
                "created_at": utc_now(),
                "run_metadata": run_metadata or {},
                "nodes": [],
                "events": [],
            }
            self.save()

    @property
    def nodes(self) -> List[Dict[str, Any]]:
        return self.data["nodes"]

    def add_node(self, node: ArchiveNode) -> None:
        if any(item["id"] == node.id for item in self.nodes):
            raise ValueError(f"duplicate archive node: {node.id}")
        self.nodes.append(asdict(node))
        self.save()

    def add_event(self, kind: str, payload: Dict[str, Any]) -> None:
        self.data["events"].append({"time": utc_now(), "kind": kind, "payload": payload})
        self.save()

    def get(self, node_id: str) -> Dict[str, Any]:
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        raise KeyError(node_id)

    def increment_children(self, node_id: str) -> None:
        node = self.get(node_id)
        node["children"] = int(node.get("children", 0)) + 1
        self.save()

    def best(self) -> Dict[str, Any]:
        passed = [n for n in self.nodes if n.get("goldilocks_status") == "PASS"]
        if not passed:
            raise RuntimeError("archive has no Goldilocks-PASS node")
        return max(passed, key=lambda n: (float(n["score"]), -int(n["cycle"]), n["id"]))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
