from __future__ import annotations

import ast
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .repository import safe_relative, sha256_file


@dataclass(frozen=True)
class MutationResult:
    target_file: str
    rationale: str
    rule: str
    before_sha256: str
    after_sha256: str
    edit_count: int


def create_candidate_copy(parent_root: Path, candidate_root: Path, copy_paths: Iterable[str]) -> None:
    if candidate_root.exists():
        raise FileExistsError(candidate_root)
    candidate_root.mkdir(parents=True)
    for relative in copy_paths:
        source = safe_relative(parent_root, relative)
        target = safe_relative(candidate_root, relative)
        if not source.exists():
            raise FileNotFoundError(f"copy path does not exist: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "data", "outputs"),
            )
        else:
            shutil.copy2(source, target)


def _is_frozen(relative: str, frozen_files: Iterable[str]) -> bool:
    normalized = Path(relative).as_posix().lstrip("./")
    for frozen in frozen_files:
        frozen_norm = Path(frozen).as_posix().lstrip("./")
        if normalized == frozen_norm or normalized.startswith(frozen_norm.rstrip("/") + "/"):
            return True
    return False


def apply_exact_edits(
    candidate_root: Path,
    proposal: Dict[str, Any],
    mutable_files: Iterable[str],
    frozen_files: Iterable[str],
    max_file_bytes: int,
) -> MutationResult:
    target_relative = str(proposal.get("target_file", ""))
    allowed = {Path(item).as_posix().lstrip("./") for item in mutable_files}
    normalized = Path(target_relative).as_posix().lstrip("./")
    if normalized not in allowed:
        raise ValueError(f"target is not in mutable whitelist: {target_relative}")
    if _is_frozen(normalized, frozen_files):
        raise ValueError(f"target is frozen: {target_relative}")

    target = safe_relative(candidate_root, normalized)
    if not target.is_file():
        raise FileNotFoundError(target)
    if target.stat().st_size > max_file_bytes:
        raise ValueError(f"mutable file exceeds {max_file_bytes} bytes: {normalized}")

    edits = proposal.get("edits")
    if not isinstance(edits, list) or not 1 <= len(edits) <= 8:
        raise ValueError("proposal must contain 1-8 exact edits")
    content = target.read_text(encoding="utf-8")
    before_hash = sha256_file(target)
    for index, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict):
            raise ValueError(f"edit {index} is not an object")
        old, new = edit.get("old"), edit.get("new")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise ValueError(f"edit {index} must have non-empty old and string new")
        occurrences = content.count(old)
        if occurrences != 1:
            raise ValueError(f"edit {index} old text occurs {occurrences} times; expected exactly once")
        content = content.replace(old, new, 1)
    if target.suffix == ".py":
        ast.parse(content, filename=normalized)
    target.write_text(content, encoding="utf-8")
    return MutationResult(
        target_file=normalized,
        rationale=str(proposal.get("rationale", ""))[:4000],
        rule=str(proposal.get("rule", ""))[:2000],
        before_sha256=before_hash,
        after_sha256=sha256_file(target),
        edit_count=len(edits),
    )


def parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1 :] if first_newline >= 0 else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value
