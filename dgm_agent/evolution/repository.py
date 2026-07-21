from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {relative}") from exc
    return candidate


def snapshot_hashes(root: Path, relative_paths: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for relative in relative_paths:
        path = safe_relative(root, relative)
        if path.is_file():
            result[relative] = sha256_file(path)
        elif path.is_dir():
            for child in sorted(
                item
                for item in path.rglob("*")
                if item.is_file()
                and "__pycache__" not in item.parts
                and item.suffix not in {".pyc", ".pyo"}
            ):
                child_relative = child.relative_to(root).as_posix()
                result[child_relative] = sha256_file(child)
    return result


def repository_map(root: Path, mutable_files: Iterable[str]) -> str:
    lines: List[str] = []
    for relative in mutable_files:
        path = safe_relative(root, relative)
        if not path.exists():
            lines.append(f"- {relative}: MISSING")
            continue
        if path.suffix != ".py":
            lines.append(f"- {relative}: {path.stat().st_size} bytes")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            symbols = []
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(node.name)
            lines.append(f"- {relative}: {path.stat().st_size} bytes; symbols={symbols[:40]}")
        except (SyntaxError, UnicodeDecodeError) as exc:
            lines.append(f"- {relative}: parse error: {exc}")
    return "\n".join(lines)
