from __future__ import annotations

import ast
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


FORBIDDEN_IMPORT_ROOTS = {"os", "sys", "subprocess", "shutil", "pathlib", "socket", "ctypes"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "input"}
FORBIDDEN_ATTRIBUTES = {"__subclasses__", "__globals__", "__code__", "__getattribute__"}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str


def static_validate(source: str) -> ValidationResult:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ValidationResult(False, f"syntax error: {exc}")

    strategy_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORT_ROOTS:
                    return ValidationResult(False, f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                return ValidationResult(False, f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return ValidationResult(False, f"forbidden call: {node.func.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            return ValidationResult(False, f"forbidden attribute: {node.attr}")
        elif isinstance(node, ast.ClassDef) and node.name == "EvolutionStrategy":
            strategy_class = node

    if strategy_class is None:
        return ValidationResult(False, "EvolutionStrategy class is missing")
    methods = {n.name for n in strategy_class.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "select_parent" not in methods:
        return ValidationResult(False, "select_parent method is missing")
    return ValidationResult(True, "static validation passed")


def runtime_validate(source: str, python: str, timeout_seconds: float = 3.0) -> ValidationResult:
    static = static_validate(source)
    if not static.valid:
        return static
    wrapper = source + """

if __name__ == "__main__":
    import json
    import random
    mock = [
        {"id": "seed", "score": 0.5, "children": 0, "cycle": 0},
        {"id": "node-1", "score": 0.8, "children": 2, "cycle": 1},
    ]
    value = EvolutionStrategy().select_parent(mock, random.Random(7))
    if value not in {"seed", "node-1"}:
        raise SystemExit("invalid parent id")
    print(json.dumps({"selected": value}))
"""
    with tempfile.TemporaryDirectory(prefix="tdgm-strategy-") as temp_dir:
        script = Path(temp_dir) / "candidate_strategy.py"
        script.write_text(wrapper, encoding="utf-8")
        try:
            result = subprocess.run(
                [python, "-I", str(script)],
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(False, f"runtime exceeded {timeout_seconds}s")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1000:]
            return ValidationResult(False, f"runtime validation failed: {detail}")
    return ValidationResult(True, "runtime validation passed")
