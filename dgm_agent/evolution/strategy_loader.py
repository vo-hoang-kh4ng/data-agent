from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def load_strategy(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"tdgm_strategy_{path.stat().st_mtime_ns}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load strategy: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy_type = getattr(module, "EvolutionStrategy", None)
    if strategy_type is None:
        raise ImportError("EvolutionStrategy class is missing")
    return strategy_type()
