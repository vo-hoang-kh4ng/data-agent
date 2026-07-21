from __future__ import annotations

import math
import zlib
from collections import Counter


def compressed_size(text: str) -> int:
    return len(zlib.compress(text.encode("utf-8")))


def shannon_entropy(text: str) -> float:
    data = text.encode("utf-8")
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def self_complexity(code: str) -> float:
    """Epiplexity axis 1: H(c) * (1 - |zlib(c)| / |c|)."""
    data = code.encode("utf-8")
    if not data:
        return 0.0
    ratio = len(zlib.compress(data)) / len(data)
    return shannon_entropy(code) * (1.0 - ratio)


def normalized_compression_distance(x: str, y: str) -> float:
    cx, cy = compressed_size(x), compressed_size(y)
    denominator = max(cx, cy)
    if denominator == 0:
        return 0.0
    cxy = compressed_size(x + "\n" + y)
    return (cxy - min(cx, cy)) / denominator


def task_solution_epiplexity(task: str, solution: str) -> float:
    """Epiplexity axis 2 from the paper: 2 * NCD(task, solution)."""
    return 2.0 * normalized_compression_distance(task, solution)


def goldilocks_status(value: float, lower: float = 0.5, upper: float = 2.2) -> str:
    if value < lower:
        return "LOW"
    if value > upper:
        return "HIGH"
    return "PASS"


def runtime_score(stage1_pass_rate: float) -> float:
    if not 0.0 <= stage1_pass_rate <= 1.0:
        raise ValueError("stage1_pass_rate must be within [0, 1]")
    return 0.5 + 0.5 * stage1_pass_rate
