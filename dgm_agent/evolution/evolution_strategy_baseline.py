from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class EvolutionStrategy:
    """Immutable probabilistic rollback strategy from Appendix A.3."""

    def select_parent(self, archive: List[Dict[str, Any]], rng: Any) -> Optional[str]:
        if not archive:
            return None
        weights = []
        for node in archive:
            score = float(node.get("score", 0.0))
            children = int(node.get("children", 0))
            sigmoid = 1.0 / (1.0 + math.exp(-10.0 * (score - 0.5)))
            weights.append(sigmoid / (1 + children))
        total = sum(weights)
        if total <= 0:
            return str(rng.choice(archive)["id"])
        needle = rng.random() * total
        cumulative = 0.0
        for node, weight in zip(archive, weights):
            cumulative += weight
            if cumulative >= needle:
                return str(node["id"])
        return str(archive[-1]["id"])
