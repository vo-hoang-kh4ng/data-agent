from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class EvolutionStrategy:
    """Editable UCB-style parent-selection policy from Appendix A.2."""

    def select_parent(self, archive: List[Dict[str, Any]], rng: Any) -> Optional[str]:
        if not archive:
            return None
        total = 1 + sum(int(node.get("children", 0)) for node in archive)
        scored = []
        for node in archive:
            children = int(node.get("children", 0))
            value = float(node.get("score", 0.0)) + 0.1 * math.sqrt(math.log(total) / (1 + children))
            scored.append((value, -int(node.get("cycle", 0)), str(node["id"])))
        return max(scored)[2]
