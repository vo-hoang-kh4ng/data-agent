from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List


class RuleMemory:
    def __init__(self, path: Path):
        self.path = path
        self.rules: List[Dict[str, Any]] = []
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.rules = list(raw.get("rules", []))

    @staticmethod
    def _token_count(text: str) -> int:
        return max(1, len(text.split()))

    def add(self, text: str, error: str, cycle: int) -> None:
        normalized = " ".join(text.split())[:2000]
        if not normalized:
            return
        existing = next((rule for rule in self.rules if rule["text"] == normalized), None)
        if existing:
            existing["corrections"] = int(existing.get("corrections", 0)) + 1
            existing["last_cycle"] = cycle
        else:
            self.rules.append({
                "text": normalized,
                "error": error[-2000:],
                "corrections": 1,
                "tokens": self._token_count(normalized),
                "first_cycle": cycle,
                "last_cycle": cycle,
            })
        self.save()

    def top(self, limit: int) -> List[Dict[str, Any]]:
        return sorted(
            self.rules,
            key=lambda rule: (float(rule["corrections"]) / max(1, int(rule["tokens"])), int(rule["last_cycle"])),
            reverse=True,
        )[:limit]

    def render(self, limit: int) -> str:
        rules = self.top(limit)
        return "\n".join(f"- {item['text']}" for item in rules) if rules else "(no prior repair rules)"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": 1, "rules": self.rules}, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
