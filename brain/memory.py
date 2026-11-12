from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_MEMORY_PATH = Path(__file__).parent.parent / "data" / "memory.json"


def _norm_key(key: str) -> str:
    return " ".join(str(key).strip().lower().split())


class LongTermMemory:
    """Small JSON-backed store for facts BMO should remember across sessions."""

    def __init__(self, path: str | Path = DEFAULT_MEMORY_PATH, max_prompt_items: int = 15) -> None:
        self.path = Path(path)
        self.max_prompt_items = max_prompt_items
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError:
            return
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, dict):
            return
        for key, item in items.items():
            if isinstance(item, dict) and isinstance(item.get("value"), str):
                self._items[str(key)] = item

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"version": 1, "items": self._items}, indent=2, ensure_ascii=False)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".memory-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")
            os.replace(tmp_name, self.path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def remember(self, key: str, value: str, category: str = "fact") -> dict[str, Any]:
        norm = _norm_key(key)
        now = time.time()
        with self._lock:
            existing = self._items.get(norm)
            item = {
                "key": norm,
                "value": str(value).strip(),
                "category": str(category or "fact").strip() or "fact",
                "created_at": existing.get("created_at", now) if existing else now,
                "updated_at": now,
            }
            self._items[norm] = item
            self._save_locked()
        return dict(item)

    def forget(self, key: str) -> bool:
        norm = _norm_key(key)
        with self._lock:
            if norm not in self._items:
                return False
            del self._items[norm]
            self._save_locked()
        return True

    def recall(self, key: str) -> dict[str, Any] | None:
        norm = _norm_key(key)
        with self._lock:
            item = self._items.get(norm)
            return dict(item) if item else None

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        q = _norm_key(query)
        with self._lock:
            if not q:
                return []
            hits = [
                dict(item)
                for key, item in self._items.items()
                if q in key or q in str(item.get("value", "")).lower()
            ]
        hits.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
        return hits[: max(1, limit)]

    def items(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [dict(item) for item in self._items.values()]
        rows.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
        return rows

    def as_prompt_lines(self, limit: int | None = None) -> list[str]:
        rows = self.items()
        cap = self.max_prompt_items if limit is None else max(0, limit)
        return [f"- {item['key']}: {item['value']}" for item in rows[:cap]]
