from collections.abc import Callable
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from typing import Any

import orjson

from brain.privacy import sanitize_payload


class JSONLLogger:
    def __init__(
        self,
        log_path: str | Path,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._clock = clock
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"jsonl.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handler = RotatingFileHandler(
            str(self._path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(self._handler)

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        sanitized_fields = sanitize_payload(fields)
        record: dict[str, Any] = {
            "ts": self._clock(),
            "event": event,
            **sanitized_fields,
        }
        line = orjson.dumps(record).decode("utf-8")
        self._logger.info(line)
        return record

    def close(self) -> None:
        self._handler.close()
        self._logger.removeHandler(self._handler)
