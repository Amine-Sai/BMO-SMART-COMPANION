from collections.abc import Callable, Mapping
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from typing import Any

import orjson

from brain.privacy import sanitize_payload


def _default_serializer(obj: Any) -> Any:
    try:
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"


class JSONLLogger:
    def __init__(
        self,
        log_path: str | Path,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        clock: Callable[[], float] = time.time,
        allowlist: Mapping[str, set[str]] | None = None,
        redact_pii: bool = True,
    ) -> None:
        self._clock = clock
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._allowlist = allowlist
        self._redact_pii = redact_pii
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
        sanitized_fields = sanitize_payload(
            fields,
            event=event,
            redact_pii=self._redact_pii,
            allowlist=self._allowlist,
        )
        record: dict[str, Any] = {
            "ts": sanitized_fields.get("ts", self._clock()),
            "event": event,
            **sanitized_fields,
        }
        try:
            line = orjson.dumps(
                record,
                default=_default_serializer,
                option=orjson.OPT_SERIALIZE_NUMPY,
            ).decode("utf-8")
        except Exception:
            fallback = {
                "ts": record.get("ts", self._clock()),
                "event": event,
                "error": "<unserializable_payload>",
            }
            line = orjson.dumps(fallback).decode("utf-8")
        self._logger.info(line)
        return record

    def close(self) -> None:
        try:
            self._handler.flush()
        except Exception:
            pass
        if self._handler in self._logger.handlers:
            try:
                self._handler.close()
            except Exception:
                pass
            self._logger.removeHandler(self._handler)
