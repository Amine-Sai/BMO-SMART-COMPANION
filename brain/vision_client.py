from __future__ import annotations

from collections.abc import Callable
import queue
import threading
import time
from typing import Any, Literal
import uuid

import orjson
from bmo_contracts.models import VisionCmd, VisionCmdResult
from bmo_contracts.topics import TOPICS, VISION_CMD, VISION_CMD_RESULT


class VisionClient:
    """Client for asynchronous request-response vision commands over MQTT."""

    def __init__(
        self,
        publish_fn: Callable[[str, bytes, int, bool], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._publish_fn = publish_fn
        self._clock = clock
        self._pending: dict[str, queue.Queue[VisionCmdResult]] = {}
        self._lock = threading.Lock()

    def handle_cmd_result(self, payload: VisionCmdResult | bytes | str) -> None:
        res = payload if isinstance(payload, VisionCmdResult) else VisionCmdResult.model_validate_json(payload)
        with self._lock:
            q = self._pending.get(res.id)
        if q is not None:
            q.put(res)

    def handle_mqtt_message(self, topic: str, payload: bytes | str) -> None:
        if topic == VISION_CMD_RESULT:
            self.handle_cmd_result(payload)

    def send_command(
        self,
        cmd: Literal["enroll", "forget", "reload_gallery", "set_modules"],
        args: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        cmd_id: str | None = None,
    ) -> VisionCmdResult:
        timeout = (15.0 if cmd == "enroll" else 3.0) if timeout_s is None else timeout_s
        req = VisionCmd(
            id=cmd_id or f"vcmd-{uuid.uuid4().hex[:12]}",
            cmd=cmd,
            args=dict(args or {}),
        )
        q: queue.Queue[VisionCmdResult] = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[req.id] = q

        meta = TOPICS[VISION_CMD]
        payload = orjson.dumps(req.model_dump(mode="json"))
        self._publish_fn(VISION_CMD, payload, meta.qos, meta.retain)

        deadline = self._clock() + timeout
        try:
            remaining = max(0.0, deadline - self._clock())
            return q.get(timeout=remaining)
        except queue.Empty:
            return VisionCmdResult(id=req.id, ok=False, detail="timeout")
        finally:
            with self._lock:
                self._pending.pop(req.id, None)

    def enroll(self, name: str, shots: int = 8, timeout_s: float = 15.0) -> VisionCmdResult:
        return self.send_command(
            "enroll",
            args={"name": name, "shots": shots},
            timeout_s=timeout_s,
        )

    def forget(self, name: str, timeout_s: float = 3.0) -> VisionCmdResult:
        return self.send_command(
            "forget",
            args={"name": name},
            timeout_s=timeout_s,
        )

    def reload_gallery(self, timeout_s: float = 3.0) -> VisionCmdResult:
        return self.send_command(
            "reload_gallery",
            args={},
            timeout_s=timeout_s,
        )

    def set_modules(
        self,
        *,
        face: bool | None = None,
        sign: bool | None = None,
        objects: bool | None = None,
        timeout_s: float = 3.0,
    ) -> VisionCmdResult:
        args: dict[str, Any] = {}
        if face is not None:
            args["face"] = face
        if sign is not None:
            args["sign"] = sign
        if objects is not None:
            args["objects"] = objects
        return self.send_command("set_modules", args=args, timeout_s=timeout_s)
