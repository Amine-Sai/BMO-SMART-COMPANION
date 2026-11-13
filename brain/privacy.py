from collections.abc import Mapping
import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

MAX_NUMERIC_ARRAY_LEN = 32

EVENT_ALLOWLIST: dict[str, set[str]] = {
    "face_check": {
        "embedding",
        "audio",
        "frame",
        "safe_meta",
        "name",
        "bbox",
        "conf",
        "track_id",
        "ts",
    },
    "face_identified": {
        "v",
        "id",
        "ts",
        "type",
        "data",
        "name",
        "conf",
        "track_id",
    },
    "FaceIdentified": {
        "v",
        "id",
        "ts",
        "type",
        "data",
        "name",
        "conf",
        "track_id",
    },
    "person_detected": {
        "v",
        "id",
        "ts",
        "type",
        "data",
        "track_id",
        "pos",
        "proximity",
    },
    "PersonDetected": {
        "v",
        "id",
        "ts",
        "type",
        "data",
        "track_id",
        "pos",
        "proximity",
    },
    "person_left": {
        "v",
        "id",
        "ts",
        "type",
        "data",
        "track_id",
    },
    "PersonLeft": {
        "v",
        "id",
        "ts",
        "type",
        "data",
        "track_id",
    },
    "object_seen": {
        "v",
        "id",
        "ts",
        "type",
        "data",
        "label",
        "conf",
    },
    "ObjectSeen": {
        "v",
        "id",
        "ts",
        "type",
        "data",
        "label",
        "conf",
    },
    "turn_latency": {
        "vad_end_ms",
        "stt_ms",
        "llm_ttft_ms",
        "tts_first_chunk_ms",
        "total_ms",
        "utt_id",
        "ts",
        "e2e_voice_ms",
        "turn_ms",
    },
    "remember_person": {
        "name",
        "person_name",
        "ok",
        "id",
        "detail",
        "arguments",
        "result",
        "ts",
    },
    "forget_person": {
        "name",
        "person_name",
        "ok",
        "id",
        "detail",
        "arguments",
        "result",
        "ts",
    },
    "tool_action": {
        "tool",
        "name",
        "arguments",
        "result",
        "ok",
        "id",
        "detail",
        "call_id",
        "ts",
    },
    "tool_call": {
        "name",
        "tool",
        "arguments",
        "call_id",
        "ts",
    },
    "transcript": {
        "text",
        "transcript",
        "user_text",
        "utterance",
        "utt_id",
        "conf",
        "duration_s",
        "language",
        "speaker",
        "ts",
    },
    "transcripts": {
        "text",
        "transcript",
        "user_text",
        "utterance",
        "utt_id",
        "conf",
        "duration_s",
        "language",
        "speaker",
        "ts",
    },
    "stt": {
        "text",
        "transcript",
        "user_text",
        "utterance",
        "utt_id",
        "conf",
        "duration_s",
        "ts",
    },
    "user_voice": {
        "text",
        "user_text",
        "expected_reply",
        "expected_tool",
        "expected_args",
        "ts",
    },
    "vision_status": {
        "online",
        "fps",
        "ts",
        "status",
    },
    "scene_context": {
        "ts",
        "people",
        "objects",
        "recent",
    },
    "hw_event": {
        "ts",
        "kind",
        "value",
    },
    "mute_switch": {
        "ts",
        "kind",
        "value",
    },
    "step": {
        "index",
        "filler",
        "step",
        "ts",
        "obj",
        "value",
        "data",
        "details",
    },
}

PERSON_NAME_EVENTS: set[str] = {
    "face_check",
    "face_identified",
    "FaceIdentified",
    "remember_person",
    "forget_person",
}

TRANSCRIPT_EVENTS: set[str] = {
    "transcript",
    "transcripts",
    "stt",
    "user_voice",
    "user_text",
}

TRANSCRIPT_KEYS: set[str] = {"transcript", "user_text", "utterance"}
PERSON_TOOLS: set[str] = {"remember_person", "forget_person"}


def hash_identifier(val: Any) -> Any:
    if not isinstance(val, str) or not val:
        return val
    return hashlib.sha256(val.encode("utf-8")).hexdigest()[:12]


hash_pii = hash_identifier


def _should_hash_field(
    key: str,
    val: Any,
    event: str | None,
    parent_key: str | None,
    tool: str | None,
) -> bool:
    if not isinstance(val, (str, list, tuple)):
        return False

    if key == "person_name":
        return True

    if key in TRANSCRIPT_KEYS:
        return True

    if key == "text":
        return event in TRANSCRIPT_EVENTS or event is None

    if key == "name":
        if event in PERSON_NAME_EVENTS:
            return True
        if parent_key in ("safe_meta", "data"):
            return True
        if parent_key == "arguments":
            return tool in PERSON_TOOLS or event in PERSON_TOOLS
        if event is None and parent_key is None:
            return False

    return False


def sanitize_payload(
    obj: Any,
    max_array_len: int = MAX_NUMERIC_ARRAY_LEN,
    *,
    event: str | None = None,
    redact_pii: bool = True,
    allowlist: Mapping[str, set[str]] | None = None,
    seen: set[int] | None = None,
    parent_key: str | None = None,
    tool_context: str | None = None,
) -> Any:
    if seen is None:
        seen = set()

    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)

    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<redacted:bytes:len={len(obj)}>"
    if isinstance(obj, np.ndarray):
        return f"<redacted:ndarray:shape={obj.shape}:dtype={obj.dtype}>"

    if isinstance(obj, Path):
        return str(obj)

    obj_id = id(obj)
    if isinstance(obj, (Mapping, list, tuple, set, frozenset)):
        if obj_id in seen:
            return "<circular_ref>"
        seen.add(obj_id)

    try:
        if isinstance(obj, Mapping):
            current_tool = tool_context
            if "tool" in obj and isinstance(obj["tool"], str):
                current_tool = obj["tool"]

            effective_allow = allowlist if allowlist is not None else EVENT_ALLOWLIST
            allowed_fields: set[str] | None = None
            if event is not None and parent_key is None:
                allowed_fields = effective_allow.get(event, set())

            result: dict[str, Any] = {}
            for k, v in obj.items():
                str_k = str(k)
                if allowed_fields is not None and str_k not in allowed_fields:
                    continue

                if redact_pii and _should_hash_field(
                    str_k, v, event=event, parent_key=parent_key, tool=current_tool
                ):
                    if isinstance(v, (list, tuple)):
                        sanitized_v = [hash_identifier(x) for x in v]
                    else:
                        sanitized_v = hash_identifier(v)
                else:
                    sanitized_v = sanitize_payload(
                        v,
                        max_array_len,
                        event=event,
                        redact_pii=redact_pii,
                        allowlist=allowlist,
                        seen=seen,
                        parent_key=str_k,
                        tool_context=current_tool,
                    )
                result[str_k] = sanitized_v
            return result

        if isinstance(obj, (list, tuple)):
            if len(obj) > max_array_len and all(
                isinstance(x, (int, float, np.number)) and not isinstance(x, bool) for x in obj
            ):
                return f"<redacted:numeric_array:len={len(obj)}>"
            return [
                sanitize_payload(
                    x,
                    max_array_len,
                    event=event,
                    redact_pii=redact_pii,
                    allowlist=allowlist,
                    seen=seen,
                    parent_key=parent_key,
                    tool_context=tool_context,
                )
                for x in obj
            ]

        if isinstance(obj, (set, frozenset)):
            return [
                sanitize_payload(
                    x,
                    max_array_len,
                    event=event,
                    redact_pii=redact_pii,
                    allowlist=allowlist,
                    seen=seen,
                    parent_key=parent_key,
                    tool_context=tool_context,
                )
                for x in sorted(obj, key=str)
            ]

        return obj
    finally:
        if isinstance(obj, (Mapping, list, tuple, set, frozenset)):
            seen.discard(obj_id)
