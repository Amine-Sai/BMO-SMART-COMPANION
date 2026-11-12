from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any
from bmo_contracts.models import Scene, VisionStatus
from bmo_contracts.topics import VISION_CONTEXT, VISION_STATUS
from brain.llm import count_tokens
from brain.settings import SceneSettings


def _format_person(p: Any) -> str:
    if isinstance(p, str):
        return p
    name = getattr(p, "name", None) or (p.get("name") if isinstance(p, dict) else None) or "person"
    pos = (
        getattr(p, "pos", None)
        or getattr(p, "position", None)
        or getattr(p, "zone", None)
        or getattr(p, "location", None)
    )
    if pos is None and isinstance(p, dict):
        pos = p.get("pos") or p.get("position") or p.get("zone") or p.get("location")
    dist = getattr(p, "proximity", None) or getattr(p, "distance", None) or getattr(p, "depth", None)
    if dist is None and isinstance(p, dict):
        dist = p.get("proximity") or p.get("distance") or p.get("depth")
    dur = (
        getattr(p, "since_s", None)
        or getattr(p, "duration_s", None)
        or getattr(p, "duration", None)
        or getattr(p, "seen_s", None)
        or getattr(p, "age_s", None)
    )
    if dur is None and isinstance(p, dict):
        dur = p.get("since_s") or p.get("duration_s") or p.get("duration") or p.get("seen_s") or p.get("age_s")

    details: list[str] = []
    if pos:
        details.append(str(pos))
    if dist:
        details.append(str(dist))
    if dur is not None:
        dur_str = f"{int(dur)}s" if isinstance(dur, (int, float)) else str(dur)
        details.append(dur_str)

    if details:
        return f"{name}({','.join(details)})"
    return str(name)


def _format_object(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return str(obj.get("name") or obj.get("label") or obj)
    return str(getattr(obj, "name", None) or getattr(obj, "label", None) or obj)


def _format_sign(val: Any) -> str:
    if not val:
        return "none"
    if isinstance(val, list):
        if not val:
            return "none"
        val = val[0]
    if isinstance(val, str):
        return val
    text = getattr(val, "text", None) or (val.get("text") if isinstance(val, dict) else None)
    if not text:
        text = str(val)
    age = (
        getattr(val, "age_s", None)
        or getattr(val, "duration_s", None)
        or getattr(val, "ago_s", None)
        or getattr(val, "age", None)
    )
    if age is None and isinstance(val, dict):
        age = val.get("age_s") or val.get("duration_s") or val.get("ago_s") or val.get("age")
    if age is not None:
        age_str = f"{int(age)}s ago" if isinstance(age, (int, float)) else str(age)
        return f"{text}({age_str})"
    return str(text)


def _get_field(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
    return None


class SceneCache:
    def __init__(
        self,
        settings: SceneSettings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._scene: Scene | None = None
        self._updated_at: float = 0.0
        self._online: bool = False
        self._lock = threading.RLock()

    def update_scene(self, scene: Scene | bytes | str) -> None:
        if isinstance(scene, Scene):
            parsed = scene
        elif isinstance(scene, (bytes, str)):
            parsed = Scene.model_validate_json(scene)
        else:
            parsed = Scene.model_validate(scene)
        with self._lock:
            self._scene = parsed
            self._updated_at = self._clock()

    def update_status(self, status: VisionStatus | bytes | str) -> None:
        if isinstance(status, VisionStatus):
            parsed = status
        elif isinstance(status, (bytes, str)):
            parsed = VisionStatus.model_validate_json(status)
        else:
            parsed = VisionStatus.model_validate(status)
        with self._lock:
            self._online = bool(parsed.online)

    def handle_mqtt_message(self, topic: str, payload: bytes | str | None = None) -> None:
        if payload is None:
            raw = topic
            if isinstance(raw, VisionStatus) or (
                isinstance(raw, (bytes, str)) and ('"online"' in str(raw) or "'online'" in str(raw))
            ):
                self.update_status(raw)
            else:
                self.update_scene(raw)
            return

        if topic == VISION_CONTEXT:
            self.update_scene(payload)
        elif topic == VISION_STATUS:
            self.update_status(payload)

    def is_online(self) -> bool:
        with self._lock:
            return self._online

    def last_update_time(self) -> float:
        """Monotonic timestamp of the last scene update (0.0 if none)."""
        with self._lock:
            return self._updated_at

    def is_stale(self) -> bool:
        with self._lock:
            if not self._online:
                return True
            if self._scene is None:
                return True
            if (self._clock() - self._updated_at) > self._settings.max_age_s:
                return True
            return False

    def current(self) -> Scene | None:
        with self._lock:
            if self.is_stale():
                return None
            return self._scene

    def as_prompt_line(self, target: str | None = None) -> str:
        with self._lock:
            if not self._online or self.is_stale() or self._scene is None:
                return "Camera: no fresh view available"
            scene = self._scene

        raw_people = _get_field(scene, "people", "persons")
        people_list: list[str] = []
        if isinstance(raw_people, list):
            people_list = [_format_person(p) for p in raw_people if p]
        elif raw_people:
            people_list = [_format_person(raw_people)]

        raw_objects = _get_field(scene, "objects", "items")
        objects_list: list[str] = []
        if isinstance(raw_objects, list):
            objects_list = [_format_object(o) for o in raw_objects if o]
        elif raw_objects:
            objects_list = [_format_object(raw_objects)]

        raw_sign = _get_field(scene, "last_sign", "sign", "text_sign", "signs")
        sign_str = _format_sign(raw_sign)

        raw_recent = _get_field(scene, "recent", "recent_events", "events")
        recent_list: list[str] = []
        if isinstance(raw_recent, list):
            recent_list = [str(r) for r in raw_recent if r]
        elif raw_recent:
            recent_list = [str(raw_recent)]

        # Only surface the fields the user actually asked about.
        if target in ("people", "person", "face"):
            objects_list = []
            sign_str = "none"
            recent_list = []
        elif target in ("objects", "object", "thing", "item"):
            people_list = []
            sign_str = "none"
            recent_list = []

        def render(p_items: list[str], o_items: list[str], s_val: str, r_items: list[str]) -> str:
            p_str = ", ".join(p_items) if p_items else "none"
            o_str = ",".join(o_items) if o_items else "none"
            r_str = ", ".join(r_items) if r_items else "none"
            return f"Camera: online | people: {p_str} | objects: {o_str} | sign: {s_val} | recent: {r_str}"

        line = render(people_list, objects_list, sign_str, recent_list)
        if count_tokens(line) <= 60:
            return line

        while recent_list and count_tokens(render(people_list, objects_list, sign_str, recent_list)) > 60:
            recent_list.pop()

        line = render(people_list, objects_list, sign_str, recent_list)
        if count_tokens(line) <= 60:
            return line

        while objects_list and count_tokens(render(people_list, objects_list, sign_str, recent_list)) > 60:
            objects_list.pop()

        line = render(people_list, objects_list, sign_str, recent_list)
        if count_tokens(line) <= 60:
            return line

        while people_list and count_tokens(render(people_list, objects_list, sign_str, recent_list)) > 60:
            people_list.pop()

        line = render(people_list, objects_list, sign_str, recent_list)
        if count_tokens(line) <= 60:
            return line

        sign_str = "none"
        line = render(people_list, objects_list, sign_str, recent_list)
        if count_tokens(line) <= 60:
            return line

        while count_tokens(line) > 60 and len(line) > 15:
            line = line[:-1]
        return line
