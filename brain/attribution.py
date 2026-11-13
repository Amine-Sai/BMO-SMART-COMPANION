"""Speaker attribution based on visual scene context."""

from __future__ import annotations

from bmo_contracts.models import Scene
from brain.scene import SceneCache


def _is_known(name: str) -> bool:
    stripped = name.strip().lower() if name else ""
    return bool(stripped and stripped not in ("unknown", "person", "stranger", "?"))


def attribute_speaker(scene_or_cache: Scene | SceneCache | None) -> str:
    """Attribute a spoken utterance to a person in the scene."""
    if isinstance(scene_or_cache, SceneCache):
        scene = scene_or_cache.current()
    else:
        scene = scene_or_cache

    if scene is None or not scene.people:
        return "unknown"

    if len(scene.people) == 1:
        p = scene.people[0]
        return p.name if _is_known(p.name) else "unknown"

    candidates = [p for p in scene.people if p.pos == "center" and p.proximity == "near"]
    if len(candidates) == 1 and _is_known(candidates[0].name):
        return candidates[0].name

    return "unknown"


def format_attributed_text(text: str, scene_or_cache: Scene | SceneCache | None) -> str:
    """Format utterance with attributed speaker prefix if known."""
    speaker = attribute_speaker(scene_or_cache)
    if speaker != "unknown":
        return f"[{speaker}]: {text}"
    return text
