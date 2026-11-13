from __future__ import annotations

from typing import Any

from brain.scene import SceneCache
from brain.tools.registry import ToolRegistry, ToolSpec
from brain.tools.schemas import EmptyArgs, FindObjectArgs


class SceneTools:
    """Read-only scene tools for querying the current scene state."""

    def __init__(self, scene_cache: SceneCache) -> None:
        self._scene_cache = scene_cache

    def get_scene(self, _args: EmptyArgs | dict[str, Any] | None = None, **_kwargs: Any) -> dict[str, Any]:
        scene = self._scene_cache.current()
        if scene is None:
            return {"ok": False, "error": "scene_stale"}
        return {"ok": True, "scene": scene.model_dump(mode="json")}

    def find_object(self, args: FindObjectArgs | dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        if args is None:
            parsed = FindObjectArgs.model_validate(kwargs)
        elif isinstance(args, dict):
            parsed = FindObjectArgs.model_validate(args)
        else:
            parsed = args

        scene = self._scene_cache.current()
        if scene is None:
            return {"ok": False, "error": "scene_stale"}

        query = parsed.name.strip().lower()
        found = query in [obj.lower() for obj in scene.objects]
        return {
            "ok": True,
            "name": parsed.name,
            "found": found,
            "objects": list(scene.objects),
        }

    def list_visible_people(self, _args: EmptyArgs | dict[str, Any] | None = None, **_kwargs: Any) -> dict[str, Any]:
        scene = self._scene_cache.current()
        if scene is None:
            return {"ok": False, "error": "scene_stale"}
        return {
            "ok": True,
            "people": [p.model_dump(mode="json") for p in scene.people],
        }


def register_scene_tools(registry: ToolRegistry, scene_cache: SceneCache) -> SceneTools:
    """Register read-only scene tools with the given tool registry."""
    tools = SceneTools(scene_cache)
    registry.register_if_absent(
        ToolSpec(
            name="get_scene",
            description="Get the current scene description and detected objects or people.",
            schema=EmptyArgs,
            handler=tools.get_scene,
            risky=False,
        )
    )
    registry.register_if_absent(
        ToolSpec(
            name="find_object",
            description="Check if a specific object is visible in the current scene.",
            schema=FindObjectArgs,
            handler=tools.find_object,
            risky=False,
        )
    )
    registry.register_if_absent(
        ToolSpec(
            name="list_visible_people",
            description="List people currently visible in the scene.",
            schema=EmptyArgs,
            handler=tools.list_visible_people,
            risky=False,
        )
    )
    return tools
