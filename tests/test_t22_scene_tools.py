from __future__ import annotations

from typing import Any

from bmo_contracts.models import LastSign, Scene, ScenePerson, VisionStatus
from brain.scene import SceneCache
from brain.settings import SceneSettings
from brain.tools.registry import ToolRegistry
from brain.tools.schemas import EmptyArgs, FindObjectArgs
from brain.tools.scene_tools import SceneTools, register_scene_tools


def _dispatch(registry: ToolRegistry, name: str, args: Any = None) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    if hasattr(args, "model_dump"):
        arguments = args.model_dump()
    elif isinstance(args, dict):
        arguments = args
    return registry.dispatch({"name": name, "arguments": arguments})


def _make_scene_cache(clock_fn: Any, max_age_s: float = 5.0) -> SceneCache:
    return SceneCache(settings=SceneSettings(max_age_s=max_age_s), clock=clock_fn)


def _make_person(name: str = "Alice") -> ScenePerson:
    return ScenePerson(
        name=name,
        pos="center",
        proximity="near",
        since_s=1.0,
    )


def _make_scene(objects: list[str], people: list[ScenePerson]) -> Scene:
    return Scene(
        ts=100.0,
        people=people,
        objects=objects,
        recent=[],
    )


def test_scene_tools_fresh() -> None:
    assert LastSign is not None
    current_time = [100.0]
    cache = _make_scene_cache(lambda: current_time[0], max_age_s=5.0)

    person = _make_person("Alice")
    scene = _make_scene(objects=["cup", "laptop"], people=[person])

    cache.update_status(VisionStatus(online=True))
    cache.update_scene(scene)

    registry = ToolRegistry()
    tools = register_scene_tools(registry, cache)
    assert isinstance(tools, SceneTools)

    direct_scene = tools.get_scene()
    assert direct_scene["ok"] is True
    assert direct_scene["scene"] == scene.model_dump(mode="json")

    direct_found = tools.find_object(FindObjectArgs(name="cup"))
    assert direct_found["ok"] is True
    assert direct_found["found"] is True

    direct_people = tools.list_people()
    assert direct_people["ok"] is True
    assert len(direct_people["people"]) == 1

    res_scene = _dispatch(registry, "get_scene", EmptyArgs())
    assert res_scene["ok"] is True
    assert res_scene["scene"] == scene.model_dump(mode="json")

    res_found = _dispatch(registry, "find_object", FindObjectArgs(name="cup"))
    assert res_found["ok"] is True
    assert res_found["found"] is True
    assert res_found["name"] == "cup"
    assert "cup" in res_found["objects"]

    res_case = _dispatch(registry, "find_object", FindObjectArgs(name="  CUP  "))
    assert res_case["ok"] is True
    assert res_case["found"] is True

    res_not_found = _dispatch(registry, "find_object", FindObjectArgs(name="banana"))
    assert res_not_found["ok"] is True
    assert res_not_found["found"] is False

    res_people = _dispatch(registry, "list_people", EmptyArgs())
    assert res_people["ok"] is True
    assert res_people["people"] == [p.model_dump(mode="json") for p in scene.people]


def test_scene_tools_stale_before_scene() -> None:
    current_time = [100.0]
    cache = _make_scene_cache(lambda: current_time[0], max_age_s=5.0)
    registry = ToolRegistry()
    register_scene_tools(registry, cache)

    res_scene = _dispatch(registry, "get_scene", EmptyArgs())
    assert res_scene == {"ok": False, "error": "scene_stale"}

    res_find = _dispatch(registry, "find_object", FindObjectArgs(name="cup"))
    assert res_find == {"ok": False, "error": "scene_stale"}

    res_people = _dispatch(registry, "list_people", EmptyArgs())
    assert res_people == {"ok": False, "error": "scene_stale"}


def test_scene_tools_stale_vision_offline() -> None:
    current_time = [100.0]
    cache = _make_scene_cache(lambda: current_time[0], max_age_s=5.0)
    person = _make_person("Bob")
    scene = _make_scene(objects=["mug"], people=[person])

    cache.update_status(VisionStatus(online=True))
    cache.update_scene(scene)

    registry = ToolRegistry()
    register_scene_tools(registry, cache)

    cache.update_status(VisionStatus(online=False))

    res_scene = _dispatch(registry, "get_scene", EmptyArgs())
    assert res_scene == {"ok": False, "error": "scene_stale"}

    res_find = _dispatch(registry, "find_object", FindObjectArgs(name="mug"))
    assert res_find == {"ok": False, "error": "scene_stale"}

    res_people = _dispatch(registry, "list_people", EmptyArgs())
    assert res_people == {"ok": False, "error": "scene_stale"}


def test_scene_tools_stale_clock_expired() -> None:
    current_time = [100.0]
    cache = _make_scene_cache(lambda: current_time[0], max_age_s=5.0)
    person = _make_person("Charlie")
    scene = _make_scene(objects=["bottle"], people=[person])

    cache.update_status(VisionStatus(online=True))
    cache.update_scene(scene)

    registry = ToolRegistry()
    register_scene_tools(registry, cache)

    current_time[0] += 10.0

    res_scene = _dispatch(registry, "get_scene", EmptyArgs())
    assert res_scene == {"ok": False, "error": "scene_stale"}

    res_find = _dispatch(registry, "find_object", FindObjectArgs(name="bottle"))
    assert res_find == {"ok": False, "error": "scene_stale"}

    res_people = _dispatch(registry, "list_people", EmptyArgs())
    assert res_people == {"ok": False, "error": "scene_stale"}
