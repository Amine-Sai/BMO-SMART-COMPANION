from __future__ import annotations

from bmo_contracts.models import Scene, ScenePerson, VisionStatus
from brain.attribution import attribute_speaker, format_attributed_text
from brain.scene import SceneCache
from brain.settings import SceneSettings


def _make_person(name: str, pos: str = "center", proximity: str = "near") -> ScenePerson:
    return ScenePerson(track_id=1, name=name, pos=pos, proximity=proximity, since_s=0.0)


def _make_scene(people: list[ScenePerson], ts: float = 100.0) -> Scene:
    return Scene(ts=ts, people=people, objects=[], recent=[])


def test_case_1_zero_people_or_none_or_stale_or_offline() -> None:
    assert attribute_speaker(None) == "unknown"
    assert attribute_speaker(_make_scene([])) == "unknown"

    now = 100.0
    cache = SceneCache(settings=SceneSettings(), clock=lambda: now)
    cache.update_status(VisionStatus(online=True))
    cache.update_scene(_make_scene([_make_person("Jud")], ts=now))
    assert attribute_speaker(cache) == "Jud"

    # Stale cache returns None from current()
    now += 1000.0
    assert attribute_speaker(cache) == "unknown"

    # Offline cache returns None from current()
    now = 100.0
    cache.update_status(VisionStatus(online=False))
    assert attribute_speaker(cache) == "unknown"


def test_case_2_single_known_person() -> None:
    scene = _make_scene([_make_person("Jud")])
    assert attribute_speaker(scene) == "Jud"


def test_case_3_single_unknown_person() -> None:
    for name in ("unknown", "person", "stranger", "?", "", "   "):
        scene = _make_scene([_make_person(name)])
        assert attribute_speaker(scene) == "unknown"


def test_case_4_two_people_one_center_near() -> None:
    scene = _make_scene(
        [
            _make_person("Jud", pos="center", proximity="near"),
            _make_person("Alice", pos="left", proximity="far"),
        ]
    )
    assert attribute_speaker(scene) == "Jud"


def test_case_5_two_people_both_center_near() -> None:
    scene = _make_scene(
        [
            _make_person("Jud", pos="center", proximity="near"),
            _make_person("Alice", pos="center", proximity="near"),
        ]
    )
    assert attribute_speaker(scene) == "unknown"


def test_case_6_two_people_neither_center_near() -> None:
    scene = _make_scene(
        [
            _make_person("Jud", pos="left", proximity="near"),
            _make_person("Alice", pos="right", proximity="far"),
        ]
    )
    assert attribute_speaker(scene) == "unknown"


def test_format_attributed_text_with_scene_cache() -> None:
    now = 100.0
    cache = SceneCache(settings=SceneSettings(), clock=lambda: now)
    cache.update_status(VisionStatus(online=True))
    cache.update_scene(_make_scene([_make_person("Jud")], ts=now))
    assert format_attributed_text("hello", cache) == "[Jud]: hello"

    # When speaker is unknown, raw text is returned
    cache.update_scene(_make_scene([_make_person("unknown")], ts=now))
    assert format_attributed_text("hello", cache) == "hello"

    # When scene is None
    assert format_attributed_text("hello", None) == "hello"
