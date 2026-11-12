from unittest.mock import Mock
from brain.llm import count_tokens
from brain.prompt import BMO_SYSTEM_PROMPT, PromptBuilder, RollingHistory
from brain.scene import SceneCache


def _make_scene_cache() -> SceneCache:
    try:
        from brain.config import BrainSettings

        return SceneCache(settings=BrainSettings())
    except Exception:
        cache = Mock(spec=SceneCache)
        cache.as_prompt_line.return_value = "Camera: offline/stale"
        return cache


def test_system_prompt_tokens_and_format() -> None:
    tokens = count_tokens(BMO_SYSTEM_PROMPT)
    assert tokens <= 220
    assert "markdown" not in BMO_SYSTEM_PROMPT.lower()
    for bullet in ("- ", "* ", "•"):
        assert bullet not in BMO_SYSTEM_PROMPT


def test_rolling_history_basic() -> None:
    history = RollingHistory(max_turns=2, max_total_tokens=200)
    assert history.max_turns == 2
    assert history.max_total_tokens == 200

    history.add_message("user", "hi")
    assert len(history.messages) == 1
    history.add_turn("turn user", "turn assistant")
    assert len(history.messages) <= 4

    history.clear()
    assert len(history.messages) == 0


def test_prompt_builder_fifty_turns_simulation() -> None:
    cache = _make_scene_cache()
    builder = PromptBuilder(scene_cache=cache, max_turns=6, max_total_tokens=900)
    assert builder.scene_cache is cache
    assert builder.system_prompt == BMO_SYSTEM_PROMPT
    assert builder.max_turns == 6
    assert builder.max_total_tokens == 900
    assert isinstance(builder.history, RollingHistory)

    for i in range(50):
        user_text = f"User input turn {i} asking BMO about the environment"
        messages = builder.build_messages(user_text)
        assistant_text = f"BMO response turn {i} everything looks great!"
        builder.record_turn(user_text, assistant_text)

        assert len(messages) <= 14
        total_tokens = sum(count_tokens(m["content"]) + 4 for m in messages)
        assert total_tokens <= 900

    final_messages = builder.build_messages("Final check")
    assert len(final_messages) == 14
    assert sum(count_tokens(m["content"]) + 4 for m in final_messages) <= 900


def test_scene_cache_prompt_injection() -> None:
    cache = _make_scene_cache()
    builder = PromptBuilder(scene_cache=cache)

    # Non-vision questions must not include the camera scene.
    msgs = builder.build_messages("Hello BMO")
    assert "Scene:" not in msgs[0]["content"]

    cache.as_prompt_line = Mock(return_value="Camera: 1 person (Alice, 1.2m, waving), 1 cup")
    msgs_online = builder.build_messages("What do you see?", scene_target="both")
    assert "Scene: Camera: 1 person (Alice, 1.2m, waving), 1 cup" in msgs_online[0]["content"]
    cache.as_prompt_line.assert_called_with(target="both")
