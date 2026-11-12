from __future__ import annotations

from brain.memory import LongTermMemory
from brain.memory_extract import apply_memory_effects, extract_memories, extract_forget
from brain.prompt import PromptBuilder
from brain.scene import SceneCache
from brain.settings import SceneSettings
from brain.tools.memory_tools import register_memory_tools
from brain.tools.registry import ToolRegistry


def test_long_term_memory_persists(tmp_path):
    path = tmp_path / "memory.json"
    memory = LongTermMemory(path)
    memory.remember("User name", "Judas")
    memory.remember("favorite color", "blue")

    reloaded = LongTermMemory(path)
    assert reloaded.recall("user name")["value"] == "Judas"
    assert reloaded.recall("favorite color")["value"] == "blue"


def test_extract_memories_and_forget():
    facts = extract_memories("Hey BMO, remember the wifi password is hunter2")
    assert ("wifi password", "hunter2", "note") in facts
    assert extract_memories("my name is Judas") == [("user name", "Judas", "identity")]
    assert extract_memories("my favorite color is blue") == [
        ("favorite color", "blue", "preference")
    ]
    assert extract_forget("forget my favorite color") == "my favorite color"


def test_apply_memory_effects(tmp_path):
    memory = LongTermMemory(tmp_path / "memory.json")
    apply_memory_effects("remember that I like pizza", memory)
    assert memory.recall("likes")["value"] == "pizza"

    apply_memory_effects("forget likes", memory)
    assert memory.recall("likes") is None


def test_prompt_includes_saved_memory(tmp_path):
    memory = LongTermMemory(tmp_path / "memory.json")
    memory.remember("user name", "Judas")
    builder = PromptBuilder(SceneCache(SceneSettings()), memory=memory)

    messages = builder.build_messages("what is my name", scene_target=None)

    system = messages[0]["content"]
    assert "MEMORY" in system
    assert "user name: Judas" in system


def test_memory_tools_roundtrip(tmp_path):
    memory = LongTermMemory(tmp_path / "memory.json")
    registry = ToolRegistry()
    register_memory_tools(registry, memory)

    saved = registry.dispatch(
        {"name": "remember_fact", "arguments": {"key": "home", "value": "Berlin"}}
    )
    assert saved["ok"] is True
    assert memory.recall("home")["value"] == "Berlin"

    recalled = registry.dispatch({"name": "recall_fact", "arguments": {"key": "home"}})
    assert recalled["found"] is True
