"""Unit tests verifying all new settings tunables and edge configuration."""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest
from pydantic import ValidationError

from brain.settings import (
    AudioSettings,
    EdgeSettings,
    LLMSettings,
    PolicySettings,
    SceneSettings,
    Settings,
    load_settings,
)


def test_load_settings_contains_all_tunables_with_defaults() -> None:
    """Verify that load_settings loads all expected keys from config/settings.d."""
    settings = load_settings()
    assert isinstance(settings, Settings)

    # scene
    assert settings.scene is not None
    assert settings.scene.max_prompt_tokens == 60

    # llm
    assert settings.llm.history_turns == 3
    assert settings.llm.history_max_tokens == 250

    # audio
    assert settings.audio is not None
    assert settings.audio.vad_chunk_samples == 512
    assert settings.audio.vad_preroll_ms == 250
    assert settings.audio.tts_max_clause_chars == 80

    # edge
    assert settings.edge is not None
    assert settings.edge.failover_debounce == 3

    # policy
    assert settings.policy is not None
    assert settings.policy.tick_interval_s == 1.0


def test_settings_classes_default_instantiation() -> None:
    """Verify default values on individual model classes without YAML input."""
    assert SceneSettings().max_prompt_tokens == 60
    assert LLMSettings().history_turns == 3
    assert LLMSettings().history_max_tokens == 250
    assert AudioSettings().vad_chunk_samples == 512
    assert AudioSettings().vad_preroll_ms == 250
    assert AudioSettings().tts_max_clause_chars == 80
    assert EdgeSettings().failover_debounce == 3

    policy = PolicySettings(
        face_min_conf=0.65,
        greeting_cooldown_s=900.0,
        global_cooldown_s=30.0,
        walk_by_window_s=1.5,
        sign_min_conf=0.75,
        sign_debounce_s=2.0,
        idle_unload_s=300.0,
    )
    assert policy.tick_interval_s == 1.0


def test_edge_settings_frozen_and_extra_forbidden() -> None:
    """Verify EdgeSettings enforces immutability and forbids unknown keys."""
    edge = EdgeSettings()
    assert edge.failover_debounce == 3

    with pytest.raises(ValidationError):
        edge.failover_debounce = 5  # type: ignore[misc]

    with pytest.raises(ValidationError):
        EdgeSettings(unknown="foo")  # type: ignore[call-arg]


def test_edge_yaml_loaded_standalone() -> None:
    """Verify edge.yaml loads correctly in isolation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        edge_file = Path(temp_dir) / "edge.yaml"
        edge_file.write_text("edge:\n  failover_debounce: 5\n", encoding="utf-8")
        settings = load_settings(temp_dir)
        assert settings.edge is not None
        assert settings.edge.failover_debounce == 5
