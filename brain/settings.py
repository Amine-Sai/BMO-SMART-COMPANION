"""Application settings loaded from YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict, Field


class LLMSettings(BaseModel):
    """Configuration for LLM clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"
    num_ctx: int = 4096
    keep_alive: str = "5m"
    timeout_s: float = 30.0
    temperature: float = 0.7


class SceneSettings(BaseModel):
    """Configuration for scene perception and state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_age_s: float = 60.0


class AudioSettings(BaseModel):
    """Configuration for audio capture, VAD, STT, and TTS."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_rate: int = 16000
    frame_ms: int = 20
    wake_threshold: float = 0.5
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 400
    stt_model: str = "base.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    tts_voice: str = "en_US-lessac-medium"


class Settings(BaseModel):
    """Root configuration holding all subsystem settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    llm: LLMSettings = Field(default_factory=LLMSettings)
    scene: SceneSettings = Field(default_factory=SceneSettings)
    audio: AudioSettings | None = None


def load_settings(config_dir: str | Path = "config/settings.d") -> Settings:
    """Load and merge configuration files from config_dir or settings.d."""
    path = Path(config_dir)
    if not path.exists():
        raise FileNotFoundError(f"Configuration path does not exist: {path}")

    if path.is_file():
        files = [path]
    else:
        target_dir = path / "settings.d" if (path / "settings.d").is_dir() else path
        files = sorted(target_dir.glob("*.yaml"))

    merged: dict[str, Any] = {}
    for file_path in files:
        content = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            raise ValueError(f"Expected top-level mapping in {file_path}, got {type(content).__name__}")
        for key, value in content.items():
            if key in merged:
                raise ValueError(f"Duplicate top-level key '{key}' found in {file_path}")
            merged[key] = value

    return Settings.model_validate(merged)
