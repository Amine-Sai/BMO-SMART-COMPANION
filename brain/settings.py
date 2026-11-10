"""Application settings loaded from YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict, Field


class LLMSettings(BaseModel):
    """Configuration for LLM clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str = "ollama"  # "ollama" or "llamacpp"
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"  # for ollama: base URL; for llamacpp: model path
    num_ctx: int = 4096
    keep_alive: str = "5m"
    timeout_s: float = 30.0
    temperature: float = 0.7
    history_turns: int = 6
    history_max_tokens: int = 900
    # llamacpp optimization settings
    n_gpu_layers: int = -1  # -1 = all layers on GPU (if available), 0 = CPU only
    n_threads: int = 8
    use_flash_attn: bool = True
    use_mmap: bool = True
    use_mlock: bool = False


class SceneSettings(BaseModel):
    """Configuration for scene perception and state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_age_s: float = 60.0
    max_prompt_tokens: int = 60


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
    vad_chunk_samples: int = 512
    vad_preroll_ms: int = 250
    tts_max_clause_chars: int = 80


class PolicySettings(BaseModel):
    """Configuration for proactive event policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    face_min_conf: float
    greeting_cooldown_s: float
    global_cooldown_s: float
    walk_by_window_s: float
    sign_min_conf: float
    sign_debounce_s: float
    idle_unload_s: float
    tick_interval_s: float = 1.0


class EdgeSettings(BaseModel):
    """Configuration for edge hardware and failover."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    failover_debounce: int = 3


class Settings(BaseModel):
    """Root configuration holding all subsystem settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    llm: LLMSettings = Field(default_factory=LLMSettings)
    scene: SceneSettings | None = None
    audio: AudioSettings | None = None
    policy: PolicySettings | None = None
    edge: EdgeSettings | None = None


def load_settings(settings_dir: str | Path = "config/settings.d") -> Settings:
    """Load and merge configuration files from settings_dir or settings.d."""
    path = Path(settings_dir)
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
