from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    host: str
    num_ctx: int
    keep_alive: str | int
    timeout_s: float
    temperature: float


class SceneSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_age_s: float


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    llm: LLMSettings
    scene: SceneSettings | None = None
    audio: Any | None = None
    policy: Any | None = None


def load_settings(settings_dir: Path | str = "config/settings.d") -> Settings:
    directory = Path(settings_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Settings directory not found: {directory}")

    payload: dict[str, Any] = {}
    for yaml_path in sorted(directory.glob("*.yaml")):
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(content, dict):
            raise ValueError(f"YAML file must contain a top-level mapping: {yaml_path}")
        for section_key, section_data in content.items():
            if section_key in payload:
                raise ValueError(f"Duplicate top-level key {section_key!r} found in {yaml_path.name}")
            payload[section_key] = section_data

    return Settings.model_validate(payload)
