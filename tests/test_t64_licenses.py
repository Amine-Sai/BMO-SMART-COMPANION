"""Tests for T6.4 license and model documentation compliance."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_requirements_lockfile_exists() -> None:
    req_file = REPO_ROOT / "requirements.lock.txt"
    assert req_file.exists(), f"Missing requirements file: {req_file}"
    content = req_file.read_text(encoding="utf-8").strip()
    assert len(content) > 0, f"Requirements file is empty: {req_file}"


def test_models_documented_in_readme() -> None:
    readme_file = REPO_ROOT / "README.md"
    assert readme_file.exists(), f"Missing README file: {readme_file}"
    readme_text = readme_file.read_text(encoding="utf-8")

    required_models = [
        "InsightFace",
        "YOLO26s",
        "faster-whisper",
        "Piper",
        "Silero VAD",
        "openWakeWord",
    ]

    for model in required_models:
        assert model.lower() in readme_text.lower(), f"Model '{model}' not found in README.md"
