"""Tests for T6.4 license documentation compliance."""

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_requirements_documented_in_licenses_md() -> None:
    req_file = REPO_ROOT / "requirements" / "brain.txt"
    licenses_file = REPO_ROOT / "docs" / "licenses.md"

    assert req_file.exists(), f"Missing requirements file: {req_file}"
    assert licenses_file.exists(), f"Missing licenses file: {licenses_file}"

    licenses_text = licenses_file.read_text(encoding="utf-8")

    packages: list[str] = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg_name = re.split(r"[><=~!;]", line)[0].strip()
        if pkg_name:
            packages.append(pkg_name)

    assert len(packages) > 0, "No packages found in requirements/brain.txt"

    for pkg in packages:
        pattern = rf"\|\s*`?{re.escape(pkg)}`?\s*\|\s*([^|\s]+[^|]*)\s*\|"
        match = re.search(pattern, licenses_text, flags=re.IGNORECASE)
        assert match is not None, f"Package '{pkg}' missing from docs/licenses.md table"
        license_entry = match.group(1).strip()
        assert len(license_entry) > 0, f"Package '{pkg}' has empty license entry"


def test_models_documented_in_licenses_md() -> None:
    licenses_file = REPO_ROOT / "docs" / "licenses.md"
    assert licenses_file.exists(), f"Missing licenses file: {licenses_file}"
    licenses_text = licenses_file.read_text(encoding="utf-8")

    required_models = [
        "buffalo_s",
        "YOLO26s",
        "Qwen2.5",
        "faster-whisper",
        "Piper",
        "Silero VAD",
        "openWakeWord",
    ]

    for model in required_models:
        assert model.lower() in licenses_text.lower(), f"Model '{model}' not found in docs/licenses.md"

    assert "non-commercial" in licenses_text.lower(), (
        "buffalo_s non-commercial restriction note missing from docs/licenses.md"
    )
