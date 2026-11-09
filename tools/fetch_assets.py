"""Fetch and verify model assets for BMO Assistant (T0.6)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import shutil
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


def check_ollama(client_check_fn: Callable[[], bool] | None = None) -> dict[str, object]:
    """Check Ollama installation and daemon status."""
    user_ollama = Path.home() / ".local" / "bin" / "ollama"
    ollama_bin = shutil.which("ollama") or (str(user_ollama) if user_ollama.is_file() else None)

    daemon_running = False
    if client_check_fn is not None:
        daemon_running = client_check_fn()
    else:
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "bmo-probe"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                daemon_running = resp.status == 200
        except Exception:
            daemon_running = False

    return {
        "engine": "ollama",
        "installed": ollama_bin is not None,
        "path": ollama_bin,
        "daemon_running": daemon_running,
        "escalation": "H2: sudo pacman -S ollama-cuda && sudo systemctl enable --now ollama"
        if not ollama_bin or not daemon_running
        else None,
    }


def fetch_openwakeword_models(
    model_names: list[str] | None = None,
    download_fn: Callable[[list[str]], None] | None = None,
) -> dict[str, object]:
    """Verify or download stock openWakeWord models."""
    names = model_names or ["hey_jarvis_v0.1"]
    import openwakeword
    import openwakeword.utils

    models_dir = Path(openwakeword.__file__).parent / "resources" / "models"

    missing = []
    for name in names:
        target_name = f"{name}.onnx" if not name.endswith(".onnx") else name
        if not (models_dir / target_name).is_file():
            missing.append(name)

    if missing:
        if download_fn is not None:
            download_fn(missing)
        else:
            try:
                openwakeword.utils.download_models(missing)
            except Exception as exc:
                return {
                    "engine": "openwakeword",
                    "ok": False,
                    "models_dir": str(models_dir),
                    "missing": missing,
                    "error": str(exc),
                }

    present = [p.name for p in models_dir.glob("*.onnx")] if models_dir.is_dir() else []
    return {
        "engine": "openwakeword",
        "ok": any("hey_jarvis" in p for p in present),
        "models_dir": str(models_dir),
        "models": present,
    }


def fetch_piper_voice(
    voice_name: str = "en_US-lessac-medium",
    target_dir: Path | None = None,
    fetch_remote: bool = True,
) -> dict[str, object]:
    """Verify or download Piper ONNX voice model and config."""
    out_dir = target_dir or (DEFAULT_MODELS_DIR / "piper")
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_file = out_dir / f"{voice_name}.onnx"
    json_file = out_dir / f"{voice_name}.onnx.json"

    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"

    if fetch_remote and (not onnx_file.is_file() or not json_file.is_file()):
        for file_path, remote_name in [
            (onnx_file, f"{voice_name}.onnx"),
            (json_file, f"{voice_name}.onnx.json"),
        ]:
            if not file_path.is_file():
                url = f"{base_url}/{remote_name}"
                try:
                    urllib.request.urlretrieve(url, str(file_path))
                except Exception as exc:
                    return {
                        "engine": "piper",
                        "ok": False,
                        "voice": voice_name,
                        "dir": str(out_dir),
                        "error": f"Failed to download {remote_name}: {exc}",
                    }

    return {
        "engine": "piper",
        "ok": onnx_file.is_file() and json_file.is_file(),
        "voice": voice_name,
        "onnx_path": str(onnx_file) if onnx_file.is_file() else None,
        "json_path": str(json_file) if json_file.is_file() else None,
    }


def fetch_faster_whisper(
    model_size: str = "base.en",
    download_fn: Callable[[str], str] | None = None,
) -> dict[str, object]:
    """Verify or download Faster-Whisper CT2 weights."""
    import faster_whisper

    try:
        if download_fn is not None:
            cache_dir = download_fn(model_size)
        else:
            cache_dir = faster_whisper.download_model(model_size)
        return {
            "engine": "faster-whisper",
            "ok": True,
            "model": model_size,
            "cache_dir": str(cache_dir),
        }
    except Exception as exc:
        return {
            "engine": "faster-whisper",
            "ok": False,
            "model": model_size,
            "error": str(exc),
        }


def fetch_all(check_only: bool = False) -> dict[str, object]:
    """Run verification or download for all 4 engines."""
    results: dict[str, object] = {
        "ollama": check_ollama(),
        "openwakeword": fetch_openwakeword_models()
        if not check_only
        else {"engine": "openwakeword", "check_only": True},
        "piper": fetch_piper_voice(fetch_remote=not check_only),
        "faster_whisper": fetch_faster_whisper()
        if not check_only
        else {"engine": "faster_whisper", "check_only": True},
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify BMO Assistant model assets.")
    parser.add_argument("--check-only", action="store_true", help="Only verify existing assets without downloading")
    args = parser.parse_args()

    print("Checking BMO model assets...")
    results = fetch_all(check_only=args.check_only)
    all_ok = True

    for name, res in results.items():
        print(f"[{name}] {res}")
        if isinstance(res, dict) and res.get("ok") is False:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
