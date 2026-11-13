"""Fetch and verify model assets for BMO Assistant (T0.6 / T13)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
from pathlib import Path
import shutil
import sys
from typing import Any
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
CV_MODELS_DIR = REPO_ROOT / "assistant_cv" / "models"

try:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from assistant_cv.config import MODELS_DIR as _CONFIG_CV_MODELS_DIR

    CV_MODELS_DIR = _CONFIG_CV_MODELS_DIR
except Exception:
    pass


def get_openwakeword_dir() -> Path:
    """Resolve directory where openWakeWord looks for resource models."""
    try:
        import openwakeword

        return Path(openwakeword.__file__).parent / "resources" / "models"
    except Exception:
        return (
            REPO_ROOT
            / ".venv"
            / "lib"
            / "python3.14"
            / "site-packages"
            / "openwakeword"
            / "resources"
            / "models"
        )


ASSETS: dict[str, dict[str, Any]] = {
    "piper_voice": {
        "name": "piper_voice",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "path": DEFAULT_MODELS_DIR / "piper" / "en_US-lessac-medium.onnx",
        "sha256": "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
    },
    "piper_config": {
        "name": "piper_config",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "path": DEFAULT_MODELS_DIR / "piper" / "en_US-lessac-medium.onnx.json",
        "sha256": "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
    },
    "hey_jarvis_v0.1": {
        "name": "hey_jarvis_v0.1",
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",
        "path": get_openwakeword_dir() / "hey_jarvis_v0.1.onnx",
        "sha256": "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
    },
    "melspectrogram": {
        "name": "melspectrogram",
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
        "path": get_openwakeword_dir() / "melspectrogram.onnx",
        "sha256": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    },
    "embedding_model": {
        "name": "embedding_model",
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
        "path": get_openwakeword_dir() / "embedding_model.onnx",
        "sha256": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    },
}


def find_asset(key_or_name: str) -> dict[str, Any] | None:
    """Find an asset in ASSETS by key, name, or filename."""
    if key_or_name in ASSETS:
        return ASSETS[key_or_name]
    for asset in ASSETS.values():
        if asset["name"] == key_or_name or Path(asset["path"]).name == key_or_name:
            return asset
    return None


def compute_sha256(path: Path | str) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify(
    url: str,
    target_path: Path | str,
    expected_sha256: str | None = None,
    timeout: float = 30.0,
    download_fn: Callable[[str, Path], None] | None = None,
) -> Path:
    """Download a file with timeout and verify its SHA256 checksum.

    If verification fails or download errors out, deletes the target file and raises an exception.
    """
    dest = Path(target_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if download_fn is not None:
            download_fn(url, dest)
        else:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (bmo-fetcher)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)

        if expected_sha256 is not None:
            actual = compute_sha256(dest)
            if actual.lower() != expected_sha256.lower():
                dest.unlink(missing_ok=True)
                raise ValueError(
                    f"Checksum mismatch for {dest.name}: expected {expected_sha256}, got {actual}"
                )
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return dest


def verify_asset_file(target_path: Path | str, expected_sha256: str | None = None) -> bool:
    """Check if a file exists and matches the expected SHA256 checksum."""
    p = Path(target_path)
    if not p.is_file():
        return False
    if expected_sha256 is not None:
        try:
            return compute_sha256(p).lower() == expected_sha256.lower()
        except OSError:
            return False
    return True


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
    file_download_fn: Callable[[str, Path], None] | None = None,
) -> dict[str, object]:
    """Verify or download stock openWakeWord models."""
    names = model_names or ["hey_jarvis_v0.1"]
    models_dir = get_openwakeword_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in names:
        target_name = f"{name}.onnx" if not name.endswith(".onnx") else name
        asset = find_asset(target_name) or find_asset(name)
        sha = asset["sha256"] if asset else None
        if not verify_asset_file(models_dir / target_name, sha):
            missing.append(name)

    if missing:
        if download_fn is not None:
            download_fn(missing)
        else:
            try:
                # Ensure preprocessor models are present if using stock models
                for prep in ("melspectrogram", "embedding_model"):
                    prep_asset = ASSETS.get(prep)
                    if prep_asset and not verify_asset_file(models_dir / f"{prep}.onnx", prep_asset["sha256"]):
                        download_and_verify(
                            url=prep_asset["url"],
                            target_path=models_dir / f"{prep}.onnx",
                            expected_sha256=prep_asset["sha256"],
                            timeout=30.0,
                            download_fn=file_download_fn,
                        )

                for name in missing:
                    target_name = f"{name}.onnx" if not name.endswith(".onnx") else name
                    asset = find_asset(target_name) or find_asset(name)
                    if asset is not None:
                        download_and_verify(
                            url=asset["url"],
                            target_path=models_dir / target_name,
                            expected_sha256=asset["sha256"],
                            timeout=30.0,
                            download_fn=file_download_fn,
                        )
                    else:
                        import openwakeword.utils

                        openwakeword.utils.download_models([name])
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
    download_fn: Callable[[str, Path], None] | None = None,
) -> dict[str, object]:
    """Verify or download Piper ONNX voice model and config."""
    out_dir = target_dir or (DEFAULT_MODELS_DIR / "piper")
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_file = out_dir / f"{voice_name}.onnx"
    json_file = out_dir / f"{voice_name}.onnx.json"

    onnx_asset = find_asset(f"{voice_name}.onnx") or (
        ASSETS.get("piper_voice") if voice_name == "en_US-lessac-medium" else None
    )
    json_asset = find_asset(f"{voice_name}.onnx.json") or (
        ASSETS.get("piper_config") if voice_name == "en_US-lessac-medium" else None
    )

    onnx_sha = onnx_asset["sha256"] if onnx_asset else None
    json_sha = json_asset["sha256"] if json_asset else None

    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"

    if fetch_remote:
        for file_path, remote_name, expected_sha, asset_entry in [
            (onnx_file, f"{voice_name}.onnx", onnx_sha, onnx_asset),
            (json_file, f"{voice_name}.onnx.json", json_sha, json_asset),
        ]:
            if not verify_asset_file(file_path, expected_sha):
                url = asset_entry["url"] if asset_entry else f"{base_url}/{remote_name}"
                try:
                    download_and_verify(
                        url=url,
                        target_path=file_path,
                        expected_sha256=expected_sha,
                        timeout=30.0,
                        download_fn=download_fn,
                    )
                except Exception as exc:
                    return {
                        "engine": "piper",
                        "ok": False,
                        "voice": voice_name,
                        "dir": str(out_dir),
                        "error": f"Failed to download {remote_name}: {exc}",
                    }

    ok = verify_asset_file(onnx_file, onnx_sha) and verify_asset_file(json_file, json_sha)
    return {
        "engine": "piper",
        "ok": ok,
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
    """Run verification or download for all engines."""
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
