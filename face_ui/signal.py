"""File-based speech/state signal shared by the brain and the face process.

MQTT is convenient but the face must keep moving even if a status message is
lost. The brain writes this small JSON file on every state change; the face
polls it and can also read an audio envelope for lip sync. Both processes use
``time.monotonic()``, which shares one origin on Linux.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "face_state.json"


def default_path() -> Path:
    return Path(os.environ.get("BMO_FACE_STATE", DEFAULT_PATH))


@dataclass
class FaceSignal:
    """Atomic writer for the face state file (safe to call from any thread)."""

    path: Path = field(default_factory=default_path)
    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _state: dict = field(
        default_factory=lambda: {
            "v": 1,
            "seq": 0,
            "mode": "boot",
            "emotion": "neutral",
            "online": True,
            "text": "",
            "ts": 0.0,
            "speech": None,
        },
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self) -> None:
        self._seq += 1
        self._state["seq"] = self._seq
        self._state["ts"] = time.monotonic()
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(
                json.dumps(self._state, separators=(",", ":")), encoding="utf-8"
            )
            os.replace(tmp, self.path)
        except OSError:
            # A dropped frame is better than taking the brain down.
            return

    def set_status(self, mode: str, online: bool = True) -> None:
        with self._lock:
            self._state["mode"] = str(mode)
            self._state["online"] = bool(online)
            if str(mode) != "speaking":
                self._state["speech"] = None
            self._write()

    def set_say(self, text: str, emotion: str) -> None:
        with self._lock:
            self._state["mode"] = "speaking"
            self._state["online"] = True
            self._state["emotion"] = str(emotion)
            self._state["text"] = str(text)
            self._write()

    def set_speech(self, envelope: list[float], t0: float, rate: float) -> None:
        with self._lock:
            self._state["speech"] = {
                "t0": float(t0),
                "rate": float(rate),
                "env": [round(float(value), 3) for value in envelope],
            }
            self._write()


class FaceSignalReader:
    """Read the state file only when it changes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_path()
        self._mtime_ns: int | None = None
        self._data: dict | None = None

    def read(self) -> dict | None:
        try:
            stat = self.path.stat()
        except OSError:
            return self._data
        if stat.st_mtime_ns == self._mtime_ns:
            return self._data
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self._data
        self._mtime_ns = stat.st_mtime_ns
        self._data = data if isinstance(data, dict) else None
        return self._data


def speech_mouth(signal: dict | None, now: float) -> float | None:
    """Return the mouth opening for the current playback position, if known."""
    speech = signal.get("speech") if signal else None
    if not isinstance(speech, dict):
        return None
    env = speech.get("env")
    if not isinstance(env, list) or not env:
        return None
    try:
        t0 = float(speech["t0"])
        rate = float(speech.get("rate", 30.0))
    except (KeyError, TypeError, ValueError):
        return None
    index = int((now - t0) * rate)
    if index < 0:
        return 0.0
    if index < len(env):
        return max(0.0, min(1.0, float(env[index])))
    # The envelope is over; let the caller fall back to its idle mouth motion.
    return None
