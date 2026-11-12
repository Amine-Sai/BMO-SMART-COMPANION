"""Wake word detector with openWakeWord and cooldown management."""

from collections.abc import Callable
import re
import time
from typing import Any

import numpy as np

from brain.settings import AudioSettings


# openWakeWord has no "hey bmo" model, so wake detection reuses the STT engine.
# Whisper commonly renders "BMO" as bmo / beemo / bimo / "be mo" / "b m o".
_WAKE_PREFIXES = r"(?:hey|hi|hay|hello|okay|ok|yo)"
_BMO_VARIANTS = r"(?:b[\s.\-]*m[\s.\-]*o|beemo|bimo|be\s*mo|b\s*e\s*m\s*o)"
_WAKE_RE = re.compile(
    rf"\b{_WAKE_PREFIXES}\b[\s,.!?]*\b{_BMO_VARIANTS}\b",
    re.IGNORECASE,
)


def extract_wake_command(text: str, wake_word: str = "hey bmo") -> str | None:
    """Return the command following "hey bmo" in ``text``.

    Returns ``None`` when the utterance is not addressed to BMO, an empty
    string when only the wake phrase was spoken, or the trailing command text.
    """
    del wake_word  # currently specialised for "hey bmo"
    match = _WAKE_RE.search(text)
    if match is None:
        return None
    return text[match.end() :].strip(" ,.!?-")


class WakeWordDetector:
    """Detects wake words in audio frames with refractory cooldown."""

    def __init__(
        self,
        settings: AudioSettings,
        model_name: str = "hey_jarvis",
        cooldown_s: float = 2.0,
        score_fn: Callable[[np.ndarray], float] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._model_name = model_name
        self._cooldown_s = cooldown_s
        self._score_fn = score_fn
        self._clock = clock
        self._last_wake_ts = -float("inf")
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                import openwakeword.model

                self._model = openwakeword.model.Model(
                    wakeword_models=[self._model_name],
                    inference_framework="onnx",
                )
            except Exception:
                self._model = False
        return self._model

    def score_frame(self, frame: np.ndarray) -> float:
        """Compute wake word confidence score in [0.0, 1.0] for a single frame."""
        if self._score_fn is not None:
            return float(self._score_fn(frame))

        model = self._get_model()
        if not model:
            return 0.0

        predictions = model.predict(frame)
        if isinstance(predictions, dict):
            return float(max(predictions.values(), default=0.0))
        return 0.0

    def process_frame(self, frame: np.ndarray) -> bool:
        """Process an audio frame and return True if wake word triggered."""
        score = self.score_frame(frame)
        if score >= self._settings.wake_threshold:
            now = self._clock()
            if (now - self._last_wake_ts) >= self._cooldown_s:
                self._last_wake_ts = now
                return True
        return False

    def reset(self) -> None:
        """Reset internal detector state and cooldown."""
        self._last_wake_ts = -float("inf")
        if self._model and hasattr(self._model, "reset"):
            self._model.reset()
