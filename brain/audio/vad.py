"""Voice Activity Detector (VAD) and utterance segmentation."""

from collections.abc import Callable
from typing import Any

import numpy as np

from brain.audio.io import AudioSource
from brain.settings import AudioSettings


class VoiceActivityDetector:
    """Detects speech activity and segments utterances from audio streams."""

    def __init__(
        self,
        settings: AudioSettings,
        speech_prob_fn: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self._settings = settings
        self._speech_prob_fn = speech_prob_fn
        self.min_silence_frames = max(1, int(settings.vad_min_silence_ms // settings.frame_ms))
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from silero_vad import load_silero_vad

                self._model = load_silero_vad()
            except Exception:
                self._model = False
        return self._model

    def speech_probability(self, frame: np.ndarray) -> float:
        """Calculate speech probability in [0.0, 1.0] for an audio frame."""
        if self._speech_prob_fn is not None:
            return float(self._speech_prob_fn(frame))

        model = self._get_model()
        if model:
            try:
                import torch

                tensor = torch.from_numpy(frame).float() / 32768.0
                return float(model(tensor, self._settings.sample_rate).item())
            except Exception:
                pass

        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        return min(1.0, rms / 1000.0)

    def is_speech(self, frame: np.ndarray) -> bool:
        """Check if frame contains speech above configured threshold."""
        return self.speech_probability(frame) >= self._settings.vad_threshold

    def collect_utterance(
        self,
        source: AudioSource,
        max_duration_s: float = 10.0,
    ) -> np.ndarray | None:
        """Collect a speech utterance bounded by silence run or maximum duration."""
        max_frames = int(max_duration_s * 1000 // self._settings.frame_ms)
        collected: list[np.ndarray] = []
        silence_run = 0
        in_speech = False

        for _ in range(max_frames):
            frame = source.read_frame()
            if frame is None:
                break

            if not in_speech:
                if self.is_speech(frame):
                    in_speech = True
                    collected.append(frame)
                    silence_run = 0
            else:
                collected.append(frame)
                if self.is_speech(frame):
                    silence_run = 0
                else:
                    silence_run += 1
                    if silence_run >= self.min_silence_frames:
                        break

        if not collected or not in_speech:
            return None

        return np.concatenate(collected)
