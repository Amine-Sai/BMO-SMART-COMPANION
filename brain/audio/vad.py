"""Voice Activity Detector (VAD) and utterance segmentation."""

from collections import deque
from collections.abc import Callable
from math import ceil
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
        chunk_ms = settings.vad_chunk_samples / settings.sample_rate * 1000
        self.min_silence_chunks = max(1, int(settings.vad_min_silence_ms // chunk_ms))
        self._model: Any = None

        self._chunk_samples = settings.vad_chunk_samples
        self._preroll_maxlen = max(1, ceil(settings.vad_preroll_ms * settings.sample_rate / 1000 / settings.vad_chunk_samples))
        self._remainder: np.ndarray = np.empty(0, dtype=np.int16)
        self._preroll: deque[np.ndarray] = deque(maxlen=self._preroll_maxlen)

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from silero_vad import load_silero_vad

                self._model = load_silero_vad()
            except Exception:
                self._model = False
        return self._model

    def _rechunk_frame(self, frame: np.ndarray) -> list[np.ndarray]:
        """Re-chunk 320-sample input frames to vad_chunk_samples (512) frames."""
        combined = np.concatenate([self._remainder, frame]) if self._remainder.size else frame
        num_chunks = len(combined) // self._chunk_samples
        chunks = []
        for i in range(num_chunks):
            start = i * self._chunk_samples
            end = start + self._chunk_samples
            chunks.append(combined[start:end].copy())
        self._remainder = combined[num_chunks * self._chunk_samples :]
        return chunks

    def _compute_speech_prob(self, chunk: np.ndarray) -> float:
        """Compute speech probability for a 512-sample chunk."""
        if self._speech_prob_fn is not None:
            return float(self._speech_prob_fn(chunk))

        model = self._get_model()
        if model:
            try:
                import torch

                tensor = torch.from_numpy(chunk).float() / 32768.0
                return float(model(tensor, self._settings.sample_rate).item())
            except Exception:
                pass

        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        return min(1.0, rms / 1000.0)

    def speech_probability(self, frame: np.ndarray) -> float:
        """Calculate speech probability in [0.0, 1.0] for an audio frame."""
        chunks = self._rechunk_frame(frame)
        if not chunks:
            return 0.0
        return self._compute_speech_prob(chunks[-1])

    def is_speech(self, frame: np.ndarray) -> bool:
        """Check if frame contains speech above configured threshold."""
        return self.speech_probability(frame) >= self._settings.vad_threshold

    def collect_utterance(
        self,
        source: AudioSource,
        max_duration_s: float = 10.0,
    ) -> np.ndarray | None:
        """Collect a speech utterance bounded by silence run or maximum duration."""
        chunk_ms = self._chunk_samples / self._settings.sample_rate * 1000
        max_chunks = int(max_duration_s * 1000 // chunk_ms)
        collected: list[np.ndarray] = []
        silence_run = 0
        in_speech = False
        chunks_processed = 0

        while chunks_processed < max_chunks:
            frame = source.read_frame()
            if frame is None:
                break

            chunks = self._rechunk_frame(frame)
            for chunk in chunks:
                chunks_processed += 1
                is_speech = self._compute_speech_prob(chunk) >= self._settings.vad_threshold

                if not in_speech:
                    if is_speech:
                        in_speech = True
                        collected.extend(self._preroll)
                        collected.append(chunk)
                        silence_run = 0
                    else:
                        self._preroll.append(chunk.copy())
                else:
                    collected.append(chunk)
                    if is_speech:
                        silence_run = 0
                    else:
                        silence_run += 1
                        if silence_run >= self.min_silence_chunks:
                            break

                if in_speech and silence_run >= self.min_silence_chunks:
                    break

            if in_speech and silence_run >= self.min_silence_chunks:
                break

        if not collected or not in_speech:
            return None

        return np.concatenate(collected)