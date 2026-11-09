from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np

from brain.settings import AudioSettings


@dataclass(frozen=True)
class STTResult:
    text: str
    confidence: float
    latency_ms: float


class SpeechToText:
    def __init__(
        self,
        settings: AudioSettings,
        transcribe_fn: Callable[[np.ndarray], tuple[str, float] | None] | None = None,
        min_rms: float = 150.0,
        max_no_speech_prob: float = 0.6,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._settings = settings
        self._transcribe_fn = transcribe_fn
        self._min_rms = min_rms
        self._max_no_speech_prob = max_no_speech_prob
        self._clock = clock
        self._model = None

    def transcribe(self, pcm: np.ndarray | bytes) -> STTResult | None:
        if isinstance(pcm, bytes):
            arr = np.frombuffer(pcm, dtype=np.int16)
        else:
            arr = np.asarray(pcm, dtype=np.int16).reshape(-1)

        if len(arr) == 0:
            return None

        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
        if rms < self._min_rms:
            return None

        t0 = self._clock()

        if self._transcribe_fn is not None:
            res = self._transcribe_fn(arr)
            if res is None:
                return None
            text, conf = res
            if not text.strip() or conf < (1.0 - self._max_no_speech_prob):
                return None
            return STTResult(
                text=text.strip(),
                confidence=float(conf),
                latency_ms=(self._clock() - t0) * 1000.0,
            )

        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._settings.stt_model,
                device=self._settings.stt_device,
                compute_type=self._settings.stt_compute_type,
            )

        audio_f32 = arr.astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio_f32, beam_size=1, language="en")
        valid_texts: list[str] = []
        total_conf = 0.0
        count = 0
        for seg in segments:
            if seg.no_speech_prob <= self._max_no_speech_prob and seg.text.strip():
                valid_texts.append(seg.text.strip())
                total_conf += 1.0 - seg.no_speech_prob
                count += 1

        if not valid_texts:
            return None

        return STTResult(
            text=" ".join(valid_texts),
            confidence=float(total_conf / count) if count > 0 else 0.0,
            latency_ms=(self._clock() - t0) * 1000.0,
        )
