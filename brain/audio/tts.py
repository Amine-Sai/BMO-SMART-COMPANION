from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import threading
import time

import numpy as np

from brain.audio.io import AudioSink, FRAME_SAMPLES, SAMPLE_RATE
from brain.settings import AudioSettings
from brain.stream import split_sentences


class TextToSpeech:
    """Streaming Text-to-Speech synthesizer and audio player."""

    def __init__(
        self,
        settings: AudioSettings,
        synth_fn: Callable[[str], np.ndarray] | None = None,
        model_path: str | Path | None = None,
        chunk_delay_s: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._settings = settings
        self._synth_fn = synth_fn
        self._model_path = Path(model_path) if model_path is not None else None
        self._chunk_delay_s = chunk_delay_s
        self._clock = clock
        self._cancel_event = threading.Event()
        self._active_sink: AudioSink | None = None

    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.empty(0, dtype=np.int16)

        if self._synth_fn is not None:
            return np.asarray(self._synth_fn(text), dtype=np.int16).reshape(-1)

        resolved_path = (
            self._model_path if self._model_path is not None else Path(f"models/piper/{self._settings.tts_voice}.onnx")
        )
        if resolved_path.exists():
            try:
                import piper.voice  # type: ignore[import-untyped]

                voice = piper.voice.PiperVoice.load(str(resolved_path), use_cuda=False)
                chunks = list(voice.synthesize(text))
                audio = np.concatenate([c.audio_int16_array for c in chunks])

                # Resample from model sample rate (usually 22050) to system SAMPLE_RATE (16000)
                model_sr = voice.config.sample_rate
                if model_sr != SAMPLE_RATE:
                    # Simple linear interpolation resampling
                    duration = len(audio) / model_sr
                    target_len = int(duration * SAMPLE_RATE)
                    if target_len > 0:
                        indices = np.linspace(0, len(audio) - 1, target_len)
                        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)
                return audio
            except Exception as exc:
                import logging
                logging.getLogger(__name__).error("Piper synthesis failed: %s", exc)

        # Fallback: sine wave
        word_count = max(1, len(text.split()))
        num_samples = word_count * FRAME_SAMPLES * 4
        t = np.linspace(0, num_samples / SAMPLE_RATE, num_samples, endpoint=False)
        return (np.sin(2 * np.pi * 440.0 * t) * 8000.0).astype(np.int16)

    def speak_sentence(
        self,
        text: str,
        sink: AudioSink,
        cancel_event: threading.Event | None = None,
    ) -> float:
        self._cancel_event.clear()
        self._active_sink = sink
        t0 = self._clock()
        pcm = self.synthesize(text)
        if len(pcm) == 0:
            return 0.0

        first_chunk_ms = 0.0
        has_written = False

        for start in range(0, len(pcm), FRAME_SAMPLES):
            if self._cancel_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                sink.stop()
                return first_chunk_ms

            if self._chunk_delay_s > 0.0:
                time.sleep(self._chunk_delay_s)
                if self._cancel_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                    sink.stop()
                    return first_chunk_ms

            chunk = pcm[start : start + FRAME_SAMPLES]
            sink.write_samples(chunk)
            if not has_written:
                first_chunk_ms = (self._clock() - t0) * 1000.0
                has_written = True

        return first_chunk_ms

    def speak_stream(
        self,
        token_stream: Iterable[str],
        sink: AudioSink,
        cancel_event: threading.Event | None = None,
    ) -> list[float]:
        latencies: list[float] = []
        for sentence in split_sentences(token_stream):
            if self._cancel_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                sink.stop()
                break
            first_chunk_ms = self.speak_sentence(sentence, sink, cancel_event)
            latencies.append(first_chunk_ms)
            if self._cancel_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                break
        return latencies

    def cancel(self) -> None:
        self._cancel_event.set()
        if self._active_sink is not None:
            self._active_sink.stop()
