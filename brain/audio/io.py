"""Audio I/O abstractions and concrete sources/sinks."""

from __future__ import annotations

import threading
import time
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import sounddevice as sd

from brain.settings import AudioSettings

_audio_defaults = AudioSettings()
SAMPLE_RATE: int = _audio_defaults.sample_rate
FRAME_MS: int = _audio_defaults.frame_ms
FRAME_SAMPLES: int = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320 samples
FRAME_BYTES: int = FRAME_SAMPLES * 2  # 640 bytes (int16 mono)


def to_int16(pcm: np.ndarray | bytes | bytearray | memoryview) -> np.ndarray:
    """Convert PCM audio (bytes, int16 array, or float [-1.0, 1.0]) to 1D int16 array."""
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        return np.frombuffer(pcm, dtype=np.int16).copy()
    arr = np.asarray(pcm)
    if np.issubdtype(arr.dtype, np.floating):
        clipped = np.clip(arr, -1.0, 1.0)
        return np.rint(clipped * 32767.0).astype(np.int16).reshape(-1)
    return arr.astype(np.int16, copy=False).reshape(-1)


@runtime_checkable
class AudioSource(Protocol):
    """Protocol for reading fixed-length audio frames."""

    def read_frame(self, timeout_s: float | None = None) -> np.ndarray | None:
        """Read a 1D int16 frame of FRAME_SAMPLES, or None on EOF/timeout."""
        ...

    def close(self) -> None:
        """Release underlying audio resources."""
        ...


@runtime_checkable
class AudioSink(Protocol):
    """Protocol for writing arbitrary-length audio samples."""

    def write_samples(self, pcm: np.ndarray | bytes) -> None:
        """Write 16 kHz mono int16 PCM samples or bytes."""
        ...

    def stop(self) -> None:
        """Immediately stop playback and discard pending buffers."""
        ...

    def close(self) -> None:
        """Release underlying audio resources."""
        ...


class MemorySource:
    """Yields 20 ms (320-sample) int16 frames from an in-memory buffer."""

    def __init__(self, pcm: np.ndarray | bytes) -> None:
        self._samples = to_int16(pcm).copy()
        self._cursor: int = 0
        self._closed: bool = False

    def read_frame(self, timeout_s: float | None = None) -> np.ndarray | None:
        del timeout_s
        if self._closed or self._cursor >= len(self._samples):
            return None
        chunk = self._samples[self._cursor : self._cursor + FRAME_SAMPLES]
        self._cursor += len(chunk)
        if len(chunk) < FRAME_SAMPLES:
            padded = np.zeros(FRAME_SAMPLES, dtype=np.int16)
            padded[: len(chunk)] = chunk
            return padded
        return chunk.copy()

    def close(self) -> None:
        self._closed = True


class MemorySink:
    """Accumulates written int16 samples into an in-memory buffer."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self.stopped: bool = False
        self._closed: bool = False

    def write_samples(self, pcm: np.ndarray | bytes) -> None:
        if self._closed:
            return
        arr = to_int16(pcm)
        self._chunks.append(arr.copy())
        self.stopped = False

    def get_samples(self) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.int16)
        return np.concatenate(self._chunks)

    def stop(self) -> None:
        self.stopped = True
        self._chunks.clear()

    def close(self) -> None:
        self._closed = True


class _LoopbackShared:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buffer = bytearray()
        self._cond = threading.Condition()
        self._closed = False
        self._sink_closed = False

    def write(self, pcm: np.ndarray | bytes) -> None:
        raw = to_int16(pcm).tobytes()
        with self._cond:
            if self._sink_closed or self._closed:
                return
            self._buffer.extend(raw)
            self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._buffer.clear()
            self._cond.notify_all()

    def close_sink(self) -> None:
        with self._cond:
            self._sink_closed = True
            self._cond.notify_all()

    def close_source(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def read_frame(self, timeout_s: float | None = None) -> np.ndarray | None:
        with self._cond:
            if self._closed:
                return None

            if timeout_s is None:
                while len(self._buffer) < FRAME_BYTES and not (self._closed or self._sink_closed):
                    self._cond.wait()
            elif timeout_s > 0:
                deadline = self._clock() + timeout_s
                while len(self._buffer) < FRAME_BYTES and not (self._closed or self._sink_closed):
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        break
                    self._cond.wait(remaining)

            if len(self._buffer) >= FRAME_BYTES:
                frame_bytes = bytes(self._buffer[:FRAME_BYTES])
                del self._buffer[:FRAME_BYTES]
                return np.frombuffer(frame_bytes, dtype=np.int16).copy()

            if self._sink_closed and len(self._buffer) > 0:
                padded = bytearray(FRAME_BYTES)
                padded[: len(self._buffer)] = self._buffer
                self._buffer.clear()
                return np.frombuffer(bytes(padded), dtype=np.int16).copy()

            return None


class _LoopbackSink:
    def __init__(self, shared: _LoopbackShared) -> None:
        self._shared = shared

    def write_samples(self, pcm: np.ndarray | bytes) -> None:
        self._shared.write(pcm)

    def stop(self) -> None:
        self._shared.stop()

    def close(self) -> None:
        self._shared.close_sink()


class _LoopbackSource:
    def __init__(self, shared: _LoopbackShared) -> None:
        self._shared = shared

    def read_frame(self, timeout_s: float | None = None) -> np.ndarray | None:
        return self._shared.read_frame(timeout_s)

    def close(self) -> None:
        self._shared.close_source()


class LoopbackPair:
    """Thread-safe queue connecting an AudioSink and AudioSource in memory."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        shared = _LoopbackShared(clock=clock)
        self.sink: AudioSink = _LoopbackSink(shared)
        self.source: AudioSource = _LoopbackSource(shared)


class SoundDeviceSource:
    """AudioSource implementation backed by sounddevice."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._stream: sd.InputStream | None = None

    def read_frame(self, timeout_s: float | None = None) -> np.ndarray | None:
        del timeout_s
        if self._stream is None:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
            )
            self._stream.start()
        data, _ = self._stream.read(FRAME_SAMPLES)
        return np.asarray(data, dtype=np.int16).reshape(-1)

    def record_seconds(self, duration_s: float) -> np.ndarray:
        frames = int(duration_s * self.sample_rate)
        data = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocking=True,
        )
        return np.asarray(data, dtype=np.int16).reshape(-1)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class SoundDeviceSink:
    """AudioSink implementation backed by sounddevice."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: int | str | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._stream: sd.OutputStream | None = None
        self._written_samples = 0
        self._played_samples = 0

    def write_samples(self, pcm: np.ndarray | bytes) -> None:
        arr = to_int16(pcm)
        if len(arr) == 0:
            return
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=self.device,
            )
            self._stream.start()
        self._stream.write(arr)
        self._written_samples += len(arr)

    def drain(self) -> None:
        """Wait for all written audio to finish playing."""
        if self._stream is not None:
            try:
                # Wait for the stream to finish playing all queued audio
                while self._stream.active:
                    sd.sleep(10)
            except Exception:
                # Stream might be invalid, ignore
                pass

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
