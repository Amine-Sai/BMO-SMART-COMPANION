"""Tests for Audio I/O abstraction and implementations."""

import numpy as np
import pytest
import sounddevice as sd

from brain.audio.io import (
    FRAME_BYTES,
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AudioSink,
    AudioSource,
    LoopbackPair,
    MemorySink,
    MemorySource,
    SoundDeviceSink,
    SoundDeviceSource,
)
from brain.settings import load_settings


def test_audio_settings_load() -> None:
    settings = load_settings("config")
    assert settings.audio is not None
    assert settings.audio.sample_rate == 16000
    assert settings.audio.frame_ms == 20
    assert settings.audio.wake_threshold == 0.5
    assert settings.audio.vad_threshold == 0.5
    assert settings.audio.vad_min_silence_ms == 400
    assert settings.audio.stt_model == "base.en"
    assert settings.audio.stt_device == "cpu"
    assert settings.audio.stt_compute_type == "int8"
    assert settings.audio.tts_voice == "en_US-lessac-medium"


def test_audio_constants() -> None:
    assert SAMPLE_RATE == 16000
    assert FRAME_MS == 20
    assert FRAME_SAMPLES == 320
    assert FRAME_BYTES == 640


def test_loopback_sine_wave_roundtrip() -> None:
    sample_rate = 16000
    duration_s = 5.0
    frequency = 440.0
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    sine_wave = (np.sin(2.0 * np.pi * frequency * t) * 32767).astype(np.int16)
    assert len(sine_wave) == 80000

    pair = LoopbackPair()
    assert isinstance(pair.sink, AudioSink)
    assert isinstance(pair.source, AudioSource)

    pair.sink.write_samples(sine_wave)
    pair.sink.close()

    frames: list[np.ndarray] = []
    for _ in range(250):
        frame = pair.source.read_frame()
        assert frame is not None
        assert frame.shape == (320,)
        assert frame.dtype == np.int16
        frames.append(frame)

    assert pair.source.read_frame() is None
    reconstructed = np.concatenate(frames)
    assert np.array_equal(sine_wave, reconstructed)
    pair.source.close()


def test_loopback_timeout_and_stop() -> None:
    pair = LoopbackPair()
    # Reading empty queue with timeout returns None
    assert pair.source.read_frame(timeout_s=0.01) is None

    # Writing samples then stopping clears buffer
    pair.sink.write_samples(np.ones(100, dtype=np.int16))
    pair.sink.stop()
    pair.sink.close()
    assert pair.source.read_frame(timeout_s=0.01) is None
    pair.source.close()


def test_memory_source_and_sink() -> None:
    pcm = np.arange(960, dtype=np.int16)
    source = MemorySource(pcm)
    assert isinstance(source, AudioSource)

    f1 = source.read_frame()
    assert f1 is not None and len(f1) == 320
    f2 = source.read_frame()
    assert f2 is not None and len(f2) == 320
    f3 = source.read_frame()
    assert f3 is not None and len(f3) == 320
    f4 = source.read_frame()
    assert f4 is None
    source.close()

    # Partial frame padding test
    partial_source = MemorySource(np.ones(10, dtype=np.int16))
    padded_frame = partial_source.read_frame()
    assert padded_frame is not None and len(padded_frame) == 320
    assert padded_frame[0] == 1
    assert padded_frame[10] == 0
    partial_source.close()

    # MemorySink roundtrip and stop
    sink = MemorySink()
    assert isinstance(sink, AudioSink)
    assert not sink.stopped
    sink.write_samples(f1)
    sink.write_samples(f2.tobytes())
    samples = sink.get_samples()
    assert len(samples) == 640
    assert np.array_equal(samples[:320], f1)
    assert np.array_equal(samples[320:], f2)

    sink.stop()
    assert sink.stopped
    assert len(sink.get_samples()) == 0
    sink.close()


def test_sounddevice_sink_protocol() -> None:
    sink = SoundDeviceSink()
    assert isinstance(sink, AudioSink)
    sink.stop()
    sink.close()


@pytest.mark.audio_hw
def test_sounddevice_record_1s() -> None:
    try:
        sd.query_devices(kind="input")
    except Exception as exc:
        pytest.skip(f"No audio input device available: {exc}")

    source = SoundDeviceSource()
    assert isinstance(source, AudioSource)
    try:
        data = source.record_seconds(0.2)
        assert data.dtype == np.int16
        assert len(data) == int(0.2 * SAMPLE_RATE)
    except sd.PortAudioError as exc:
        pytest.skip(f"PortAudio error on input device: {exc}")
    finally:
        source.close()
