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
    to_int16,
)
from brain.settings import AudioSettings, load_settings


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
    defaults = AudioSettings()
    assert SAMPLE_RATE == defaults.sample_rate
    assert FRAME_MS == defaults.frame_ms
    assert FRAME_SAMPLES == int(defaults.sample_rate * defaults.frame_ms / 1000)
    assert FRAME_BYTES == FRAME_SAMPLES * 2
    assert SAMPLE_RATE == 16000
    assert FRAME_MS == 20
    assert FRAME_SAMPLES == 320
    assert FRAME_BYTES == 640


def test_to_int16_conversions() -> None:
    # bytes input
    raw = np.array([100, -200, 32767, -32768], dtype=np.int16).tobytes()
    res = to_int16(raw)
    assert res.dtype == np.int16
    assert np.array_equal(res, [100, -200, 32767, -32768])

    # int16 ndarray input
    int_arr = np.array([1, 2, 3], dtype=np.int16)
    assert np.array_equal(to_int16(int_arr), int_arr)

    # float input [-1.0, 1.0] scaling
    floats = np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    converted = to_int16(floats)
    assert converted.dtype == np.int16
    assert converted[0] == 0
    assert converted[1] == 32767
    assert converted[2] == -32767
    assert converted[3] == 16384
    assert converted[4] == -16384

    # float clipping outside [-1.0, 1.0]
    out_of_bounds = np.array([2.5, -3.0], dtype=np.float64)
    clipped = to_int16(out_of_bounds)
    assert np.array_equal(clipped, [32767, -32767])


def test_memory_sink_and_source_with_floats() -> None:
    # Float inputs should be scaled into int16, not truncated to zero
    float_audio = np.array([0.5, -0.5, 0.25], dtype=np.float32)
    sink = MemorySink()
    sink.write_samples(float_audio)
    samples = sink.get_samples()
    assert samples.dtype == np.int16
    assert not np.all(samples == 0)
    assert np.array_equal(samples, [16384, -16384, 8192])
    sink.close()

    source = MemorySource(float_audio)
    frame = source.read_frame()
    assert frame is not None
    assert frame.dtype == np.int16
    assert frame[0] == 16384
    assert frame[1] == -16384
    assert frame[2] == 8192
    source.close()


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


def test_speech_envelope_callback() -> None:
    calls = []
    sink = SoundDeviceSink(
        speech_cb=lambda env, t0, rate: calls.append((env, t0, rate))
    )
    audio = np.full(8000, 3000, dtype=np.int16)
    sink._emit_speech_envelope(audio)
    assert len(calls) == 1
    envelope, t0, rate = calls[0]
    assert rate == 30.0
    assert t0 > 0
    assert len(envelope) > 1
    assert max(envelope) == 1.0
    assert min(envelope) == 1.0


def test_silent_speech_envelope_is_zero() -> None:
    calls = []
    sink = SoundDeviceSink(speech_cb=lambda env, _t0, _rate: calls.append(env))
    sink._emit_speech_envelope(np.zeros(8000, dtype=np.int16))
    assert calls and set(calls[0]) == {0.0}


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
