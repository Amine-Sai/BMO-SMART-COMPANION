from __future__ import annotations

import threading
import time

import numpy as np

from brain.audio.io import MemorySink
from brain.audio.tts import TextToSpeech
from brain.settings import load_settings


def test_speak_sentence_emits_audio() -> None:
    settings = load_settings().audio
    tts = TextToSpeech(settings)
    sink = MemorySink()
    latency = tts.speak_sentence("Hello friend!", sink)
    assert latency < 300.0
    if hasattr(sink, "samples"):
        raw_samples = sink.samples
        samples = np.concatenate(raw_samples) if isinstance(raw_samples, list) else np.asarray(raw_samples)
    elif hasattr(sink, "get_samples"):
        samples = np.asarray(sink.get_samples())
    else:
        samples = np.asarray(sink.buffer)
    assert len(samples) > 0
    assert samples.dtype == np.int16


def test_speak_stream_splits_sentences() -> None:
    settings = load_settings().audio
    tts = TextToSpeech(settings)
    sink = MemorySink()
    tokens = ["Hello ", "world! ", "How are ", "you?"]
    latencies = tts.speak_stream(tokens, sink)
    assert len(latencies) == 2
    assert all(lat >= 0.0 for lat in latencies)
    if hasattr(sink, "samples"):
        raw_samples = sink.samples
        samples = np.concatenate(raw_samples) if isinstance(raw_samples, list) else np.asarray(raw_samples)
    elif hasattr(sink, "get_samples"):
        samples = np.asarray(sink.get_samples())
    else:
        samples = np.asarray(sink.buffer)
    assert len(samples) > 0
    assert samples.dtype == np.int16


def test_cancel_mid_playback() -> None:
    settings = load_settings().audio
    tts = TextToSpeech(settings, chunk_delay_s=0.01)
    sink = MemorySink()
    text = "word " * 50
    t0 = time.perf_counter()

    def run_speak() -> None:
        tts.speak_sentence(text, sink)

    thread = threading.Thread(target=run_speak)
    thread.start()

    time.sleep(0.03)
    tts.cancel()
    thread.join(timeout=1.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert not thread.is_alive()
    assert elapsed_ms < 150.0
    assert sink.stopped is True


def test_synthesize_empty_and_synth_fn() -> None:
    settings = load_settings().audio
    tts = TextToSpeech(settings)
    empty = tts.synthesize("   ")
    assert len(empty) == 0
    assert empty.dtype == np.int16

    custom_tts = TextToSpeech(
        settings,
        synth_fn=lambda _text: np.array([1, 2, 3], dtype=np.int16),
    )
    result = custom_tts.synthesize("test")
    assert np.array_equal(result, np.array([1, 2, 3], dtype=np.int16))
