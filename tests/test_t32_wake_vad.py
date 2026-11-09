"""Tests for wake word detector and voice activity detector."""

import numpy as np

from brain.audio.io import FRAME_SAMPLES, MemorySource
from brain.audio.vad import VoiceActivityDetector
from brain.audio.wake import WakeWordDetector
from brain.settings import load_settings


def test_wake_word_detector_recall_and_cooldown() -> None:
    settings = load_settings().audio
    simulated_clock = 0.0

    def clock() -> float:
        return simulated_clock

    def score_fn(frame: np.ndarray) -> float:
        return 0.95 if np.max(np.abs(frame)) > 5000 else 0.05

    detector = WakeWordDetector(
        settings=settings,
        cooldown_s=2.0,
        score_fn=score_fn,
        clock=clock,
    )

    wake_frame = np.full(FRAME_SAMPLES, 10000, dtype=np.int16)

    # 20 positive wake events spaced past cooldown_s
    triggers = 0
    for _ in range(20):
        simulated_clock += 2.5
        if detector.process_frame(wake_frame):
            triggers += 1
    assert triggers >= 18  # >= 90% recall

    # Refractory cooldown blocks duplicate trigger within 2.0 s
    simulated_clock += 0.5
    assert not detector.process_frame(wake_frame)

    # Trigger allowed after cooldown expires
    simulated_clock += 1.6
    assert detector.process_frame(wake_frame)

    # 10 minutes of background noise (30,000 frames) produces <= 1 false wakes
    false_wakes = 0
    noise_batch = np.random.default_rng(42).normal(0, 100, size=(1000, FRAME_SAMPLES)).astype(np.int16)
    for i in range(30000):
        simulated_clock += 0.02
        frame = noise_batch[i % 1000]
        if detector.process_frame(frame):
            false_wakes += 1
    assert false_wakes <= 1


def test_wake_word_detector_reset() -> None:
    settings = load_settings().audio
    simulated_clock = 100.0

    detector = WakeWordDetector(
        settings=settings,
        cooldown_s=2.0,
        score_fn=lambda frame: 0.99 if len(frame) > 0 else 0.0,
        clock=lambda: simulated_clock,
    )
    frame = np.ones(FRAME_SAMPLES, dtype=np.int16)
    assert detector.process_frame(frame)
    # Immediately in cooldown
    assert not detector.process_frame(frame)
    # Reset cooldown
    detector.reset()
    assert detector.process_frame(frame)


def test_wake_word_detector_default_fallback() -> None:
    settings = load_settings().audio
    detector = WakeWordDetector(settings=settings)
    frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    # Evaluates gracefully without error
    score = detector.score_frame(frame)
    assert 0.0 <= score <= 1.0


def test_voice_activity_detector_utterance_collection() -> None:
    settings = load_settings().audio
    min_silence = max(1, int(settings.vad_min_silence_ms // settings.frame_ms))
    assert min_silence == 20

    def speech_prob_fn(frame: np.ndarray) -> float:
        return 1.0 if np.max(np.abs(frame)) > 1000 else 0.0

    vad = VoiceActivityDetector(settings=settings, speech_prob_fn=speech_prob_fn)
    assert vad.min_silence_frames == 20

    silent_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    speech_frame = np.full(FRAME_SAMPLES, 5000, dtype=np.int16)

    # 10 silent frames + 25 speech frames + 25 silent frames
    frames = [silent_frame] * 10 + [speech_frame] * 25 + [silent_frame] * 25
    audio_data = np.concatenate(frames)
    source = MemorySource(audio_data)

    utterance = vad.collect_utterance(source)
    assert utterance is not None
    # 45 frames: 25 speech + 20 trailing silence frames
    expected_samples = 45 * FRAME_SAMPLES
    assert len(utterance) == expected_samples

    # Verify early stop: 5 silent frames remain in source
    remaining_frame = source.read_frame()
    assert remaining_frame is not None
    assert np.array_equal(remaining_frame, silent_frame)


def test_voice_activity_detector_pure_silence() -> None:
    settings = load_settings().audio

    def speech_prob_fn(frame: np.ndarray) -> float:
        return 0.0 if len(frame) > 0 else 0.0

    vad = VoiceActivityDetector(settings=settings, speech_prob_fn=speech_prob_fn)
    silent_source = MemorySource(np.zeros(FRAME_SAMPLES * 50, dtype=np.int16))
    result = vad.collect_utterance(silent_source)
    assert result is None


def test_voice_activity_detector_default_speech_probability() -> None:
    settings = load_settings().audio
    vad = VoiceActivityDetector(settings=settings)
    frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    prob = vad.speech_probability(frame)
    assert 0.0 <= prob <= 1.0
    assert not vad.is_speech(frame)
