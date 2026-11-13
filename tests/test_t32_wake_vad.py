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
    chunk_ms = settings.vad_chunk_samples / settings.sample_rate * 1000
    min_silence_chunks = max(1, int(settings.vad_min_silence_ms // chunk_ms))
    assert min_silence_chunks == 12  # 400ms / 32ms = 12

    def speech_prob_fn(frame: np.ndarray) -> float:
        return 1.0 if np.max(np.abs(frame)) > 1000 else 0.0

    vad = VoiceActivityDetector(settings=settings, speech_prob_fn=speech_prob_fn)
    assert vad.min_silence_chunks == min_silence_chunks

    silent_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    speech_frame = np.full(FRAME_SAMPLES, 5000, dtype=np.int16)

    # 10 silent frames (3200 samples = 6 chunks + 128 remainder) + 25 speech frames (8000 samples) + 25 silent frames
    frames = [silent_frame] * 10 + [speech_frame] * 25 + [silent_frame] * 25
    audio_data = np.concatenate(frames)
    source = MemorySource(audio_data)

    utterance = vad.collect_utterance(source)
    assert utterance is not None

    # Output includes pre-roll + speech chunks + trailing silence chunks
    # 10 silent frames = 3200 samples = 6 chunks + 128 rem -> pre-roll has 6 chunks
    # 25 speech frames = 8000 samples + 128 rem = 8128 = 15 full chunks + 448 rem
    # But boundary creates 1 mixed chunk at speech start + 1 mixed chunk at speech end
    # Total speech chunks: 16 (including mixed boundary chunks)
    # Trailing silence: stops after 12 silence chunks
    # Total: pre-roll (6) + speech (16) + 12 silence = 34 chunks * 512 = 17408 samples
    expected_chunks = 6 + 16 + min_silence_chunks  # pre-roll + speech + trailing silence
    expected_samples = expected_chunks * settings.vad_chunk_samples
    assert len(utterance) == expected_samples

    # Verify early stop: some silent frames remain in source
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


def test_voice_activity_detector_rechunks_320_to_512() -> None:
    """Verify VAD internally re-chunks 320-sample frames to 512 for Silero."""
    settings = load_settings().audio
    assert settings.vad_chunk_samples == 512

    received_frames: list[np.ndarray] = []

    def capture_speech_prob_fn(frame: np.ndarray) -> float:
        received_frames.append(frame.copy())
        return 1.0 if np.max(np.abs(frame)) > 1000 else 0.0

    vad = VoiceActivityDetector(settings=settings, speech_prob_fn=capture_speech_prob_fn)
    speech_frame = np.full(FRAME_SAMPLES, 5000, dtype=np.int16)

    # Feed 5 speech frames (5 * 320 = 1600 samples)
    # Should produce 3 full 512-sample chunks + remainder
    frames = [speech_frame] * 5
    audio_data = np.concatenate(frames)
    source = MemorySource(audio_data)

    vad.collect_utterance(source)

    # All frames passed to speech_prob_fn should be 512 samples
    assert all(len(f) == settings.vad_chunk_samples for f in received_frames)
    # 5 * 320 = 1600 samples -> 3 full 512 chunks (1536) + 64 remainder
    # But we only process full chunks, so 3 frames of 512
    assert len(received_frames) == 3


def test_voice_activity_detector_preroll_prepended() -> None:
    """Verify ~250ms pre-roll is prepended to collected utterances."""
    settings = load_settings().audio
    assert settings.vad_preroll_ms == 250

    # Pre-roll frames at 512 samples: 250ms / (512/16000*1000) ≈ 7.8 frames -> 8 frames
    preroll_frames_count = max(1, int(settings.vad_preroll_ms * settings.sample_rate / 1000 / settings.vad_chunk_samples))
    # Need enough silent frames to fill pre-roll: 8 * 512 / 320 = 12.8 -> 13 frames
    min_silent_frames = (preroll_frames_count * settings.vad_chunk_samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES

    def speech_prob_fn(frame: np.ndarray) -> float:
        return 1.0 if np.max(np.abs(frame)) > 1000 else 0.0

    vad = VoiceActivityDetector(settings=settings, speech_prob_fn=speech_prob_fn)

    silent_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    speech_frame = np.full(FRAME_SAMPLES, 5000, dtype=np.int16)

    # Enough silent frames to fill pre-roll + speech frames + trailing silence
    frames = [silent_frame] * (min_silent_frames + 2) + [speech_frame] * 10 + [silent_frame] * 20
    audio_data = np.concatenate(frames)
    source = MemorySource(audio_data)

    utterance = vad.collect_utterance(source)
    assert utterance is not None

    # Utterance should include pre-roll + speech + trailing silence
    preroll_samples = preroll_frames_count * settings.vad_chunk_samples
    assert len(utterance) >= preroll_samples

    # First preroll_samples should be silence (from the silent frames before speech)
    preroll_portion = utterance[:preroll_samples]
    assert np.all(preroll_portion == 0)
