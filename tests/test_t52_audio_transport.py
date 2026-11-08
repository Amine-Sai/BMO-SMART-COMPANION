"""Tests for T5.2: Pi UDP and HTTP audio streaming transports."""

from __future__ import annotations

import inspect
import time

import numpy as np
from brain.audio.io import FRAME_SAMPLES, SAMPLE_RATE, AudioSink, AudioSource
import edge.audio_transport
from edge.audio_transport import (
    HTTPAudioServer,
    UDPAudioReceiver,
    UDPAudioSender,
    fetch_tts_wav,
    pcm_to_wav_bytes,
    wav_bytes_to_pcm,
)


def test_audio_transport_interfaces() -> None:
    receiver = UDPAudioReceiver(host="127.0.0.1", port=0)
    server = HTTPAudioServer(host="127.0.0.1", port=0)

    try:
        assert isinstance(receiver, AudioSource)
        assert isinstance(server, AudioSink)
        assert receiver.port > 0
        assert server.port > 0
        assert server.host == "127.0.0.1"
        assert server.base_url == f"http://127.0.0.1:{server.port}"
    finally:
        receiver.close()
        server.close()


def test_udp_audio_streaming_roundtrip() -> None:
    duration_s = 5.0
    n_samples = int(duration_s * SAMPLE_RATE)  # 80000 samples
    t = np.arange(n_samples) / SAMPLE_RATE
    audio = (np.sin(2 * np.pi * 440.0 * t) * 16000).astype(np.int16)

    total_frames = n_samples // FRAME_SAMPLES  # 250 frames
    assert total_frames == 250

    receiver = UDPAudioReceiver(host="127.0.0.1", port=0)
    sender = UDPAudioSender(host="127.0.0.1", port=receiver.port)

    assert sender.host == "127.0.0.1"
    assert sender.port == receiver.port

    try:
        # Test send_frame on first frame
        first_frame = audio[:FRAME_SAMPLES]
        bytes_sent = sender.send_frame(first_frame)
        assert bytes_sent == 644

        # Stream remaining frames via send_pcm
        remaining = audio[FRAME_SAMPLES:]
        sender.send_pcm(remaining)

        # Allow loopback delivery
        time.sleep(0.05)

        received_frames: list[np.ndarray] = []
        for _ in range(total_frames):
            frame = receiver.read_frame(timeout_s=0.5)
            if frame is None:
                break
            received_frames.append(frame)

        # Loss < 1% requires >= 248 frames out of 250
        assert len(received_frames) >= 248
        assert receiver.received_packets >= 248
        assert receiver.dropped_packets <= 2

        # Verify received samples match sent samples exactly
        reconstructed = np.concatenate(received_frames)
        expected = audio[: len(reconstructed)]
        assert np.array_equal(reconstructed, expected)
    finally:
        sender.close()
        receiver.close()


def test_http_wav_transport_roundtrip() -> None:
    duration_s = 5.0
    n_samples = int(duration_s * SAMPLE_RATE)
    t = np.arange(n_samples) / SAMPLE_RATE
    audio = (np.sin(2 * np.pi * 880.0 * t) * 12000).astype(np.int16)

    # Test direct WAV helper roundtrip
    wav_data = pcm_to_wav_bytes(audio)
    decoded_pcm = wav_bytes_to_pcm(wav_data)
    assert np.array_equal(audio, decoded_pcm)

    server = HTTPAudioServer(host="127.0.0.1", port=0)
    try:
        utt_id = "utt_roundtrip_test"
        url = server.store_utterance(utt_id, audio)
        assert url == f"{server.base_url}/tts/{utt_id}.wav"

        # Fetch utterance WAV over HTTP
        fetched = fetch_tts_wav(url)
        assert np.array_equal(audio, fetched)

        # Fetch latest WAV
        latest_url = f"{server.base_url}/tts/latest.wav"
        latest_fetched = fetch_tts_wav(latest_url)
        assert np.array_equal(audio, latest_fetched)

        # Test write_samples sink interface
        short_pcm = audio[:FRAME_SAMPLES]
        server.write_samples(short_pcm)
        updated_latest = fetch_tts_wav(latest_url)
        assert np.array_equal(short_pcm, updated_latest)

        server.stop()
    finally:
        server.close()


def test_zero_mqtt_usage() -> None:
    source = inspect.getsource(edge.audio_transport)
    assert "mqtt" not in source.lower()
    topic_prefix = "bmo" + "/"
    assert topic_prefix not in source
