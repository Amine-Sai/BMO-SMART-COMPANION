"""Tests for T5.2: Pi UDP and HTTP audio streaming transports."""

from __future__ import annotations

import inspect
import socket
import struct
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


def test_udp_audio_receiver_drops_late_packets() -> None:
    receiver = UDPAudioReceiver(host="127.0.0.1", port=0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        f0 = np.full(FRAME_SAMPLES, 100, dtype=np.int16)
        f1 = np.full(FRAME_SAMPLES, 200, dtype=np.int16)
        f2 = np.full(FRAME_SAMPLES, 300, dtype=np.int16)

        # Send seq 0
        sock.sendto(struct.pack("!I", 0) + f0.tobytes(), ("127.0.0.1", receiver.port))
        # Send seq 2 (causes gap of 1 for seq 1)
        sock.sendto(struct.pack("!I", 2) + f2.tobytes(), ("127.0.0.1", receiver.port))
        # Send seq 1 (late packet, older than seq 2)
        sock.sendto(struct.pack("!I", 1) + f1.tobytes(), ("127.0.0.1", receiver.port))

        time.sleep(0.05)

        # Receiver should yield: f0, zero-fill for missing 1, f2.
        # Late packet seq 1 must be dropped, not queued.
        r0 = receiver.read_frame(timeout_s=0.5)
        r1_fill = receiver.read_frame(timeout_s=0.5)
        r2 = receiver.read_frame(timeout_s=0.5)
        r_none = receiver.read_frame(timeout_s=0.1)

        assert r0 is not None and np.array_equal(r0, f0)
        assert r1_fill is not None and np.all(r1_fill == 0)
        assert r2 is not None and np.array_equal(r2, f2)
        assert r_none is None
        assert receiver.late_packets == 1
        assert receiver.dropped_packets == 1
    finally:
        sock.close()
        receiver.close()


def test_udp_audio_receiver_gap_filling() -> None:
    receiver = UDPAudioReceiver(host="127.0.0.1", port=0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        f0 = np.full(FRAME_SAMPLES, 500, dtype=np.int16)
        f3 = np.full(FRAME_SAMPLES, 800, dtype=np.int16)

        # Send seq 0, skip seq 1 and seq 2, send seq 3
        sock.sendto(struct.pack("!I", 0) + f0.tobytes(), ("127.0.0.1", receiver.port))
        sock.sendto(struct.pack("!I", 3) + f3.tobytes(), ("127.0.0.1", receiver.port))

        time.sleep(0.05)

        r0 = receiver.read_frame(timeout_s=0.5)
        r1 = receiver.read_frame(timeout_s=0.5)
        r2 = receiver.read_frame(timeout_s=0.5)
        r3 = receiver.read_frame(timeout_s=0.5)
        r_end = receiver.read_frame(timeout_s=0.1)

        assert r0 is not None and np.array_equal(r0, f0)
        assert r1 is not None and np.all(r1 == 0)
        assert r2 is not None and np.all(r2 == 0)
        assert r3 is not None and np.array_equal(r3, f3)
        assert r_end is None
        assert receiver.dropped_packets == 2
    finally:
        sock.close()
        receiver.close()


def test_udp_sender_float_audio() -> None:
    receiver = UDPAudioReceiver(host="127.0.0.1", port=0)
    sender = UDPAudioSender(host="127.0.0.1", port=receiver.port)
    try:
        float_samples = np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)
        sender.send_pcm(float_samples)

        frame = receiver.read_frame(timeout_s=0.5)
        assert frame is not None
        assert frame.dtype == np.int16
        assert np.all(frame == 16384)
    finally:
        sender.close()
        receiver.close()
