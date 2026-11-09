from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any

import numpy as np
import pytest

from brain.audio.io import AudioSink, AudioSource, MemorySink, MemorySource
from brain.audio.stt import STTResult, SpeechToText
from brain.audio.tts import TextToSpeech
from brain.audio.vad import VoiceActivityDetector
from brain.audio.wake import WakeWordDetector
from brain.llm import ChatChunk, LLMClient
from brain.logging_jsonl import JSONLLogger
from brain.prompt import PromptBuilder
from brain.state import BrainStateMachine
from brain.telemetry import LatencyTracker, TurnMetrics
from brain.tools.registry import ToolRegistry
from brain.turn import TurnManager


def _noop_publish(_topic: str, _payload: bytes, _qos: int = 1, _retain: bool = False) -> None:
    pass


class MockWake(WakeWordDetector):
    def __init__(self) -> None:
        pass

    def score_frame(self, frame: np.ndarray) -> float:
        _ = frame
        return 1.0

    def process_frame(self, frame: np.ndarray) -> bool:
        _ = frame
        return True


class MockVAD(VoiceActivityDetector):
    def __init__(self) -> None:
        pass

    def is_speech(self, frame: np.ndarray) -> bool:
        _ = frame
        return True

    def collect_utterance(self, source: AudioSource) -> bytes | None:
        _ = source
        return b"\x00" * 3200


class MockSTT(SpeechToText):
    def __init__(self) -> None:
        pass

    def transcribe(self, pcm: bytes) -> STTResult | None:
        _ = pcm
        return STTResult(text="Hello BMO", latency_ms=15.0, confidence=0.99)


class MockLLM(LLMClient):
    def __init__(self) -> None:
        pass

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        cancel_event: threading.Event | None = None,
    ):
        _ = (messages, tools, stream, cancel_event)
        yield ChatChunk(text="Hello friend!")


class MockTTS(TextToSpeech):
    def __init__(self) -> None:
        pass

    def speak_sentence(
        self,
        text: str,
        sink: AudioSink,
        cancel_event: threading.Event | None = None,
    ) -> None:
        _ = (text, cancel_event)
        if hasattr(sink, "write_samples"):
            sink.write_samples(b"\x00" * 640)
        elif hasattr(sink, "write"):
            sink.write(b"\x00" * 640)

    def cancel(self) -> None:
        pass


class MockPromptBuilder(PromptBuilder):
    def __init__(self) -> None:
        pass

    def build_messages(self, user_text: str) -> list[dict[str, Any]]:
        _ = user_text
        return [{"role": "user", "content": user_text}]

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        _ = (user_text, assistant_text)


def test_latency_tracker_basic_logging(tmp_path: Path) -> None:
    log_file = tmp_path / "telemetry.jsonl"
    logger = JSONLLogger(log_file)
    tracker = LatencyTracker(logger=logger, clock=lambda: 1000.0)

    metrics_input = {
        "vad_end_ms": 25.0,
        "stt_ms": 110.0,
        "llm_ttft_ms": 180.0,
        "tts_first_chunk_ms": 95.0,
        "total_ms": 410.0,
    }
    recorded = tracker.record_turn(metrics_input, utt_id="turn-1")

    assert recorded.vad_end_ms == 25.0
    assert recorded.stt_ms == 110.0
    assert recorded.llm_ttft_ms == 180.0
    assert recorded.tts_first_chunk_ms == 95.0
    assert recorded.total_ms == 410.0
    assert recorded.utt_id == "turn-1"
    assert recorded.ts == 1000.0

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    logged_dict = json.loads(lines[0])
    assert logged_dict["event"] == "turn_latency"
    for field in (
        "vad_end_ms",
        "stt_ms",
        "llm_ttft_ms",
        "tts_first_chunk_ms",
        "total_ms",
        "utt_id",
        "ts",
    ):
        assert field in logged_dict


def test_turn_metrics_fallback_keys() -> None:
    tracker = LatencyTracker()
    recorded_e2e = tracker.record_turn({"e2e_voice_ms": 320.0})
    assert recorded_e2e.total_ms == 320.0

    recorded_turn = tracker.record_turn({"turn_ms": 240.0})
    assert recorded_turn.total_ms == 240.0

    metric_obj = TurnMetrics(total_ms=150.0)
    recorded_obj = tracker.record_turn(metric_obj, utt_id="test-obj")
    assert recorded_obj.total_ms == 150.0
    assert recorded_obj.utt_id == "test-obj"


def test_percentile_calculations_and_summary() -> None:
    tracker = LatencyTracker()
    assert tracker.p50() == 0.0
    assert tracker.p95() == 0.0

    empty_summary = tracker.summary()
    assert empty_summary["total_ms"]["count"] == 0.0

    for val in [100.0, 200.0, 300.0, 400.0, 500.0]:
        tracker.record_turn({"total_ms": val, "stt_ms": val / 2})

    assert tracker.p50("total_ms") == 300.0
    assert tracker.p95("total_ms") == pytest.approx(480.0)
    assert tracker.p50("stt_ms") == 150.0

    s = tracker.summary()
    assert s["total_ms"]["p50"] == 300.0
    assert s["total_ms"]["min"] == 100.0
    assert s["total_ms"]["max"] == 500.0
    assert s["total_ms"]["count"] == 5.0
    assert len(tracker.records) == 5


def test_turn_manager_integration_20_turns() -> None:
    sm = BrainStateMachine(publish_fn=_noop_publish)
    wake = MockWake()
    vad = MockVAD()
    stt = MockSTT()
    llm = MockLLM()
    prompt_builder = MockPromptBuilder()
    tools = ToolRegistry()
    tts = MockTTS()
    sink = MemorySink()

    tracker = LatencyTracker()
    manager = TurnManager(
        state_machine=sm,
        wake=wake,
        vad=vad,
        stt=stt,
        llm=llm,
        prompt_builder=prompt_builder,
        tools=tools,
        tts=tts,
        sink=sink,
        on_turn_metrics=tracker.on_turn_metrics,
    )

    for i in range(20):
        source = MemorySource(pcm=b"\x00" * 640)
        reply = manager.listen_and_respond(source)
        assert reply == "Hello friend!"
        assert len(tracker.records) == i + 1

    records = tracker.records
    assert len(records) == 20
    for r in records:
        assert r.total_ms > 0.0
        assert r.stt_ms > 0.0

    p50_total = tracker.p50("total_ms")
    p95_total = tracker.p95("total_ms")
    assert p50_total <= 2500.0
    assert p95_total <= 2500.0

    summary = tracker.summary()
    assert summary["total_ms"]["count"] == 20.0
    assert summary["total_ms"]["p50"] <= 2500.0


@pytest.mark.gpu
@pytest.mark.slow
def test_real_model_latency_pipeline() -> None:
    pytest.skip("Real model GPU evaluation skipped when model weights are not loaded")
